"""Temporal patterns: *when* usage happens, not just how much.

:mod:`token_usage.analytics` aggregates by day/week/month, and
:mod:`token_usage.insights` asks whether a value is unusual. Neither answers the
questions that drive scheduling and capacity decisions:

* which weekday is most expensive, and is the weekend actually cheaper?
* which hours of the day carry the load?
* is spend trending up or down, and how strong is that trend?
* are there runs of consecutive over-budget days?
* is usage steady, variable, or bursty?

All functions are pure: records (+ catalog) in, plain dicts out. No I/O.

Timezone caveat
---------------
Timestamps are stored in UTC, so hour-of-day buckets are **UTC hours**. For a
team in UTC+8 the "busiest hour" will appear eight hours earlier than the wall
clock they recognise. This is deliberate -- shifting would require knowing each
user's zone, and a wrong shift is worse than an honest UTC label.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from token_usage import analytics
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import parse_iso, round_usd

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WEEKEND_INDEXES = (5, 6)  # Saturday, Sunday


def _empty_bucket(label: str) -> Dict:
    return {"label": label, "records": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0}


def _accumulate(bucket: Dict, record: UsageRecord, cost: float) -> None:
    bucket["records"] += 1
    bucket["input_tokens"] += record.input_tokens
    bucket["output_tokens"] += record.output_tokens
    bucket["total_tokens"] += record.total_tokens
    bucket["cost_usd"] = round_usd(bucket["cost_usd"] + cost)


def hourly_profile(records: Iterable[UsageRecord],
                   catalog: PricingCatalog) -> List[Dict]:
    """Cost and volume per UTC hour of day, always 24 buckets.

    Zero-filled so the result has a stable shape -- a chart or a caller indexing
    ``profile[14]`` should not have to handle missing hours.
    """
    buckets = [_empty_bucket(f"{hour:02d}:00") for hour in range(24)]
    for record in records:
        hour = parse_iso(record.timestamp).hour
        _accumulate(buckets[hour], record, catalog.cost_for_record(record))
    return buckets


def weekday_profile(records: Iterable[UsageRecord],
                    catalog: PricingCatalog) -> List[Dict]:
    """Cost and volume per weekday, Monday first, always 7 buckets."""
    buckets = [_empty_bucket(name) for name in WEEKDAY_NAMES]
    for record in records:
        index = parse_iso(record.timestamp).weekday()
        _accumulate(buckets[index], record, catalog.cost_for_record(record))
    return buckets


def peak_buckets(profile: Sequence[Dict], top: int = 3) -> List[Dict]:
    """The ``top`` most expensive buckets, ignoring empty ones."""
    populated = [bucket for bucket in profile if bucket["records"] > 0]
    return sorted(populated, key=lambda b: b["cost_usd"], reverse=True)[:top]


def quiet_buckets(profile: Sequence[Dict], top: int = 3) -> List[Dict]:
    """The ``top`` cheapest *populated* buckets.

    Empty buckets are excluded: "no usage at 3am" is not an insight, whereas
    "the 3am calls we do make are cheap" might be.
    """
    populated = [bucket for bucket in profile if bucket["records"] > 0]
    return sorted(populated, key=lambda b: b["cost_usd"])[:top]


def weekend_split(records: Iterable[UsageRecord],
                  catalog: PricingCatalog) -> Dict:
    """Compare weekday against weekend spend, normalised per day.

    Per-day averages matter here: a month has ~22 weekdays and ~8 weekend days,
    so comparing raw totals would always favour weekdays regardless of intensity.
    """
    weekday_cost = weekend_cost = 0.0
    weekday_days: set = set()
    weekend_days: set = set()
    weekday_records = weekend_records = 0

    for record in records:
        dt = parse_iso(record.timestamp)
        cost = catalog.cost_for_record(record)
        day = dt.strftime("%Y-%m-%d")
        if dt.weekday() in WEEKEND_INDEXES:
            weekend_cost += cost
            weekend_days.add(day)
            weekend_records += 1
        else:
            weekday_cost += cost
            weekday_days.add(day)
            weekday_records += 1

    weekday_per_day = weekday_cost / len(weekday_days) if weekday_days else 0.0
    weekend_per_day = weekend_cost / len(weekend_days) if weekend_days else 0.0
    ratio = (weekend_per_day / weekday_per_day) if weekday_per_day else 0.0
    return {
        "weekday_cost_usd": round_usd(weekday_cost),
        "weekend_cost_usd": round_usd(weekend_cost),
        "weekday_days": len(weekday_days),
        "weekend_days": len(weekend_days),
        "weekday_per_day_usd": round_usd(weekday_per_day),
        "weekend_per_day_usd": round_usd(weekend_per_day),
        "weekend_ratio": round(ratio, 4),
        "weekday_records": weekday_records,
        "weekend_records": weekend_records,
    }


def moving_average(daily: Dict[str, Dict], window: int = 7) -> List[Dict]:
    """Trailing simple moving average over the daily cost series.

    Trailing (not centred) because the point is to see what the recent level is,
    using only data that would have been available at the time. The first
    ``window - 1`` points average over fewer days rather than being dropped, so
    the series keeps its full length and stays aligned with the raw values.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    ordered = sorted(daily.items())
    series = []
    for index, (day, payload) in enumerate(ordered):
        start = max(0, index - window + 1)
        window_values = [ordered[i][1]["cost_usd"] for i in range(start, index + 1)]
        series.append({
            "day": day,
            "cost_usd": payload["cost_usd"],
            "average_usd": round_usd(sum(window_values) / len(window_values)),
            "samples": len(window_values),
        })
    return series


def trend_direction(daily: Dict[str, Dict],
                    window: Optional[int] = None) -> Dict:
    """Least-squares slope of daily cost, expressed as a percentage per day.

    ``r_squared`` says how much of the variation the line explains; a steep slope
    with a low r-squared is noise, not a trend. ``direction`` is ``stable`` when
    the slope is under 1% of the mean per day, which keeps trivial drift from
    being reported as a trend.
    """
    ordered = sorted(daily.items())
    if window:
        ordered = ordered[-window:]
    values = [payload["cost_usd"] for _, payload in ordered]
    n = len(values)
    if n < 2:
        return {"direction": "stable", "slope_usd_per_day": 0.0,
                "pct_per_day": 0.0, "r_squared": 0.0, "samples": n,
                "mean_usd": round_usd(values[0]) if values else 0.0}

    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    numerator = sum((i - mean_x) * (value - mean_y) for i, value in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    slope = numerator / denominator if denominator else 0.0

    ss_total = sum((value - mean_y) ** 2 for value in values)
    ss_residual = sum((value - (mean_y + slope * (i - mean_x))) ** 2
                      for i, value in enumerate(values))
    r_squared = 1.0 - (ss_residual / ss_total) if ss_total else 0.0

    pct_per_day = (slope / mean_y * 100.0) if mean_y else 0.0
    if abs(pct_per_day) < 1.0:
        direction = "stable"
    else:
        direction = "rising" if slope > 0 else "falling"
    return {"direction": direction, "slope_usd_per_day": round_usd(slope),
            "pct_per_day": round(pct_per_day, 2), "r_squared": round(r_squared, 4),
            "samples": n, "mean_usd": round_usd(mean_y)}


def streaks(daily: Dict[str, Dict]) -> Dict:
    """Longest and current runs of consecutive days above/below the mean.

    A run of expensive days is a different signal from a single spike: it points
    at a sustained change (a new workload, a runaway job) rather than an outlier.
    """
    ordered = sorted(daily.items())
    values = [payload["cost_usd"] for _, payload in ordered]
    if not values:
        return {"mean_usd": 0.0, "longest_above": 0, "longest_below": 0,
                "current_above": 0, "current_below": 0, "current_direction": "none"}

    mean = sum(values) / len(values)
    longest_above = longest_below = 0
    run_above = run_below = 0
    for value in values:
        if value > mean:
            run_above += 1
            run_below = 0
            longest_above = max(longest_above, run_above)
        elif value < mean:
            run_below += 1
            run_above = 0
            longest_below = max(longest_below, run_below)
        else:  # exactly at the mean breaks both runs
            run_above = run_below = 0

    if run_above:
        current_direction, current_above, current_below = "above", run_above, 0
    elif run_below:
        current_direction, current_above, current_below = "below", 0, run_below
    else:
        current_direction, current_above, current_below = "none", 0, 0

    return {"mean_usd": round_usd(mean), "longest_above": longest_above,
            "longest_below": longest_below, "current_above": current_above,
            "current_below": current_below, "current_direction": current_direction}


def burstiness(daily: Dict[str, Dict]) -> Dict:
    """How uneven the daily series is, via coefficient of variation.

    ``cv`` is stdev/mean: 0 means perfectly flat, 1 means the standard deviation
    equals the mean. Classification thresholds are conventional rather than
    universal -- they are here to make the number readable, not authoritative.
    """
    values = [payload["cost_usd"] for _, payload in sorted(daily.items())]
    if len(values) < 2:
        return {"cv": 0.0, "mean_usd": round_usd(values[0]) if values else 0.0,
                "stdev_usd": 0.0, "classification": "insufficient-data",
                "samples": len(values)}
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    cv = (stdev / mean) if mean else 0.0
    if cv < 0.3:
        classification = "steady"
    elif cv < 0.7:
        classification = "variable"
    else:
        classification = "bursty"
    return {"cv": round(cv, 4), "mean_usd": round_usd(mean),
            "stdev_usd": round_usd(stdev), "classification": classification,
            "samples": len(values)}


def seasonality_strength(profile: Sequence[Dict]) -> Dict:
    """How much the weekday pattern matters, as a max/min ratio.

    A ratio near 1 means every weekday costs about the same, so weekday-aware
    scheduling buys nothing. A large ratio means the pattern is worth exploiting.
    """
    populated = [bucket for bucket in profile if bucket["records"] > 0]
    if len(populated) < 2:
        return {"ratio": 1.0, "strongest": None, "weakest": None,
                "strength": "insufficient-data"}
    strongest = max(populated, key=lambda b: b["cost_usd"])
    weakest = min(populated, key=lambda b: b["cost_usd"])
    if weakest["cost_usd"] <= 0:
        ratio = float("inf")
    else:
        ratio = strongest["cost_usd"] / weakest["cost_usd"]
    if ratio >= 3.0:
        strength = "strong"
    elif ratio >= 1.5:
        strength = "moderate"
    else:
        strength = "weak"
    return {"ratio": round(ratio, 4) if ratio != float("inf") else None,
            "strongest": strongest["label"], "weakest": weakest["label"],
            "strength": strength}


def daily_series(records: Iterable[UsageRecord],
                 catalog: PricingCatalog) -> Dict[str, Dict]:
    """Chronological per-day totals -- the input every function here expects."""
    return analytics.by_day(records, catalog)


def summarize(records: Iterable[UsageRecord], catalog: PricingCatalog,
              window: int = 7) -> Dict:
    """One-shot pattern summary, used by the ``pattern`` plugin's default view."""
    records = list(records)
    daily = daily_series(records, catalog)
    weekdays = weekday_profile(records, catalog)
    hours = hourly_profile(records, catalog)
    return {
        "days": len(daily),
        "weekday_profile": weekdays,
        "hourly_profile": hours,
        "weekend": weekend_split(records, catalog),
        "trend": trend_direction(daily, window=window),
        "streaks": streaks(daily),
        "burstiness": burstiness(daily),
        "seasonality": seasonality_strength(weekdays),
        "peak_weekdays": peak_buckets(weekdays, top=3),
        "peak_hours": peak_buckets(hours, top=3),
    }
