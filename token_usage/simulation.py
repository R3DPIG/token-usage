"""Period-end simulation: a range, not a single number.

:mod:`token_usage.planning` projects period-end spend from the average daily
rate. That is a point estimate, and a point estimate cannot answer the question
a budget owner actually asks: *how likely am I to overshoot?*

This module answers that by resampling the observed daily costs. Two methods are
offered, because they fail in different ways and disagreement between them is
itself informative:

``bootstrap``
    Resample the observed daily costs with replacement. Makes no distributional
    assumption and preserves the real shape (including zero-spend days and
    spikes), but it can only ever produce values inside the historical range, so
    it understates risk when the period is trending upward.

``normal``
    Fit mean/stdev and draw from a normal distribution. Can exceed the observed
    range, so it says something about a trend, but it happily draws negative
    costs -- those are clamped to zero and the count is reported, because a
    simulation that quietly produces impossible values should not be trusted
    without knowing it did.

Both need the daily series **zero-filled**: a day with no calls is a real day
that cost nothing, and dropping it would inflate the mean and shrink the spread.

Seeded by default so a reported P90 can be reproduced exactly. Pure aside from
the RNG: records + catalog in, plain dicts out.
"""

from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from token_usage import analytics
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import day_key, parse_iso, period_start, round_usd

METHODS = ("bootstrap", "normal")
# Percentiles reported for every run. P50 answers "typical", P90/P95 answer
# "how bad could it get", and P10 is there so the downside is not hidden.
REPORTED_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def daily_costs(records: Iterable[UsageRecord], catalog: PricingCatalog,
                period: str = "monthly",
                now: Optional[datetime] = None) -> Dict:
    """Zero-filled daily cost series for the elapsed part of the period.

    Every calendar day from the period start through today appears, with 0.0 for
    days that had no usage. Callers get ``values`` in chronological order plus
    how many of them were zero, because a series that is mostly zeros behaves
    very differently from one with the same mean and no zeros.
    """
    now = now or datetime.now(timezone.utc)
    start = period_start(period, now=now)
    observed = analytics.by_day(analytics.filter_by_period(records, period,
                                                           now=now), catalog)

    days: List[str] = []
    cursor = start
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor <= today:
        days.append(day_key(cursor))
        cursor += timedelta(days=1)

    values = [observed.get(day, {}).get("cost_usd", 0.0) for day in days]
    return {
        "days": days,
        "values": values,
        "observed_days": len(observed),
        "zero_days": sum(1 for value in values if value == 0.0),
        "spent_usd": round_usd(sum(values)),
        "mean_usd": round_usd(statistics.mean(values)) if values else 0.0,
        "stdev_usd": round_usd(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def days_left(period: str, now: Optional[datetime] = None) -> int:
    """Days remaining after today, never negative and never zero.

    Zero would mean "no future at all", which would make every simulation equal
    the spend so far and the risk assessment meaningless. On the final day there
    is still spending ahead, so the floor is 1 -- consistent with
    :func:`token_usage.planning.days_remaining`.
    """
    now = now or datetime.now(timezone.utc)
    start = period_start(period, now=now)
    if period == "daily":
        total = 1
    elif period == "weekly":
        total = 7
    else:
        year, month = start.year, start.month
        nxt = start.replace(year=year + 1, month=1) if month == 12 \
            else start.replace(month=month + 1)
        total = (nxt - start).days
    elapsed = (now - start).days + 1
    return max(1, total - elapsed)


def _draw_bootstrap(values: List[float], horizon: int, rng: random.Random) -> float:
    return sum(rng.choice(values) for _ in range(horizon))


def _draw_normal(mean: float, stdev: float, horizon: int,
                 rng: random.Random) -> Dict:
    """Sum of ``horizon`` normal draws; negative days are clamped and counted.

    Clamping rather than discarding keeps the horizon honest: a day that would
    have cost -$3 is recorded as $0, so the total is not inflated by dropping
    a whole day out of the period.
    """
    total = 0.0
    clamped = 0
    for _ in range(horizon):
        draw = rng.gauss(mean, stdev)
        if draw < 0:
            clamped += 1
            draw = 0.0
        total += draw
    return {"total": total, "clamped": clamped}


def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list.

    Interpolation rather than nearest-rank: with a few hundred samples the
    nearest-rank P90 jumps in visible steps as the seed changes, which reads as
    instability rather than sampling noise.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (pct / 100.0)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] + (values[high] - values[low]) * fraction


def simulate(records: Iterable[UsageRecord], catalog: PricingCatalog,
             period: str = "monthly", method: str = "bootstrap",
             iterations: int = 2000, limit_usd: float = 0.0,
             seed: int = 7,
             now: Optional[datetime] = None) -> Dict:
    """Simulate period-end totals and report the distribution.

    ``limit_usd`` of 0 means no limit was supplied; ``exceed_probability`` is
    then ``None`` rather than 0, because "0% chance of overspending" and "no
    budget to compare against" are different statements.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    series = daily_costs(records, catalog, period=period, now=now)
    values = series["values"]
    horizon = days_left(period, now=now)
    spent = series["spent_usd"]
    mean = series["mean_usd"]
    stdev = series["stdev_usd"]
    rng = random.Random(seed)

    if not values or all(value == 0.0 for value in values):
        # Nothing observed: every future day is zero by the only assumption the
        # data supports. Reporting spread here would be invention.
        return {**series, "method": method, "period": period, "seed": seed,
                "iterations": iterations, "horizon_days": horizon,
                "spent_usd": spent, "limit_usd": limit_usd,
                "percentiles": {p: round_usd(spent) for p in REPORTED_PERCENTILES},
                "mean_total_usd": round_usd(spent),
                "p90_total_usd": round_usd(spent),
                "exceed_probability": (0.0 if limit_usd > 0 else None),
                "clamped_days": 0, "insufficient_data": True}

    totals: List[float] = []
    clamped_days = 0
    for _ in range(iterations):
        if method == "bootstrap":
            future = _draw_bootstrap(values, horizon, rng)
        else:
            drawn = _draw_normal(mean, stdev, horizon, rng)
            future = drawn["total"]
            clamped_days += drawn["clamped"]
        totals.append(spent + future)

    totals.sort()
    counts = {p: percentile(totals, p) for p in REPORTED_PERCENTILES}
    exceed = (sum(1 for total in totals if total > limit_usd) / len(totals)
              if limit_usd > 0 else None)

    return {
        **series,
        "method": method,
        "period": period,
        "seed": seed,
        "iterations": iterations,
        "horizon_days": horizon,
        "limit_usd": limit_usd,
        "percentiles": {p: round_usd(v) for p, v in counts.items()},
        "mean_total_usd": round_usd(statistics.mean(totals)),
        "median_total_usd": round_usd(counts[50]),
        "p90_total_usd": round_usd(counts[90]),
        "exceed_probability": (round(exceed, 4) if exceed is not None else None),
        "clamped_days": clamped_days,
        "insufficient_data": False,
    }


def compare_methods(records: Iterable[UsageRecord], catalog: PricingCatalog,
                    period: str = "monthly", limit_usd: float = 0.0,
                    iterations: int = 2000, seed: int = 7,
                    now: Optional[datetime] = None) -> Dict:
    """Run both methods and report where they disagree.

    A wide gap means the answer depends on assumptions rather than on data -- a
    trending or spiky series makes bootstrap and normal diverge. That is worth
    surfacing instead of picking one method and presenting its output as fact.
    """
    runs = {method: simulate(records, catalog, period=period, method=method,
                             iterations=iterations, limit_usd=limit_usd,
                             seed=seed, now=now)
            for method in METHODS}
    boot_p90 = runs["bootstrap"]["p90_total_usd"]
    norm_p90 = runs["normal"]["p90_total_usd"]
    spread = abs(norm_p90 - boot_p90)
    reference = max(abs(boot_p90), abs(norm_p90), 1e-9)
    return {"runs": runs,
            "p90_spread_usd": round_usd(spread),
            "p90_spread_pct": round(spread / reference * 100.0, 2),
            "agreement": ("close" if spread / reference < 0.10 else
                          "divergent"),
            "period": period}


def required_cut(projected_usd: float, limit_usd: float) -> Dict:
    """How much spending must fall to hit a limit, in dollars and percent."""
    if limit_usd <= 0:
        return {"limit_usd": limit_usd, "overage_usd": 0.0,
                "cut_usd": 0.0, "cut_pct": 0.0, "feasible": False,
                "note": "no limit supplied"}
    overage = max(0.0, projected_usd - limit_usd)
    cut_pct = (overage / projected_usd * 100.0) if projected_usd > 0 else 0.0
    return {"limit_usd": limit_usd, "overage_usd": round_usd(overage),
            "cut_usd": round_usd(overage), "cut_pct": round(cut_pct, 2),
            # Above 50% the plan is no longer a trim; say so rather than
            # presenting an impossible number as an actionable target.
            "feasible": cut_pct <= 50.0,
            "note": "" if cut_pct <= 50.0 else
                    "a cut this large is a capability change, not a tuning change"}


def summarize(records: Iterable[UsageRecord], catalog: PricingCatalog,
              period: str = "monthly", limit_usd: float = 0.0,
              iterations: int = 2000, seed: int = 7,
              now: Optional[datetime] = None) -> Dict:
    """One-shot view used by the plugin's default output."""
    data = simulate(records, catalog, period=period, method="bootstrap",
                    iterations=iterations, limit_usd=limit_usd, seed=seed, now=now)
    data["cut"] = required_cut(data["p90_total_usd"], limit_usd)
    data["methods"] = compare_methods(records, catalog, period=period,
                                      limit_usd=limit_usd,
                                      iterations=iterations, seed=seed, now=now)
    return data
