"""Derived insights: anomalies, period comparison, model efficiency.

Where :mod:`token_usage.analytics` answers "how much", this module answers
"is that normal", "how does it compare", and "which model gives the most
value". Same contract as analytics: pure functions, plain dicts out.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from token_usage import analytics
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import parse_iso, period_start, round_usd

COMPARABLE_PERIODS = ("daily", "weekly", "monthly")


def detect_anomalies(records: Iterable[UsageRecord], catalog: PricingCatalog,
                     z_threshold: float = 2.0, min_samples: int = 5) -> List[Dict]:
    """Flag days whose cost deviates from the mean by >= ``z_threshold`` sigma.

    Uses the population standard deviation. Returns ``[]`` when there are fewer
    than ``min_samples`` days or when every day costs the same (zero variance),
    because a z-score is meaningless in both cases.
    """
    daily = analytics.by_day(records, catalog)
    if len(daily) < min_samples:
        return []
    days = list(daily.items())
    values = [payload["cost_usd"] for _, payload in days]
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return []
    flagged = []
    for day, payload in days:
        z = (payload["cost_usd"] - mean) / stdev
        if abs(z) >= z_threshold:
            flagged.append({"day": day, "cost_usd": payload["cost_usd"],
                            "mean_usd": round_usd(mean), "z_score": round(z, 2),
                            "direction": "spike" if z > 0 else "dip"})
    return sorted(flagged, key=lambda item: abs(item["z_score"]), reverse=True)


def compare_periods(records: Iterable[UsageRecord], catalog: PricingCatalog,
                    period: str = "monthly",
                    now: Optional[datetime] = None) -> Dict:
    """Compare the current period against the immediately preceding one.

    The previous window has the same length as the elapsed part of the current
    period, so a mid-month comparison is apples-to-apples rather than comparing
    half a month against a full one.
    """
    if period not in COMPARABLE_PERIODS:
        raise ValueError(f"period must be one of {COMPARABLE_PERIODS}, got {period!r}")
    now = now or datetime.now(timezone.utc)
    current_start = period_start(period, now=now)
    previous_start = current_start - (now - current_start)

    records = list(records)
    current = [r for r in records if parse_iso(r.timestamp) >= current_start]
    previous = [r for r in records
                if previous_start <= parse_iso(r.timestamp) < current_start]

    cur = analytics.totals(current, catalog)
    prev = analytics.totals(previous, catalog)
    delta = cur["cost_usd"] - prev["cost_usd"]
    pct = (delta / prev["cost_usd"] * 100.0) if prev["cost_usd"] else 0.0
    return {"period": period, "current": cur, "previous": prev,
            "delta_usd": round_usd(delta), "delta_pct": round(pct, 2),
            "direction": "flat" if abs(delta) < 1e-9 else ("up" if delta > 0 else "down")}


def model_efficiency(records: Iterable[UsageRecord],
                     catalog: PricingCatalog) -> List[Dict]:
    """Per-model unit economics, cheapest cost-per-1K-tokens first.

    ``output_ratio`` is the share of tokens that were generated (rather than
    read), and ``cache_ratio`` the share of input served from cache -- both are
    useful when deciding whether a model is being used appropriately.
    """
    rows = []
    for model, payload in analytics.by_model(records, catalog).items():
        tokens = payload["total_tokens"]
        cost = payload["cost_usd"]
        rows.append({
            "model": model,
            "records": payload["records"],
            "total_tokens": tokens,
            "cost_usd": cost,
            "cost_per_1k_tokens": round_usd(cost / (tokens / 1000.0)) if tokens else 0.0,
            "output_ratio": round(payload["output_tokens"] / tokens, 4) if tokens else 0.0,
            "cache_ratio": (round(payload["cached_input_tokens"] / payload["input_tokens"], 4)
                            if payload["input_tokens"] else 0.0),
        })
    return sorted(rows, key=lambda row: row["cost_per_1k_tokens"])


def concentration(records: Iterable[UsageRecord], catalog: PricingCatalog,
                  dimension: str = "model", top: int = 3) -> Dict:
    """Measure how much of total spend the top-N buckets account for.

    A high ``top_share`` means spend is concentrated in a few models/projects,
    which is where optimisation effort pays off first.
    """
    grouped = analytics.group(records, dimension, catalog)
    total = sum(payload["cost_usd"] for payload in grouped.values())
    if total <= 0:
        return {"dimension": dimension, "total_usd": 0.0, "top": [], "top_share": 0.0}
    items = list(grouped.items())[:top]
    top_cost = sum(payload["cost_usd"] for _, payload in items)
    return {
        "dimension": dimension,
        "total_usd": round_usd(total),
        "top": [{"name": name, "cost_usd": payload["cost_usd"],
                 "share": round(payload["cost_usd"] / total, 4)}
                for name, payload in items],
        "top_share": round(top_cost / total, 4),
    }


def waste_estimate(records: Iterable[UsageRecord], catalog: PricingCatalog,
                   min_output_ratio: float = 0.05) -> Dict:
    """Estimate spend on calls that produced almost no output.

    A very low output ratio usually means a prompt was sent but the response was
    truncated, errored, or discarded -- work that was paid for but not used.
    """
    wasted = 0.0
    offenders = []
    for record in records:
        if record.total_tokens == 0:
            continue
        ratio = record.output_tokens / record.total_tokens
        if ratio < min_output_ratio:
            cost = catalog.cost_for_record(record)
            wasted += cost
            offenders.append({"model": record.model, "timestamp": record.timestamp,
                              "output_ratio": round(ratio, 4), "cost_usd": cost})
    return {"wasted_usd": round_usd(wasted), "count": len(offenders),
            "offenders": sorted(offenders, key=lambda o: o["cost_usd"], reverse=True)[:10]}
