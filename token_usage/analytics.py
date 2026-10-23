"""Aggregation and derived statistics over usage records.

Pure functions only: records (+ catalog where cost is needed) in, plain dicts
out. No I/O, so everything is testable and safe to call from plugins.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import (
    day_key,
    month_key,
    parse_date,
    parse_iso,
    period_start,
    round_usd,
    week_key,
)


def _parse_bound(value: str, end_of_day: bool = False) -> datetime:
    """Parse a full ISO timestamp or a bare ``YYYY-MM-DD`` date bound."""
    if "T" in value:
        return parse_iso(value)
    dt = parse_date(value)
    return dt.replace(hour=23, minute=59, second=59) if end_of_day else dt


def filter_records(records: Iterable[UsageRecord], project: Optional[str] = None,
                   model: Optional[str] = None, tag: Optional[str] = None,
                   since: Optional[str] = None, until: Optional[str] = None) -> List[UsageRecord]:
    """Return records matching every supplied predicate; bounds are inclusive."""
    since_dt = _parse_bound(since) if since else None
    until_dt = _parse_bound(until, end_of_day=True) if until else None
    result = []
    for record in records:
        if project is not None and record.project != project:
            continue
        if model is not None and record.model != model:
            continue
        if tag is not None and tag not in record.tags:
            continue
        ts = parse_iso(record.timestamp)
        if since_dt is not None and ts < since_dt:
            continue
        if until_dt is not None and ts > until_dt:
            continue
        result.append(record)
    return result


def filter_by_period(records: Iterable[UsageRecord], period: str,
                     now: Optional[datetime] = None) -> List[UsageRecord]:
    """Keep records at or after the start of ``period``."""
    start = period_start(period, now=now)
    return [r for r in records if parse_iso(r.timestamp) >= start]


def totals(records: Iterable[UsageRecord], catalog: PricingCatalog) -> Dict:
    """Aggregate token counts and cost across records."""
    input_tokens = output_tokens = cached_tokens = 0
    cost = 0.0
    per_model: Dict[str, int] = defaultdict(int)
    count = 0
    for record in records:
        count += 1
        input_tokens += record.input_tokens
        output_tokens += record.output_tokens
        cached_tokens += record.cached_input_tokens
        per_model[record.model] += record.total_tokens
        cost += catalog.cost_for_record(record)
    return {"records": count, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cached_input_tokens": cached_tokens, "total_tokens": input_tokens + output_tokens,
            "cost_usd": round_usd(cost), "models_used": len(per_model)}


def group_by(records: Iterable[UsageRecord], key_func,
             catalog: PricingCatalog) -> Dict[str, Dict]:
    """Group records by ``key_func``, ordered by descending cost."""
    buckets: Dict[str, List[UsageRecord]] = defaultdict(list)
    for record in records:
        buckets[key_func(record)].append(record)
    result = {name: totals(items, catalog) for name, items in buckets.items()}
    return dict(sorted(result.items(), key=lambda kv: kv[1]["cost_usd"], reverse=True))


def by_model(records, catalog) -> Dict[str, Dict]:
    return group_by(records, lambda r: r.model, catalog)


def by_project(records, catalog) -> Dict[str, Dict]:
    return group_by(records, lambda r: r.project, catalog)


def by_tag(records, catalog) -> Dict[str, Dict]:
    """Group by tag; a multi-tag record contributes to each of its tags."""
    expanded: List[UsageRecord] = []
    for record in records:
        if not record.tags:
            expanded.append(record)
            continue
        for tag in record.tags:
            expanded.append(UsageRecord(
                model=record.model, input_tokens=record.input_tokens,
                output_tokens=record.output_tokens, timestamp=record.timestamp,
                cached_input_tokens=record.cached_input_tokens, project=record.project,
                tags=[tag], note=record.note,
                cost_usd=catalog.cost_for_record(record) if len(record.tags) == 1 else None))
    return group_by(expanded, lambda r: r.tags[0] if r.tags else "(untagged)", catalog)


def by_day(records, catalog) -> Dict[str, Dict]:
    """Group by calendar day, chronological (oldest first) for time series."""
    return dict(sorted(group_by(records, lambda r: day_key(parse_iso(r.timestamp)), catalog).items()))


def by_month(records, catalog) -> Dict[str, Dict]:
    return dict(sorted(group_by(records, lambda r: month_key(parse_iso(r.timestamp)), catalog).items()))


def by_week(records, catalog) -> Dict[str, Dict]:
    return dict(sorted(group_by(records, lambda r: week_key(parse_iso(r.timestamp)), catalog).items()))


GROUPINGS = {"model": by_model, "project": by_project, "tag": by_tag,
             "day": by_day, "week": by_week, "month": by_month}


def group(records: Iterable[UsageRecord], dimension: str,
          catalog: PricingCatalog) -> Dict[str, Dict]:
    if dimension not in GROUPINGS:
        raise ValueError(f"unknown grouping {dimension!r}; expected one of {sorted(GROUPINGS)}")
    return GROUPINGS[dimension](records, catalog)


def cache_savings(records: Iterable[UsageRecord], catalog: PricingCatalog) -> Dict:
    """Estimate money saved by caching, versus billing cache reads at input rate."""
    saved = 0.0
    cached_tokens = 0
    for record in records:
        pricing = catalog.get(record.model)
        if pricing is None or record.cached_input_tokens == 0:
            continue
        cached_tokens += record.cached_input_tokens
        factor = record.cached_input_tokens / 1_000_000.0
        saved += factor * (pricing.input_per_1m - pricing.cached_input_per_1m)
    return {"cached_input_tokens": cached_tokens, "saved_usd": round_usd(max(0.0, saved))}


def average_cost_per_call(records: Iterable[UsageRecord],
                          catalog: PricingCatalog) -> float:
    total = totals(records, catalog)
    return round_usd(total["cost_usd"] / total["records"]) if total["records"] else 0.0


def linear_forecast(daily: Dict[str, Dict], days_ahead: int) -> Dict:
    """Project spend forward with a least-squares fit on per-day cost."""
    points = [(float(i), float(p["cost_usd"]))
              for i, (_, p) in enumerate(sorted(daily.items()))]
    n = len(points)
    if n == 0:
        return {"slope": 0.0, "intercept": 0.0, "projected_total": 0.0,
                "projected_daily": 0.0, "samples": 0}
    if n == 1:
        only = points[0][1]
        return {"slope": 0.0, "intercept": only,
                "projected_total": round_usd(only * days_ahead),
                "projected_daily": round_usd(only), "samples": 1}
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    slope = numerator / denominator if denominator else 0.0
    intercept = mean_y - slope * mean_x
    projected = sum(max(0.0, slope * (n + i) + intercept) for i in range(days_ahead))
    return {"slope": round_usd(slope), "intercept": round_usd(intercept),
            "projected_total": round_usd(projected),
            "projected_daily": round_usd(projected / days_ahead) if days_ahead else 0.0,
            "samples": n}


def trend(daily: Dict[str, Dict]) -> Dict:
    """Compare the most recent day's cost against the previous day's."""
    ordered = sorted(daily.items())
    if len(ordered) < 2:
        return {"direction": "flat", "delta_usd": 0.0, "delta_pct": 0.0}
    prev, curr = ordered[-2][1]["cost_usd"], ordered[-1][1]["cost_usd"]
    delta = curr - prev
    direction = "flat" if abs(delta) < 1e-9 else ("up" if delta > 0 else "down")
    return {"direction": direction, "delta_usd": round_usd(delta),
            "delta_pct": round((delta / prev * 100.0) if prev else 0.0, 2)}


def top_n(grouped: Dict[str, Dict], n: int) -> List[Tuple[str, Dict]]:
    return list(grouped.items())[:n]


def evaluate_budgets(records: Iterable[UsageRecord], catalog: PricingCatalog,
                     budgets: List, now: Optional[datetime] = None) -> List[Dict]:
    """Check each budget against spend in its period.

    Status is ``ok`` (<80%), ``warning`` (>=80%) or ``exceeded`` (>=100%).
    """
    records = list(records)
    results = []
    for budget in budgets:
        scoped = [r for r in records
                  if budget.project is None or r.project == budget.project]
        spent = totals(filter_by_period(scoped, budget.period, now=now), catalog)["cost_usd"]
        ratio = spent / budget.limit_usd if budget.limit_usd else 0.0
        results.append({
            "label": budget.label or (budget.project or "all"),
            "period": budget.period,
            "project": budget.project,
            "limit_usd": budget.limit_usd,
            "spent_usd": spent,
            "remaining_usd": round_usd(max(0.0, budget.limit_usd - spent)),
            "ratio": round(ratio, 4),
            "status": "exceeded" if ratio >= 1.0 else ("warning" if ratio >= 0.8 else "ok"),
        })
    return results
