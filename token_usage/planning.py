"""Budget planning: turn history into forward-looking decisions.

The other modules answer "what happened" (``analytics``), "is it normal"
(``insights``) and "did I cross a line" (``alerts``). None of them answer the
question a user actually asks mid-month: *at this pace, where will I land, and
what should I do about it?*

This module closes that gap. It combines the linear projection from
:mod:`token_usage.insights` with period arithmetic and budget limits to produce:

* a projected period-end total and whether it breaches a limit,
* a per-day allowance for the remaining days,
* the reduction needed to stay inside the limit,
* a quota split that distributes a budget across models/projects by history,
* a what-if scenario for shifting usage on one model.

Pure functions only: records + catalog + clock in, plain dicts out.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from token_usage import analytics, insights
from token_usage.models import Budget, UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import parse_iso, period_start, round_usd

PLANNABLE_PERIODS = ("daily", "weekly", "monthly")


def period_bounds(period: str, now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """Return the inclusive start and exclusive end of the current period.

    ``weekly`` runs Monday to Monday (ISO), ``monthly`` from the 1st to the 1st
    of the next month. ``daily`` is midnight to midnight.
    """
    if period not in PLANNABLE_PERIODS:
        raise ValueError(f"period must be one of {PLANNABLE_PERIODS}, got {period!r}")
    now = now or datetime.now(timezone.utc)
    start = period_start(period, now=now)
    if period == "daily":
        end = start + timedelta(days=1)
    elif period == "weekly":
        end = start + timedelta(days=7)
    else:  # monthly -- advance the month, handling December rollover
        year, month = start.year, start.month
        end = start.replace(year=year + 1, month=1) if month == 12 \
            else start.replace(month=month + 1)
    return start, end


def days_remaining(period: str, now: Optional[datetime] = None) -> int:
    """Whole days left in the period, excluding today.

    Defined as ``period_length - days_elapsed`` so the two figures always sum to
    the period length. Counting today in both would double-count it and inflate
    every per-day allowance. Never returns 0: on the last day there is still a
    day of spending left, and a zero would divide by zero downstream.
    """
    now = now or datetime.now(timezone.utc)
    start, end = period_bounds(period, now=now)
    total_days = max(1, int((end - start).total_seconds() / 86400.0))
    return max(1, total_days - days_elapsed(period, now=now))


def days_elapsed(period: str, now: Optional[datetime] = None) -> int:
    """Whole days already elapsed in the period, counting today as 1."""
    now = now or datetime.now(timezone.utc)
    start, _ = period_bounds(period, now=now)
    return max(1, int((now - start).total_seconds() / 86400.0) + 1)


def project_period_end(records: Iterable[UsageRecord], catalog: PricingCatalog,
                       period: str = "monthly",
                       now: Optional[datetime] = None) -> Dict:
    """Project where this period will land, based on the pace so far.

    Uses the average daily spend observed in the period rather than a
    least-squares fit: within a single period there are usually too few days for
    a regression to beat a simple mean, and the mean is far more stable when a
    period has only two or three days of data.
    """
    now = now or datetime.now(timezone.utc)
    scoped = analytics.filter_by_period(list(records), period, now=now)
    spent = analytics.totals(scoped, catalog)["cost_usd"]
    elapsed = days_elapsed(period, now=now)
    remaining = days_remaining(period, now=now)
    daily_rate = spent / elapsed if elapsed else 0.0
    projected = spent + daily_rate * remaining
    return {
        "period": period,
        "spent_usd": spent,
        "daily_rate_usd": round_usd(daily_rate),
        "days_elapsed": elapsed,
        "days_remaining": remaining,
        "projected_total_usd": round_usd(projected),
        "records": len(scoped),
    }


def budget_outcome(records: Iterable[UsageRecord], catalog: PricingCatalog,
                   limit_usd: float, period: str = "monthly",
                   now: Optional[datetime] = None) -> Dict:
    """Combine the projection with a limit into a decision-ready verdict.

    ``verdict`` is one of:

    * ``on_track``   -- projected total stays inside the limit,
    * ``at_risk``    -- projected total exceeds the limit,
    * ``exceeded``   -- the limit is already breached,
    * ``no_budget``  -- the limit is not positive, so nothing can be judged.
    """
    projection = project_period_end(records, catalog, period=period, now=now)
    if limit_usd <= 0:
        return {**projection, "limit_usd": limit_usd, "verdict": "no_budget",
                "headroom_usd": 0.0, "overage_usd": 0.0,
                "daily_allowance_usd": 0.0, "required_reduction_pct": 0.0}

    spent = projection["spent_usd"]
    projected = projection["projected_total_usd"]
    remaining_days = projection["days_remaining"]
    headroom = max(0.0, limit_usd - spent)
    overage = max(0.0, projected - limit_usd)

    if spent >= limit_usd:
        verdict = "exceeded"
    elif projected > limit_usd:
        verdict = "at_risk"
    else:
        verdict = "on_track"

    # How much of the remaining budget may be spent per remaining day.
    allowance = headroom / remaining_days if remaining_days else 0.0
    # How far the projection must fall to fit inside the limit.
    reduction = (overage / projected * 100.0) if projected > 0 and overage > 0 else 0.0

    return {
        **projection,
        "limit_usd": limit_usd,
        "verdict": verdict,
        "headroom_usd": round_usd(headroom),
        "overage_usd": round_usd(overage),
        "daily_allowance_usd": round_usd(allowance),
        "required_reduction_pct": round(reduction, 2),
    }


def allocate_quota(total_usd: float, records: Iterable[UsageRecord],
                   catalog: PricingCatalog, dimension: str = "project",
                   floor_usd: float = 0.0) -> List[Dict]:
    """Split a budget across buckets in proportion to historical spend.

    ``floor_usd`` guarantees every bucket at least that much, which matters when
    a bucket has little history but is expected to grow. The floor is taken off
    the top and the remainder is shared proportionally, so the parts always sum
    to ``total_usd``.
    """
    if total_usd < 0:
        raise ValueError("total_usd must be >= 0")
    grouped = analytics.group(records, dimension, catalog)
    if not grouped:
        return []

    names = list(grouped)
    floor_total = floor_usd * len(names)
    if floor_total > total_usd:
        # Not enough budget to honour the floor everywhere; fall back to an
        # equal split rather than producing negative shares.
        share = total_usd / len(names)
        return [{"bucket": name, "cost_usd": grouped[name]["cost_usd"],
                 "share": round(1.0 / len(names), 4), "quota_usd": round_usd(share)}
                for name in names]

    distributable = total_usd - floor_total
    spent_total = sum(payload["cost_usd"] for payload in grouped.values())
    quotas = []
    for name in names:
        cost = grouped[name]["cost_usd"]
        weight = (cost / spent_total) if spent_total > 0 else 1.0 / len(names)
        quota = floor_usd + distributable * weight
        quotas.append({"bucket": name, "cost_usd": cost,
                       "share": round(weight, 4), "quota_usd": round_usd(quota)})
    return sorted(quotas, key=lambda item: item["quota_usd"], reverse=True)


def scenario_shift(records: Iterable[UsageRecord], catalog: PricingCatalog,
                   model: str, change_pct: float,
                   period: str = "monthly",
                   now: Optional[datetime] = None) -> Dict:
    """What-if: how does the projection move if one model's usage changes?

    ``change_pct`` is a signed percentage applied to that model's cost (``-30``
    means "cut this model's spend by 30%"). The delta is applied to the
    projection, not to history, so the answer reads as "where would I land".
    """
    now = now or datetime.now(timezone.utc)
    scoped = analytics.filter_by_period(list(records), period, now=now)
    baseline = project_period_end(scoped, catalog, period=period, now=now)

    model_cost = analytics.by_model(scoped, catalog).get(model, {}).get("cost_usd", 0.0)
    if model_cost == 0:
        return {**baseline, "model": model, "change_pct": change_pct,
                "model_cost_usd": 0.0, "delta_usd": 0.0,
                "adjusted_total_usd": baseline["projected_total_usd"],
                "note": f"no spend recorded for {model!r} in this period"}

    # Scale the model's period-to-date cost by the same factor the projection
    # applies to the whole period, so the delta is comparable to the total.
    elapsed = baseline["days_elapsed"]
    remaining = baseline["days_remaining"]
    projected_model = model_cost / elapsed * (elapsed + remaining)
    delta = projected_model * (change_pct / 100.0)
    return {
        **baseline,
        "model": model,
        "change_pct": change_pct,
        "model_cost_usd": round_usd(model_cost),
        "projected_model_usd": round_usd(projected_model),
        "delta_usd": round_usd(delta),
        "adjusted_total_usd": round_usd(baseline["projected_total_usd"] + delta),
        "note": "",
    }


def plan_summary(records: Iterable[UsageRecord], catalog: PricingCatalog,
                 budgets: List[Budget], period: str = "monthly",
                 now: Optional[datetime] = None) -> Dict:
    """One-shot planning view: projection plus a verdict per applicable budget.

    Budgets scoped to a project are evaluated against that project's records
    only, so a per-project limit is not judged against global spend.
    """
    records = list(records)
    overall = project_period_end(records, catalog, period=period, now=now)
    outcomes = []
    for budget in budgets:
        scoped = [r for r in records
                  if budget.project is None or r.project == budget.project]
        # A budget's own period wins; the caller's period is the fallback.
        effective = budget.period if budget.period in PLANNABLE_PERIODS else period
        outcome = budget_outcome(scoped, catalog, budget.limit_usd,
                                 period=effective, now=now)
        outcome["label"] = budget.label or (budget.project or "all")
        outcome["project"] = budget.project
        outcomes.append(outcome)
    return {"overall": overall, "budgets": outcomes}


def trend_note(projection: Dict, outcome: Optional[Dict] = None) -> str:
    """A one-line human summary of a projection, for report headers."""
    base = (f"spent ${projection['spent_usd']:.4f} in {projection['days_elapsed']} day(s), "
            f"projected ${projection['projected_total_usd']:.4f} "
            f"by period end ({projection['days_remaining']} day(s) left)")
    if not outcome or outcome.get("verdict") in (None, "no_budget"):
        return base
    verdict = outcome["verdict"]
    if verdict == "exceeded":
        return base + f" -- limit of ${outcome['limit_usd']:.2f} already exceeded"
    if verdict == "at_risk":
        return (base + f" -- projected ${outcome['overage_usd']:.4f} over the "
                f"${outcome['limit_usd']:.2f} limit; cut "
                f"{outcome['required_reduction_pct']:.1f}% or spend at most "
                f"${outcome['daily_allowance_usd']:.4f}/day")
    return (base + f" -- on track, ${outcome['daily_allowance_usd']:.4f}/day "
            f"available for the rest of the period")
