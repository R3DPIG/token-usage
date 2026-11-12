"""Envelope tracking: did the money we allocated actually get spent as planned?

:mod:`token_usage.planning` allocates a budget (``allocate_quota``) and
:mod:`token_usage.alerts` fires when a threshold is crossed. Neither answers the
review question that comes after the fact: *we handed out $100 across four
projects -- did each project stay inside its share, and was the share even the
right size?*

An **envelope** is a named spending allocation for one scope over one period::

    {"name": "api", "limit_usd": 40, "period": "monthly",
     "scope_type": "project", "scope_value": "api"}

Reviewing an envelope compares three numbers that each answer a different
question:

``consumed``
    How much of the limit is gone *now*.
``pace``
    How much *should* be gone now, if spend were even across the period.
    Consumption below pace is not automatically good news.
``projection``
    Where the envelope lands if the current rate holds.

Four honesty rules, each of which changes a verdict:

* **A ``total`` envelope has no pace.** It has no end date, so dividing elapsed
  time into it would invent a schedule nobody agreed to. ``pace`` and
  ``projection`` come back ``None`` and the status is judged on consumption
  alone -- never on a fabricated rhythm.
* **Zero activity is not zero risk.** An envelope whose scope matches no records
  reports ``no_activity``, not "0% consumed". A typo in ``scope_value`` looks
  exactly like perfect restraint, and congratulating it would hide the typo.
* **``over`` and ``heading_over`` are different facts.** One has already
  breached the limit; the other merely projects to. Reporting both as "over"
  teaches people to ignore the word.
* **Coverage is part of the verdict.** Envelopes that govern a minority of total
  spend cannot produce a clean bill of health for the workspace, so the review
  reports what fraction of spend it actually looked at.

Pure functions only: envelopes + records + catalog + clock in, plain dicts out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from token_usage import analytics, planning
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import round_usd

SCOPE_TYPES = ("all", "project", "model", "tag")
# Periods an envelope can be defined over. ``total`` means "no reset".
PERIODS = ("daily", "weekly", "monthly", "total")

STATUS_OK = "ok"
STATUS_WATCH = "watch"
STATUS_OVER = "over"
STATUS_HEADING_OVER = "heading_over"
STATUS_NO_ACTIVITY = "no_activity"

# Consumption at or above this share counts as "watch" before the period ends.
DEFAULT_WARN_AT = 0.8


@dataclass
class Envelope:
    """One allocation of money to one scope over one period."""

    name: str
    limit_usd: float
    period: str = "monthly"
    scope_type: str = "all"
    scope_value: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Envelope.name must be a non-empty string")
        if self.limit_usd <= 0:
            raise ValueError("Envelope.limit_usd must be > 0")
        if self.period not in PERIODS:
            raise ValueError(f"Envelope.period must be one of {PERIODS}, got {self.period!r}")
        if self.scope_type not in SCOPE_TYPES:
            raise ValueError(
                f"Envelope.scope_type must be one of {SCOPE_TYPES}, got {self.scope_type!r}")
        if self.scope_type != "all" and not self.scope_value:
            raise ValueError(
                f"Envelope.scope_value is required when scope_type is {self.scope_type!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "limit_usd": self.limit_usd,
                "period": self.period, "scope_type": self.scope_type,
                "scope_value": self.scope_value, "note": self.note}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Envelope":
        return cls(name=str(data["name"]), limit_usd=float(data["limit_usd"]),
                   period=data.get("period", "monthly"),
                   scope_type=data.get("scope_type", "all"),
                   scope_value=str(data.get("scope_value") or ""),
                   note=str(data.get("note") or ""))

    def scope_label(self) -> str:
        """Human-readable scope, e.g. ``project=api`` or ``all``."""
        if self.scope_type == "all":
            return "all"
        return f"{self.scope_type}={self.scope_value}"


def load_envelopes(payload: Any) -> List[Envelope]:
    """Parse stored envelope JSON, skipping malformed entries.

    One bad envelope must not prevent the others from being reviewed, matching
    how :mod:`token_usage.storage` treats malformed rules elsewhere.
    """
    rows = payload.get("envelopes", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    result: List[Envelope] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            result.append(Envelope.from_dict(row))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def dump_envelopes(envelopes: Iterable[Envelope]) -> Dict[str, Any]:
    """Serialise envelopes for :mod:`token_usage.storage`."""
    return {"envelopes": [envelope.to_dict() for envelope in envelopes]}


def matches(envelope: Envelope, record: UsageRecord) -> bool:
    """Whether one record falls inside this envelope's scope."""
    if envelope.scope_type == "all":
        return True
    if envelope.scope_type == "project":
        return record.project == envelope.scope_value
    if envelope.scope_type == "model":
        return record.model == envelope.scope_value
    return envelope.scope_value in record.tags


def in_scope(envelope: Envelope, records: Iterable[UsageRecord]) -> List[UsageRecord]:
    """Records this envelope governs; ``all`` governs everything."""
    return [record for record in records if matches(envelope, record)]


def _pace_figures(period: str, now: Optional[datetime]) -> Optional[Dict]:
    """Elapsed/remaining days for a resettable period; ``None`` for ``total``.

    ``total`` envelopes deliberately get no schedule. Returning zeros instead
    would let callers divide by them and manufacture a pace out of nothing.
    """
    if period == "total":
        return None
    elapsed = planning.days_elapsed(period, now=now)
    remaining = planning.days_remaining(period, now=now)
    # ``days_elapsed`` counts today and ``days_remaining`` excludes it, so the
    # two already sum to the period length; subtracting again would shorten it.
    return {"days_elapsed": elapsed, "days_remaining": remaining,
            "days_total": max(1, elapsed + remaining)}


def in_window(period: str, record: UsageRecord,
              now: Optional[datetime] = None) -> bool:
    """Whether one record falls inside the envelope's current period window."""
    if period == "total":
        return True
    return analytics.filter_by_period([record], period, now=now) != []


def review_envelope(envelope: Envelope, records: Iterable[UsageRecord],
                    catalog: PricingCatalog, now: Optional[datetime] = None,
                    warn_at: float = DEFAULT_WARN_AT) -> Dict:
    """Review one envelope: consumption, pace, projection and a status.

    ``pace_usd`` is what an even spender would have consumed by now, and
    ``pace_delta_usd`` is consumption minus pace. A negative delta is *not*
    reported as success: under-spending can mean unused capacity as easily as it
    can mean thrift, and this function does not pretend to know which.
    """
    now = now or datetime.now(timezone.utc)
    scoped = [record for record in in_scope(envelope, records)
              if in_window(envelope.period, record, now)]
    spent = analytics.totals(scoped, catalog)["cost_usd"]
    consumed = round(spent / envelope.limit_usd, 4)

    if not scoped:
        return {"envelope": envelope.name, "scope": envelope.scope_label(),
                "period": envelope.period, "limit_usd": envelope.limit_usd,
                "spent_usd": 0.0, "consumed": 0.0, "remaining_usd": envelope.limit_usd,
                "records": 0, "status": STATUS_NO_ACTIVITY,
                "pace_usd": None, "pace_delta_usd": None,
                "projection_usd": None, "days_remaining": None,
                "note": (f"no records match {envelope.scope_label()} in this "
                         f"{envelope.period} period; an empty scope is not the "
                         "same as an unused budget")}

    remaining = round_usd(max(0.0, envelope.limit_usd - spent))
    pace = _pace_figures(envelope.period, now)
    if pace is None:
        pace_usd = pace_delta = projection = None
        days_remaining = None
    else:
        pace_usd = round_usd(envelope.limit_usd * pace["days_elapsed"] / pace["days_total"])
        pace_delta = round_usd(spent - pace_usd)
        daily = spent / pace["days_elapsed"]
        projection = round_usd(daily * pace["days_total"])
        days_remaining = pace["days_remaining"]

    if spent > envelope.limit_usd:
        status = STATUS_OVER
    elif projection is not None and projection > envelope.limit_usd:
        status = STATUS_HEADING_OVER
    elif consumed >= warn_at:
        status = STATUS_WATCH
    else:
        status = STATUS_OK

    return {"envelope": envelope.name, "scope": envelope.scope_label(),
            "period": envelope.period, "limit_usd": envelope.limit_usd,
            "spent_usd": spent, "consumed": consumed,
            "remaining_usd": remaining, "records": len(scoped),
            "status": status,
            "pace_usd": pace_usd, "pace_delta_usd": pace_delta,
            "projection_usd": projection, "days_remaining": days_remaining,
            "note": ""}


def overlaps(envelopes: Iterable[Envelope]) -> List[Dict]:
    """Pairs of envelopes whose scopes can both claim the same record.

    Overlapping envelopes are legal -- a project envelope inside a workspace-wide
    one is normal -- but their limits are *not* additive, and summing them would
    overstate how much money is actually governed. Reporting the pairs keeps the
    totals honest.
    """
    items = list(envelopes)
    pairs: List[Dict] = []

    def may_share(a: Envelope, b: Envelope) -> bool:
        if a.scope_type == "all" or b.scope_type == "all":
            return True
        return a.scope_type == b.scope_type and a.scope_value == b.scope_value

    for index, first in enumerate(items):
        for second in items[index + 1:]:
            if may_share(first, second):
                pairs.append({"a": first.name, "b": second.name,
                              "why": ("both include every record"
                                      if "all" in (first.scope_type, second.scope_type)
                                      else "same scope, different limits")})
    return pairs


def review(envelopes: Iterable[Envelope], records: Iterable[UsageRecord],
           catalog: PricingCatalog, now: Optional[datetime] = None,
           warn_at: float = DEFAULT_WARN_AT) -> Dict:
    """Review every envelope and summarise the whole allocation.

    ``governed_usd`` is the spend covered by at least one envelope, computed by
    union rather than by summing envelopes, so nested scopes are not counted
    twice. ``coverage`` is that share of total spend: it says how much of the
    workspace this review actually had an opinion about.
    """
    envelopes = list(envelopes)
    records = list(records)
    now = now or datetime.now(timezone.utc)
    rows = [review_envelope(envelope, records, catalog, now=now, warn_at=warn_at)
            for envelope in envelopes]
    rows.sort(key=lambda row: (row["limit_usd"], row["envelope"]), reverse=True)

    # Coverage is measured over the union of every envelope's own period window,
    # so a monthly envelope is not credited with governing last quarter's spend.
    # The denominator uses the same window: comparing governed monthly spend
    # against workspace lifetime spend would understate coverage.
    in_window_idx = [index for index, record in enumerate(records)
                     if any(in_window(item.period, record, now) for item in envelopes)]
    governed_usd = round_usd(sum(
        catalog.cost_for_record(records[index]) for index in in_window_idx
        if any(matches(item, records[index]) for item in envelopes)))
    total_spend = round_usd(sum(catalog.cost_for_record(records[index])
                                for index in in_window_idx))

    counts = {name: sum(1 for row in rows if row["status"] == name)
              for name in (STATUS_OK, STATUS_WATCH, STATUS_OVER,
                           STATUS_HEADING_OVER, STATUS_NO_ACTIVITY)}
    total_limits = round_usd(sum(row["limit_usd"] for row in rows))
    total_spent = round_usd(sum(row["spent_usd"] for row in rows))
    coverage = round(governed_usd / total_spend, 4) if total_spend > 0 else 0.0

    warnings: List[str] = []
    if not envelopes:
        warnings.append("no envelopes defined; nothing is being governed")
    for row in rows:
        if row["status"] == STATUS_NO_ACTIVITY:
            warnings.append(
                f"envelope {row['envelope']!r} matched no records under "
                f"{row['scope']}; check the scope before reading 0% as restraint")
    if envelopes and coverage < 0.5 and total_spend > 0:
        warnings.append(
            f"envelopes govern only {coverage * 100:.1f}% of spend; "
            "the rest has no allocation to be reviewed against")
    if overlaps(envelopes):
        warnings.append(
            "envelopes overlap, so their limits do not add up; "
            "totals below are per-envelope, not a workspace-wide cap")

    return {"rows": rows, "counts": counts, "coverage": coverage,
            "total_limits_usd": total_limits, "total_spent_usd": total_spent,
            "governed_usd": governed_usd, "workspace_spend_usd": total_spend,
            "overlaps": overlaps(envelopes), "warnings": warnings,
            "status": ("clean" if counts[STATUS_OVER] == 0
                       and counts[STATUS_HEADING_OVER] == 0
                       and counts[STATUS_NO_ACTIVITY] == 0 else "attention")}


def next_steps(result: Dict) -> List[Dict]:
    """Ordered actions for the rows that need one, worst first.

    Only rows with something to act on appear: an ``ok`` envelope produces no
    advice, because advice attached to a healthy row is noise that trains people
    to stop reading the list.
    """
    actions = {STATUS_OVER: "reduce spend or raise the limit; the line is crossed",
               STATUS_HEADING_OVER: "the current rate ends above the limit; "
                                    "slow down or re-plan the allocation",
               STATUS_WATCH: "most of the limit is gone while the period is not",
               STATUS_NO_ACTIVITY: "no records match this scope; "
                                   "the allocation is not tracking anything"}
    rows = [row for row in result["rows"] if row["status"] in actions]
    rows.sort(key=lambda row: row["spent_usd"], reverse=True)
    return [{"envelope": row["envelope"], "scope": row["scope"],
             "status": row["status"], "spent_usd": row["spent_usd"],
             "limit_usd": row["limit_usd"],
             "over_usd": round_usd(max(0.0, row["spent_usd"] - row["limit_usd"])),
             "action": actions[row["status"]]} for row in rows]
