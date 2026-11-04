"""Cost attribution: decide who pays for each token.

:mod:`token_usage.analytics` answers "how much did we spend". This module answers
the accounting question that follows: *which team, cost centre or customer does
that spend belong to, and can we defend the number?*

The model is deliberately simple and auditable:

* An :class:`OwnerRule` maps records to an owner using **first-match-wins**
  predicates. Order matters, so rules are stored as an ordered list and the
  first rule whose predicates all match claims the record.
* A record that matches no rule is **unattributed**, not silently dropped. The
  unattributed share is reported as a first-class number, because a chargeback
  report that quietly hides 30% of spend is worse than no report.
* Splitting is explicit: a rule may declare ``split`` weights so a shared
  resource (a gateway, a shared key) is divided across owners by a fixed ratio
  rather than an arbitrary guess.

Everything here is pure: records + rules + catalog in, plain dicts out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from token_usage import analytics
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import round_usd

# Predicate fields a rule may constrain. Kept as a whitelist so a typo in a
# hand-edited owners.json is rejected loudly instead of silently matching
# nothing (which would look like "no spend" rather than "bad config").
MATCH_FIELDS = ("project", "model", "tag", "note_contains")


@dataclass
class OwnerRule:
    """One attribution rule: match predicates plus the owner they imply.

    ``split`` maps owner -> weight. When present, the matched cost is divided
    across those owners proportionally instead of going wholly to ``owner``.
    Weights need not sum to 1; they are normalised, so ``{"a": 1, "b": 3}``
    means 25%/75% and is easier to read than ``{"a": 0.25, "b": 0.75}``.
    """

    owner: str
    project: Optional[str] = None
    model: Optional[str] = None
    tag: Optional[str] = None
    note_contains: Optional[str] = None
    split: Dict[str, float] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        if not self.owner and not self.split:
            raise ValueError("OwnerRule needs an owner or a split")
        for name in MATCH_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"OwnerRule.{name} must be a string or None")
        for name, weight in self.split.items():
            if not name:
                raise ValueError("split owner names must be non-empty")
            if weight <= 0:
                raise ValueError(f"split weight for {name!r} must be > 0")

    def matches(self, record: UsageRecord) -> bool:
        """True when every supplied predicate holds (unset predicates are free)."""
        if self.project is not None and record.project != self.project:
            return False
        if self.model is not None and record.model != self.model:
            return False
        if self.tag is not None and self.tag not in record.tags:
            return False
        if self.note_contains is not None and \
                self.note_contains.lower() not in (record.note or "").lower():
            return False
        return True

    def shares(self) -> List[Tuple[str, float]]:
        """Normalised ``(owner, weight)`` pairs; a plain rule is a single share."""
        if not self.split:
            return [(self.owner, 1.0)]
        total = sum(self.split.values())
        return [(name, weight / total) for name, weight in self.split.items()]

    def to_dict(self) -> Dict:
        data: Dict = {"owner": self.owner}
        for name in MATCH_FIELDS:
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.split:
            data["split"] = dict(self.split)
        if self.label:
            data["label"] = self.label
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "OwnerRule":
        unknown = set(data) - set(MATCH_FIELDS) - {"owner", "split", "label"}
        if unknown:
            raise ValueError(f"unknown rule field(s): {sorted(unknown)}")
        return cls(owner=data.get("owner", ""),
                   project=data.get("project"), model=data.get("model"),
                   tag=data.get("tag"), note_contains=data.get("note_contains"),
                   split={str(k): float(v)
                          for k, v in (data.get("split") or {}).items()},
                   label=data.get("label", ""))


def load_rules(items: Iterable[Dict]) -> List[OwnerRule]:
    """Parse rules, skipping malformed entries rather than failing the whole file.

    A single bad rule must not make the report unusable. The skip is not silent
    either: :func:`attribute` reports coverage, so a rule that failed to parse
    shows up as unattributed spend rather than as a wrong owner.
    """
    rules = []
    for item in items or []:
        try:
            rules.append(OwnerRule.from_dict(item))
        except (ValueError, TypeError, KeyError):
            continue
    return rules


def dump_rules(rules: Iterable[OwnerRule]) -> List[Dict]:
    return [rule.to_dict() for rule in rules]


def match_rule(record: UsageRecord,
               rules: List[OwnerRule]) -> Optional[OwnerRule]:
    """First rule that matches, or ``None``. Order is the whole contract."""
    for rule in rules:
        if rule.matches(record):
            return rule
    return None


def _orphans(records: Iterable[UsageRecord], rules: List[OwnerRule],
             catalog: PricingCatalog) -> List[Tuple[UsageRecord, float]]:
    """Records no rule claimed, paired with their cost."""
    return [(record, catalog.cost_for_record(record)) for record in records
            if match_rule(record, rules) is None]


def attribute(records: Iterable[UsageRecord], rules: List[OwnerRule],
              catalog: PricingCatalog) -> Dict:
    """Assign every record's cost to one or more owners.

    Returns per-owner totals plus the unattributed remainder. ``coverage`` is
    the share of spend that found an owner -- the single number to watch, since
    it tells you whether the report is trustworthy before you read any of it.
    """
    per_owner: Dict[str, Dict] = {}
    unattributed_cost = 0.0
    unattributed_count = 0
    total_cost = 0.0

    for record in records:
        cost = catalog.cost_for_record(record)
        total_cost += cost
        rule = match_rule(record, rules)
        if rule is None:
            unattributed_cost += cost
            unattributed_count += 1
            continue
        for owner, weight in rule.shares():
            bucket = per_owner.setdefault(owner, {"cost_usd": 0.0, "records": 0,
                                                  "tokens": 0, "rules": set()})
            bucket["cost_usd"] += cost * weight
            bucket["records"] += 1
            bucket["tokens"] += int(record.total_tokens * weight)
            bucket["rules"].add(rule.label or rule.owner)

    rows = [{"owner": owner, "cost_usd": round_usd(bucket["cost_usd"]),
             "records": bucket["records"], "tokens": bucket["tokens"],
             "share": (round(bucket["cost_usd"] / total_cost, 4)
                       if total_cost else 0.0),
             "rules": sorted(bucket["rules"])}
            for owner, bucket in per_owner.items()]
    rows.sort(key=lambda row: row["cost_usd"], reverse=True)

    return {
        "owners": rows,
        "total_usd": round_usd(total_cost),
        "attributed_usd": round_usd(total_cost - unattributed_cost),
        "unattributed_usd": round_usd(unattributed_cost),
        "unattributed_records": unattributed_count,
        "coverage": (round((total_cost - unattributed_cost) / total_cost, 4)
                     if total_cost else 0.0),
    }


def unattributed_records(records: Iterable[UsageRecord], rules: List[OwnerRule],
                         catalog: PricingCatalog, limit: int = 10) -> Dict:
    """The records no rule claimed, most expensive first.

    This is the actionable half of the coverage number: it shows exactly which
    project/model/tag combination needs a new rule.
    """
    orphans = [{"timestamp": record.timestamp, "project": record.project,
                "model": record.model, "tags": list(record.tags),
                "note": record.note, "cost_usd": round_usd(cost)}
               for record, cost in _orphans(records, rules, catalog)]
    orphans.sort(key=lambda item: item["cost_usd"], reverse=True)
    return {"count": len(orphans),
            "cost_usd": round_usd(sum(item["cost_usd"] for item in orphans)),
            "records": orphans[:limit]}


def suggest_rules(records: Iterable[UsageRecord], rules: List[OwnerRule],
                  catalog: PricingCatalog, min_cost_usd: float = 0.0) -> List[Dict]:
    """Propose rules for unattributed spend, grouped by project.

    Grouping by project is the useful default: projects are the unit teams
    usually own, and a single project rule typically closes most of the gap.
    Suggestions are proposals only -- nothing is written.
    """
    grouped: Dict[str, Dict] = {}
    for record, cost in _orphans(records, rules, catalog):
        bucket = grouped.setdefault(record.project, {"cost_usd": 0.0, "records": 0,
                                                     "models": set()})
        bucket["cost_usd"] += cost
        bucket["records"] += 1
        bucket["models"].add(record.model)

    suggestions = [{"project": project, "cost_usd": round_usd(bucket["cost_usd"]),
                    "records": bucket["records"],
                    "models": sorted(bucket["models"]),
                    "proposed_rule": {"owner": f"team-{project}",
                                      "project": project}}
                   for project, bucket in grouped.items()
                   if bucket["cost_usd"] >= min_cost_usd]
    return sorted(suggestions, key=lambda item: item["cost_usd"], reverse=True)


def invoice(records: Iterable[UsageRecord], rules: List[OwnerRule],
            catalog: PricingCatalog, owner: str) -> Dict:
    """A single owner's bill, broken down by project and model.

    The breakdown is what makes a chargeback defensible: an owner who sees
    "gpt-4o, 1.2M tokens, $3.10" can check it against their own logs.
    """
    scoped = []
    for record in records:
        rule = match_rule(record, rules)
        weight = dict(rule.shares()).get(owner) if rule else None
        if weight is not None:
            scoped.append((record, weight))

    if not scoped:
        return {"owner": owner, "found": False, "cost_usd": 0.0,
                "records": 0, "by_project": [], "by_model": []}

    by_project: Dict[str, Dict] = {}
    by_model: Dict[str, Dict] = {}
    total = 0.0
    for record, weight in scoped:
        cost = catalog.cost_for_record(record) * weight
        total += cost
        for target, key in ((by_project, record.project), (by_model, record.model)):
            bucket = target.setdefault(key, {"cost_usd": 0.0, "records": 0,
                                             "tokens": 0})
            bucket["cost_usd"] += cost
            bucket["records"] += 1
            bucket["tokens"] += int(record.total_tokens * weight)

    def _rows(source: Dict[str, Dict], key: str) -> List[Dict]:
        rows = [{key: name, "cost_usd": round_usd(b["cost_usd"]),
                 "records": b["records"], "tokens": b["tokens"],
                 "share": round(b["cost_usd"] / total, 4) if total else 0.0}
                for name, b in source.items()]
        return sorted(rows, key=lambda row: row["cost_usd"], reverse=True)

    return {"owner": owner, "found": True, "cost_usd": round_usd(total),
            "records": len(scoped), "by_project": _rows(by_project, "project"),
            "by_model": _rows(by_model, "model")}


def reconcile(records: Iterable[UsageRecord], rules: List[OwnerRule],
              catalog: PricingCatalog) -> Dict:
    """Check that attributed + unattributed equals the true total.

    A chargeback report is only usable if it balances. This recomputes the
    grand total independently of :func:`attribute` and reports any drift, so a
    rounding or split-weight bug surfaces as a number instead of a silent
    discrepancy someone finds at quarter close.
    """
    records = list(records)
    truth = analytics.totals(records, catalog)["cost_usd"]
    result = attribute(records, rules, catalog)
    accounted = round_usd(result["attributed_usd"] + result["unattributed_usd"])
    drift = round_usd(truth - accounted)
    return {
        "total_usd": truth,
        "attributed_usd": result["attributed_usd"],
        "unattributed_usd": result["unattributed_usd"],
        "accounted_usd": accounted,
        "drift_usd": drift,
        "balanced": abs(drift) < 1e-6,
        "coverage": result["coverage"],
    }


def summarize(records: Iterable[UsageRecord], rules: List[OwnerRule],
              catalog: PricingCatalog) -> Dict:
    """One-shot attribution summary used by the plugin's default view."""
    records = list(records)
    data = attribute(records, rules, catalog)
    data["reconcile"] = reconcile(records, rules, catalog)
    data["orphans"] = unattributed_records(records, rules, catalog, limit=5)
    data["suggestions"] = suggest_rules(records, rules, catalog)
    return data
