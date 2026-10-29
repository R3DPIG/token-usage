"""Alert rules: declarative conditions over usage data.

``budget`` answers one question with hard-coded logic ("did I overspend?").
This module generalises that into *rules*: a rule names a metric, a comparison,
a threshold and a scope, and the engine evaluates every rule against the stored
records. That makes conditions like "cache hit rate below 20% this week" or
"spend up more than 50% versus last month" expressible without new code.

Design notes
------------
* Rules are data, not code -- they round-trip through JSON so they can live in
  ``<home>/alerts.json`` and be edited by hand.
* Evaluation is pure: records + rules in, results out. No I/O, no clock reads
  beyond an injectable ``now``.
* A rule that cannot be evaluated (e.g. no data in scope) reports ``unknown``
  rather than ``ok``, so a silent data gap never looks like a passing check.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from token_usage import analytics, insights
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import round_usd

# Comparison operators a rule may use.
OPERATORS = ("gt", "gte", "lt", "lte")
# Metrics a rule may watch. Each maps to a function below.
METRICS = ("cost", "tokens", "records", "cache_ratio", "output_ratio",
           "cost_per_1k", "spend_change_pct")
# Dimensions a rule may be scoped to.
DIMENSIONS = ("total", "model", "project", "tag", "day")
# Severity levels, ordered from least to most urgent.
SEVERITIES = ("info", "warning", "critical")

_OPERATOR_FUNCS = {
    "gt": lambda value, threshold: value > threshold,
    "gte": lambda value, threshold: value >= threshold,
    "lt": lambda value, threshold: value < threshold,
    "lte": lambda value, threshold: value <= threshold,
}


class AlertError(ValueError):
    """Raised when a rule definition is malformed."""


@dataclass
class Rule:
    """A single alert condition.

    ``scope`` narrows which records the metric is computed over; ``dimension``
    decides whether the metric is computed once (``total``) or per bucket
    (``model``/``project``/``tag``/``day``), in which case each bucket is checked
    independently and reported separately.
    """

    name: str
    metric: str
    operator: str
    threshold: float
    dimension: str = "total"
    scope: Dict[str, str] = field(default_factory=dict)
    period: str = "total"
    severity: str = "warning"
    enabled: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise AlertError("rule name must not be empty")
        if self.metric not in METRICS:
            raise AlertError(f"unknown metric {self.metric!r}; expected one of {list(METRICS)}")
        if self.operator not in OPERATORS:
            raise AlertError(f"unknown operator {self.operator!r}; expected one of {list(OPERATORS)}")
        if self.dimension not in DIMENSIONS:
            raise AlertError(f"unknown dimension {self.dimension!r}; expected one of {list(DIMENSIONS)}")
        if self.severity not in SEVERITIES:
            raise AlertError(f"unknown severity {self.severity!r}; expected one of {list(SEVERITIES)}")
        if self.period not in ("daily", "weekly", "monthly", "total"):
            raise AlertError(f"unknown period {self.period!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        return cls(
            name=data["name"], metric=data["metric"], operator=data["operator"],
            threshold=float(data["threshold"]), dimension=data.get("dimension", "total"),
            scope=dict(data.get("scope", {})), period=data.get("period", "total"),
            severity=data.get("severity", "warning"),
            enabled=bool(data.get("enabled", True)),
            description=data.get("description", ""))

    def render(self) -> str:
        """Human-readable one-liner, e.g. ``cost gt 10.0 (model)``."""
        scope = " ".join(f"{k}={v}" for k, v in sorted(self.scope.items()))
        suffix = f" [{scope}]" if scope else ""
        return (f"{self.metric} {self.operator} {self.threshold:g} "
                f"({self.dimension}, {self.period}){suffix}")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _metric_value(metric: str, records: List[UsageRecord],
                  catalog: PricingCatalog, now: Optional[datetime]) -> Optional[float]:
    """Compute one metric over a record set; ``None`` when undefined.

    Returning ``None`` (rather than 0) for undefined metrics is what lets the
    engine distinguish "no data" from "a real zero".
    """
    if not records:
        return None
    totals = analytics.totals(records, catalog)
    if metric == "cost":
        return totals["cost_usd"]
    if metric == "tokens":
        return float(totals["total_tokens"])
    if metric == "records":
        return float(totals["records"])
    if metric == "cache_ratio":
        if not totals["input_tokens"]:
            return None
        return totals["cached_input_tokens"] / totals["input_tokens"]
    if metric == "output_ratio":
        if not totals["total_tokens"]:
            return None
        return totals["output_tokens"] / totals["total_tokens"]
    if metric == "cost_per_1k":
        if not totals["total_tokens"]:
            return None
        return totals["cost_usd"] / (totals["total_tokens"] / 1000.0)
    if metric == "spend_change_pct":
        comparison = insights.compare_periods(records, catalog, period="monthly", now=now)
        return comparison["delta_pct"]
    return None


def _scoped_records(records: Iterable[UsageRecord], rule: Rule,
                    now: Optional[datetime]) -> List[UsageRecord]:
    """Apply the rule's scope and period filters."""
    scoped = analytics.filter_records(
        records, project=rule.scope.get("project"), model=rule.scope.get("model"),
        tag=rule.scope.get("tag"))
    if rule.period != "total":
        scoped = analytics.filter_by_period(scoped, rule.period, now=now)
    return scoped


def _buckets(records: List[UsageRecord], rule: Rule,
             catalog: PricingCatalog) -> List[tuple]:
    """Split records into ``(label, records)`` pairs per the rule's dimension."""
    if rule.dimension == "total":
        return [("total", records)]
    grouped = analytics.group(records, rule.dimension, catalog)
    buckets = []
    for label in grouped:
        subset = [r for r in records if _bucket_key(r, rule.dimension) == label]
        buckets.append((label, subset))
    return buckets


def _bucket_key(record: UsageRecord, dimension: str) -> str:
    from token_usage.utils import day_key, parse_iso

    if dimension == "model":
        return record.model
    if dimension == "project":
        return record.project
    if dimension == "tag":
        return record.tags[0] if record.tags else "(untagged)"
    if dimension == "day":
        return day_key(parse_iso(record.timestamp))
    return "total"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rule(rule: Rule, records: Iterable[UsageRecord],
                  catalog: PricingCatalog,
                  now: Optional[datetime] = None) -> List[Dict]:
    """Evaluate one rule, returning a result per bucket.

    Status is ``firing`` when the condition holds, ``ok`` when it does not, and
    ``unknown`` when the metric could not be computed (no data in scope).
    """
    if not rule.enabled:
        return [{"rule": rule.name, "bucket": "-", "status": "disabled",
                 "metric": rule.metric, "value": None, "threshold": rule.threshold,
                 "severity": rule.severity, "detail": "rule is disabled"}]

    scoped = _scoped_records(records, rule, now)
    results = []
    for label, subset in _buckets(scoped, rule, catalog):
        value = _metric_value(rule.metric, subset, catalog, now)
        if value is None:
            results.append({"rule": rule.name, "bucket": label, "status": "unknown",
                            "metric": rule.metric, "value": None,
                            "threshold": rule.threshold, "severity": rule.severity,
                            "detail": "no data in scope"})
            continue
        firing = _OPERATOR_FUNCS[rule.operator](value, rule.threshold)
        results.append({
            "rule": rule.name, "bucket": label,
            "status": "firing" if firing else "ok",
            "metric": rule.metric, "value": round(value, 6),
            "threshold": rule.threshold, "severity": rule.severity,
            "detail": f"{rule.metric}={value:.4f} {rule.operator} {rule.threshold:g}",
        })
    return results


def evaluate_all(rules: Iterable[Rule], records: Iterable[UsageRecord],
                 catalog: PricingCatalog,
                 now: Optional[datetime] = None) -> List[Dict]:
    """Evaluate every rule; results are ordered by severity then rule name."""
    records = list(records)
    results = []
    for rule in rules:
        results.extend(evaluate_rule(rule, records, catalog, now=now))
    rank = {severity: index for index, severity in enumerate(reversed(SEVERITIES))}
    return sorted(results, key=lambda r: (rank.get(r["severity"], 0), r["rule"], r["bucket"]))


def summarize(results: Iterable[Dict]) -> Dict[str, int]:
    """Count results by status."""
    counts: Dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return counts


def worst_severity(results: Iterable[Dict]) -> Optional[str]:
    """Highest severity among firing results, or ``None`` when nothing fires."""
    firing = [r["severity"] for r in results if r["status"] == "firing"]
    if not firing:
        return None
    return max(firing, key=lambda s: SEVERITIES.index(s))


def exit_code_for(results: Iterable[Dict], fail_on: str = "critical") -> int:
    """Map results to a process exit code for CI use.

    ``fail_on`` is the lowest severity that should fail the run; ``never``
    always returns 0.
    """
    if fail_on == "never":
        return 0
    if fail_on not in SEVERITIES:
        raise AlertError(f"fail_on must be one of {list(SEVERITIES)} or 'never'")
    worst = worst_severity(results)
    if worst is None:
        return 0
    return 1 if SEVERITIES.index(worst) >= SEVERITIES.index(fail_on) else 0


# ---------------------------------------------------------------------------
# Rule set helpers
# ---------------------------------------------------------------------------

def parse_rule_args(name: str, metric: str, operator: str, threshold: float,
                    **kwargs) -> Rule:
    """Build a rule from CLI-style arguments, raising :class:`AlertError`."""
    return Rule(name=name, metric=metric, operator=operator, threshold=threshold,
                **kwargs)


def load_rules(data: Any) -> List[Rule]:
    """Parse a list of rule dicts, skipping malformed entries.

    A hand-edited alerts file with one bad rule should not disable the rest.
    """
    if not isinstance(data, list):
        return []
    rules = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            rules.append(Rule.from_dict(item))
        except (AlertError, KeyError, TypeError, ValueError):
            continue
    return rules


def dump_rules(rules: Iterable[Rule]) -> List[Dict]:
    return [rule.to_dict() for rule in rules]


def find_rule(rules: Iterable[Rule], name: str) -> Optional[Rule]:
    for rule in rules:
        if rule.name == name:
            return rule
    return None


def upsert_rule(rules: List[Rule], rule: Rule) -> List[Rule]:
    """Replace a rule with the same name, or append it."""
    remaining = [r for r in rules if r.name != rule.name]
    remaining.append(rule)
    return sorted(remaining, key=lambda r: r.name)


def remove_rule(rules: List[Rule], name: str) -> tuple:
    """Return ``(remaining, removed)``; ``removed`` is ``None`` when absent."""
    removed = find_rule(rules, name)
    if removed is None:
        return list(rules), None
    return [r for r in rules if r.name != name], removed


def default_rules() -> List[Rule]:
    """A small starter set, useful as ``alert --action init``."""
    return [
        Rule(name="monthly-spend", metric="cost", operator="gt", threshold=100.0,
             dimension="total", period="monthly", severity="critical",
             description="monthly spend above $100"),
        Rule(name="low-cache-hit", metric="cache_ratio", operator="lt", threshold=0.2,
             dimension="total", period="weekly", severity="info",
             description="cache hit rate below 20% this week"),
        Rule(name="spend-spike", metric="spend_change_pct", operator="gt", threshold=50.0,
             dimension="total", period="total", severity="warning",
             description="spend up more than 50% versus last month"),
    ]
