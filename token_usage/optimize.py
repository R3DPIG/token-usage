"""Cost optimisation: turn observations into quantified recommendations.

:mod:`token_usage.insights` describes the data ("waste is $3.20", "gpt-4o costs
$0.003/1K"). This module answers the next question: *what should I change, and
how much would it save?*

Every recommendation carries a real dollar figure computed by repricing the
observed tokens -- never a guess. Where a saving cannot be computed honestly
(batching small calls, for instance) the recommendation reports the opportunity
size and leaves ``savings_usd`` at ``None`` rather than inventing a number.

Confidence and effort are separate axes on purpose: a high-saving change that
requires rewriting prompts is not the same decision as a low-saving one-line
config change, and collapsing them into a single score would hide that.

Pure functions only: records + catalog in, plain dicts out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from token_usage import analytics, insights
from token_usage.models import ModelPricing, UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import round_usd

# Price bands, derived from the catalog rather than hard-coded model names, so a
# newly added model is classified automatically.
TIER_ORDER = ("premium", "mid", "economy")
PREMIUM_FLOOR = 5.0   # USD per 1M input tokens
MID_FLOOR = 1.0

CONFIDENCE_LEVELS = ("high", "medium", "low")
EFFORT_LEVELS = ("low", "medium", "high")


def price_tier(pricing: ModelPricing) -> str:
    """Classify a model by its input price band."""
    if pricing.input_per_1m >= PREMIUM_FLOOR:
        return "premium"
    if pricing.input_per_1m >= MID_FLOOR:
        return "mid"
    return "economy"


def tier_index(tier: str) -> int:
    """Position in :data:`TIER_ORDER`; higher means cheaper."""
    return TIER_ORDER.index(tier)


def generates_output(pricing: ModelPricing) -> bool:
    """Whether a model can produce output tokens at all.

    Embedding models bill nothing for output, so they can never stand in for a
    generation model no matter how much cheaper they are. Price alone cannot
    tell you that -- this is the capability check that stops the optimiser from
    recommending a swap that would simply break the caller.
    """
    return pricing.output_per_1m > 0


def _model_totals(records: Iterable[UsageRecord]) -> Dict[str, Dict[str, int]]:
    """Aggregate token counts per model (no pricing involved)."""
    totals: Dict[str, Dict[str, int]] = {}
    for record in records:
        bucket = totals.setdefault(record.model, {"input": 0, "output": 0,
                                                  "cached": 0, "records": 0})
        bucket["input"] += record.input_tokens
        bucket["output"] += record.output_tokens
        bucket["cached"] += record.cached_input_tokens
        bucket["records"] += 1
    return totals


def model_substitutions(records: Iterable[UsageRecord], catalog: PricingCatalog,
                        allow_tier_drop: int = 0,
                        min_savings_usd: float = 0.0) -> List[Dict]:
    """Find cheaper models that could serve each model's observed token profile.

    ``allow_tier_drop`` controls how far down the price bands a candidate may
    sit: ``0`` keeps substitutions within the same band (safest), ``1`` permits
    one band down. A larger drop is a bigger saving but a bigger behavioural
    risk, which is why it is opt-in rather than the default.

    Savings are computed by repricing the *same* token counts on the candidate,
    so the figure is what the bill would have been, not an extrapolation.
    """
    records = list(records)
    totals = _model_totals(records)
    if not totals:
        return []

    candidates = []
    for model in catalog.models():
        pricing = catalog.get(model)
        if pricing is not None:
            candidates.append((model, pricing, price_tier(pricing)))

    results = []
    for model, usage in totals.items():
        current_pricing = catalog.get(model)
        if current_pricing is None:
            continue
        current_tier = price_tier(current_pricing)
        current_cost = catalog.cost_for(model, usage["input"], usage["output"],
                                        usage["cached"])
        if current_cost <= 0:
            continue

        best = None
        for name, pricing, tier in candidates:
            if name == model:
                continue
            # A model that cannot generate output cannot replace one that does,
            # however cheap it is -- an embedding endpoint is not a drop-in.
            if generates_output(current_pricing) and not generates_output(pricing):
                continue
            drop = tier_index(tier) - tier_index(current_tier)
            if drop < 0 or drop > allow_tier_drop:
                continue
            alternative_cost = catalog.cost_for(name, usage["input"],
                                                usage["output"], usage["cached"])
            saving = current_cost - alternative_cost
            if saving <= 0:
                continue
            if best is None or saving > best["savings_usd"]:
                best = {"model": model, "alternative": name,
                        "current_cost_usd": round_usd(current_cost),
                        "alternative_cost_usd": round_usd(alternative_cost),
                        "savings_usd": round_usd(saving),
                        "tier_drop": drop, "records": usage["records"]}

        if best is not None and best["savings_usd"] >= min_savings_usd:
            results.append(best)
    return sorted(results, key=lambda item: item["savings_usd"], reverse=True)


def cache_opportunities(records: Iterable[UsageRecord], catalog: PricingCatalog,
                        min_tokens: int = 1000) -> List[Dict]:
    """Models whose input tokens were never served from cache.

    Only models where caching is actually cheaper are reported -- for a model
    with no cache discount the saving would be zero, and listing it would be
    noise. ``min_tokens`` filters out models with too little traffic for caching
    to be worth configuring.
    """
    results = []
    for model, usage in _model_totals(records).items():
        pricing = catalog.get(model)
        if pricing is None or usage["cached"] > 0:
            continue
        if usage["input"] < min_tokens:
            continue
        discount = pricing.input_per_1m - pricing.cached_input_per_1m
        if discount <= 0:
            continue                      # no cache discount available
        results.append({
            "model": model,
            "input_tokens": usage["input"],
            "records": usage["records"],
            "input_per_1m": pricing.input_per_1m,
            "cached_per_1m": pricing.cached_input_per_1m,
            "savings_usd": round_usd(usage["input"] / 1_000_000.0 * discount),
        })
    return sorted(results, key=lambda item: item["savings_usd"], reverse=True)


def repeated_shapes(records: Iterable[UsageRecord], catalog: PricingCatalog,
                    min_repeats: int = 3) -> List[Dict]:
    """Identical ``(model, input, output)`` shapes seen repeatedly.

    A repeated shape is strong evidence of the same prompt being sent again, so
    it is the clearest signal that caching would pay off. Savings assume every
    repeat after the first could have been a cache hit.
    """
    groups: Dict[tuple, List[UsageRecord]] = {}
    for record in records:
        key = (record.model, record.input_tokens, record.output_tokens)
        groups.setdefault(key, []).append(record)

    results = []
    for (model, input_tokens, output_tokens), items in groups.items():
        if len(items) < min_repeats:
            continue
        pricing = catalog.get(model)
        if pricing is None:
            continue
        discount = pricing.input_per_1m - pricing.cached_input_per_1m
        if discount <= 0:
            continue
        repeats = len(items) - 1
        results.append({"model": model, "input_tokens": input_tokens,
                        "output_tokens": output_tokens, "repeats": repeats,
                        "savings_usd": round_usd(
                            repeats * input_tokens / 1_000_000.0 * discount)})
    return sorted(results, key=lambda item: item["savings_usd"], reverse=True)


def small_calls(records: Iterable[UsageRecord],
                threshold_tokens: int = 500) -> Dict:
    """Calls too small to be efficient, as batching candidates.

    No dollar figure is attached: the saving from batching depends on how the
    caller restructures its requests, which this tool cannot know. Reporting a
    fabricated number here would be worse than reporting none.
    """
    records = list(records)
    small = [r for r in records if r.total_tokens < threshold_tokens]
    return {
        "threshold_tokens": threshold_tokens,
        "count": len(small),
        "share": round(len(small) / len(records), 4) if records else 0.0,
        "total_tokens": sum(r.total_tokens for r in small),
        "savings_usd": None,
    }


def output_waste(records: Iterable[UsageRecord], catalog: PricingCatalog,
                 min_output_ratio: float = 0.05) -> Dict:
    """Spend on calls that produced almost no output (thin wrapper on insights)."""
    return insights.waste_estimate(records, catalog,
                                   min_output_ratio=min_output_ratio)


@dataclass
class Recommendation:
    """One actionable change, with its quantified impact.

    ``model`` names the model the change applies to, when it is model-specific.
    It exists so overlapping recommendations can be de-duplicated: caching and
    substituting the *same* tokens are alternatives, not additive savings.
    """

    kind: str
    title: str
    detail: str
    savings_usd: Optional[float]
    confidence: str = "medium"
    effort: str = "medium"
    model: Optional[str] = None
    evidence: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"confidence must be one of {CONFIDENCE_LEVELS}")
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}")

    def to_dict(self) -> Dict:
        return {"kind": self.kind, "title": self.title, "detail": self.detail,
                "savings_usd": self.savings_usd, "confidence": self.confidence,
                "effort": self.effort, "model": self.model,
                "evidence": dict(self.evidence)}


def recommendations(records: Iterable[UsageRecord], catalog: PricingCatalog,
                    allow_tier_drop: int = 0,
                    min_savings_usd: float = 0.0) -> List[Recommendation]:
    """Build and rank every recommendation, largest saving first.

    Recommendations without a computable saving sort last: they are still worth
    reading, but they should not outrank a change with a known dollar value.
    """
    records = list(records)
    found: List[Recommendation] = []

    for item in model_substitutions(records, catalog,
                                    allow_tier_drop=allow_tier_drop,
                                    min_savings_usd=min_savings_usd):
        found.append(Recommendation(
            kind="substitute",
            title=f"Move {item['model']} traffic to {item['alternative']}",
            detail=(f"{item['records']} call(s) cost "
                    f"${item['current_cost_usd']:.4f}; the same tokens on "
                    f"{item['alternative']} would cost "
                    f"${item['alternative_cost_usd']:.4f}"),
            savings_usd=item["savings_usd"],
            confidence="medium" if item["tier_drop"] == 0 else "low",
            effort="high" if item["tier_drop"] else "medium",
            model=item["model"], evidence=item))

    for item in cache_opportunities(records, catalog):
        found.append(Recommendation(
            kind="cache",
            title=f"Enable prompt caching for {item['model']}",
            detail=(f"{item['input_tokens']:,} input tokens were billed at "
                    f"${item['input_per_1m']:g}/1M with no cache hits; cached "
                    f"input is ${item['cached_per_1m']:g}/1M"),
            savings_usd=item["savings_usd"], confidence="high", effort="low",
            model=item["model"], evidence=item))

    for item in repeated_shapes(records, catalog):
        found.append(Recommendation(
            kind="repeat",
            title=f"Cache a repeated prompt on {item['model']}",
            detail=(f"the same {item['input_tokens']:,}-token prompt was sent "
                    f"{item['repeats'] + 1} times; caching the repeats would "
                    "bill them at the cached rate"),
            savings_usd=item["savings_usd"], confidence="high", effort="low",
            model=item["model"], evidence=item))

    # Emit one waste recommendation per model so it can be de-duplicated
    # against that model's cache/substitution options -- they target the same
    # tokens, so adding them would double-count.
    by_model: Dict[str, Dict] = {}
    for offender in output_waste(records, catalog)["offenders"]:
        bucket = by_model.setdefault(offender["model"], {"cost": 0.0, "count": 0})
        bucket["cost"] += offender["cost_usd"]
        bucket["count"] += 1
    for model, bucket in sorted(by_model.items(),
                                key=lambda kv: kv[1]["cost"], reverse=True):
        found.append(Recommendation(
            kind="waste",
            title=f"Investigate low-output calls on {model}",
            detail=(f"{bucket['count']} call(s) returned under 5% output "
                    f"tokens, costing ${bucket['cost']:.4f} -- usually "
                    "truncated or failed responses"),
            savings_usd=round_usd(bucket["cost"]), confidence="medium",
            effort="medium", model=model, evidence={"count": bucket["count"]}))

    small = small_calls(records)
    if small["count"] > 0 and small["share"] >= 0.2:
        found.append(Recommendation(
            kind="batch",
            title="Batch small calls",
            detail=(f"{small['count']} call(s) ({small['share'] * 100:.0f}%) used "
                    f"under {small['threshold_tokens']} tokens; batching them "
                    "reduces per-call overhead"),
            savings_usd=None, confidence="low", effort="high", evidence=small))

    return sorted(found, key=lambda rec: (rec.savings_usd is None,
                                          -(rec.savings_usd or 0.0)))


def total_savings(recs: Iterable[Recommendation]) -> float:
    """Sum of savings, counting each model's best option only once.

    Recommendations overlap: caching a model's input and switching that same
    model to a cheaper one both reduce the *same* tokens, so adding them would
    double-count and could report savings above 100% of spend. For each model we
    therefore take only the largest single saving. Model-agnostic
    recommendations (waste, batch) are added on top, since they target different
    calls.
    """
    best_per_model: Dict[str, float] = {}
    unmodelled = 0.0
    for rec in recs:
        if rec.savings_usd is None:
            continue
        if rec.model is None:
            unmodelled += rec.savings_usd
        else:
            best_per_model[rec.model] = max(best_per_model.get(rec.model, 0.0),
                                            rec.savings_usd)
    return round_usd(sum(best_per_model.values()) + unmodelled)


def savings_ratio(recs: Iterable[Recommendation], records: Iterable[UsageRecord],
                  catalog: PricingCatalog) -> float:
    """Quantified savings as a fraction of current spend, for context."""
    spent = analytics.totals(records, catalog)["cost_usd"]
    if spent <= 0:
        return 0.0
    return round(total_savings(recs) / spent, 4)


def summarize(records: Iterable[UsageRecord], catalog: PricingCatalog,
              allow_tier_drop: int = 0) -> Dict:
    """One-shot optimisation summary used by the plugin's default view."""
    records = list(records)
    recs = recommendations(records, catalog, allow_tier_drop=allow_tier_drop)
    return {
        "recommendations": recs,
        "total_savings_usd": total_savings(recs),
        "savings_ratio": savings_ratio(recs, records, catalog),
        "spent_usd": analytics.totals(records, catalog)["cost_usd"],
        "quantified": sum(1 for rec in recs if rec.savings_usd is not None),
        "unquantified": sum(1 for rec in recs if rec.savings_usd is None),
    }
