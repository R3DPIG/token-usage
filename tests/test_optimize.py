"""Tests for the iteration-8 optimisation layer (pure logic)."""

from __future__ import annotations

import pytest

from token_usage import optimize
from token_usage.models import ModelPricing, UsageRecord
from token_usage.pricing import PricingCatalog, default_catalog


def rec(model="gpt-4o", inp=1_000_000, out=0, cached=0, ts="2024-05-06T12:00:00"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, cached_input_tokens=cached)


def catalog_with(*entries):
    return PricingCatalog([ModelPricing(model=m, input_per_1m=i, output_per_1m=o,
                                        cached_input_per_1m=c)
                           for m, i, o, c in entries], default_model="cheap")


# -- price tiers ------------------------------------------------------------

@pytest.mark.parametrize("input_price,expected", [
    (15.0, "premium"), (5.0, "premium"), (3.0, "mid"), (1.0, "mid"),
    (0.15, "economy"),
])
def test_price_tier(input_price, expected):
    pricing = ModelPricing(model="m", input_per_1m=input_price, output_per_1m=0)
    assert optimize.price_tier(pricing) == expected


def test_tier_index_orders_cheapest_last():
    assert optimize.tier_index("premium") < optimize.tier_index("mid")
    assert optimize.tier_index("mid") < optimize.tier_index("economy")


def test_generates_output_flags_embedding_models():
    catalog = default_catalog()
    assert optimize.generates_output(catalog.require("gpt-4o")) is True
    assert optimize.generates_output(catalog.require("text-embedding-3-small")) is False


# -- model substitution -----------------------------------------------------

def test_substitution_within_same_tier():
    catalog = catalog_with(("expensive", 3.0, 15.0, 0.3),
                           ("cheap", 1.0, 5.0, 0.1))
    results = optimize.model_substitutions([rec(model="expensive")], catalog)
    assert len(results) == 1
    assert results[0]["alternative"] == "cheap"
    assert results[0]["savings_usd"] > 0
    assert results[0]["tier_drop"] == 0


def test_substitution_tier_drop_gate():
    """A cross-tier swap is only offered when explicitly allowed."""
    catalog = catalog_with(("premium", 15.0, 75.0, 1.5),
                           ("economy", 0.15, 0.6, 0.075))
    assert optimize.model_substitutions([rec(model="premium")], catalog) == []
    results = optimize.model_substitutions([rec(model="premium")], catalog,
                                           allow_tier_drop=2)
    assert len(results) == 1 and results[0]["tier_drop"] == 2


def test_substitution_picks_largest_saving():
    catalog = catalog_with(("current", 3.0, 15.0, 0.3),
                           ("mid-cheap", 2.0, 10.0, 0.2),
                           ("cheapest", 1.0, 5.0, 0.1))
    results = optimize.model_substitutions([rec(model="current")], catalog)
    assert results[0]["alternative"] == "cheapest"


def test_substitution_skips_unusable_inputs():
    """No records, unknown model, or a saving below the floor -> no advice."""
    assert optimize.model_substitutions([], default_catalog()) == []
    assert optimize.model_substitutions([rec(model="not-in-catalog")],
                                        catalog_with(("a", 1.0, 1.0, 0.1))) == []
    catalog = catalog_with(("a", 3.0, 15.0, 0.3), ("b", 2.9, 14.5, 0.29))
    assert optimize.model_substitutions([rec(model="a", inp=1000)], catalog,
                                        min_savings_usd=100.0) == []


def test_substitution_never_suggests_embedding_model():
    """An embedding model bills no output, so it cannot replace a generator."""
    records = [rec(model="gpt-4o", inp=1_000_000, out=500_000)]
    results = optimize.model_substitutions(records, default_catalog(),
                                           allow_tier_drop=2)
    assert all("text-embedding" not in item["alternative"] for item in results)


# -- cache opportunities ----------------------------------------------------

def test_cache_opportunity_found():
    results = optimize.cache_opportunities([rec(model="gpt-4o", inp=1_000_000)],
                                           default_catalog())
    assert len(results) == 1
    assert results[0]["savings_usd"] > 0


@pytest.mark.parametrize("records,catalog", [
    ([rec(model="gpt-4o", inp=1_000_000, cached=500_000)], default_catalog()),
    ([rec(model="gpt-4o", inp=100)], default_catalog()),
    ([rec(model="flat")], catalog_with(("flat", 1.0, 1.0, 1.0))),
])
def test_cache_skipped(records, catalog):
    """Already caching / too little traffic / no cache discount -> no advice."""
    assert optimize.cache_opportunities(records, catalog) == []


# -- repeated shapes --------------------------------------------------------

def test_repeated_shapes_detected():
    records = [rec(model="gpt-4o", inp=10_000) for _ in range(4)]
    results = optimize.repeated_shapes(records, default_catalog(), min_repeats=3)
    assert len(results) == 1
    assert results[0]["repeats"] == 3
    assert results[0]["savings_usd"] > 0


@pytest.mark.parametrize("records", [
    [rec(model="gpt-4o", inp=10_000) for _ in range(2)],
    [rec(model="gpt-4o", inp=10_000), rec(model="gpt-4o", inp=20_000),
     rec(model="gpt-4o", inp=30_000)],
])
def test_repeated_shapes_not_grouped(records):
    """Too few repeats, or distinct sizes, must not be treated as one prompt."""
    assert optimize.repeated_shapes(records, default_catalog(), min_repeats=3) == []


# -- small calls ------------------------------------------------------------

def test_small_calls_counts_and_reports_no_saving():
    records = [rec(inp=100, out=50), rec(inp=100_000, out=50_000)]
    result = optimize.small_calls(records, threshold_tokens=500)
    assert result["count"] == 1
    assert result["share"] == 0.5
    assert result["savings_usd"] is None      # deliberately not fabricated


def test_small_calls_empty():
    result = optimize.small_calls([])
    assert result["count"] == 0 and result["savings_usd"] is None


# -- recommendations --------------------------------------------------------

def test_recommendations_ranked_by_saving():
    records = [rec(model="gpt-4o", inp=10_000_000)]
    recs = optimize.recommendations(records, default_catalog())
    quantified = [r for r in recs if r.savings_usd is not None]
    assert quantified == sorted(quantified, key=lambda r: r.savings_usd,
                                reverse=True)


def test_recommendations_unquantified_sort_last():
    records = [rec(model="gpt-4o", inp=100, out=10) for _ in range(5)]
    recs = optimize.recommendations(records, default_catalog())
    if any(r.savings_usd is None for r in recs):
        assert recs[-1].savings_usd is None


def test_recommendation_validates_confidence_and_effort():
    with pytest.raises(ValueError):
        optimize.Recommendation(kind="x", title="t", detail="d",
                                savings_usd=1.0, confidence="certain")
    with pytest.raises(ValueError):
        optimize.Recommendation(kind="x", title="t", detail="d",
                                savings_usd=1.0, effort="trivial")


def test_recommendation_to_dict():
    rec_obj = optimize.Recommendation(kind="cache", title="t", detail="d",
                                      savings_usd=1.5, evidence={"a": 1})
    data = rec_obj.to_dict()
    assert data["kind"] == "cache" and data["savings_usd"] == 1.5
    assert data["evidence"] == {"a": 1}


def test_recommendations_include_kinds():
    records = [rec(model="gpt-4o", inp=1_000_000, out=0)]
    kinds = {r.kind for r in optimize.recommendations(records, default_catalog())}
    assert {"cache", "waste"} <= kinds


def test_recommendations_empty_records():
    assert optimize.recommendations([], default_catalog()) == []


# -- totals -----------------------------------------------------------------

def test_total_savings_ignores_none_and_adds_model_agnostic():
    recs = [optimize.Recommendation(kind="cache", title="t", detail="d",
                                    savings_usd=10.0, model="m"),
            optimize.Recommendation(kind="waste", title="t", detail="d",
                                    savings_usd=3.0),
            optimize.Recommendation(kind="batch", title="t", detail="d",
                                    savings_usd=None)]
    assert optimize.total_savings(recs) == 13.0


def test_savings_ratio_never_exceeds_one():
    """Overlapping recommendations must not sum past total spend."""
    records = [rec(model="gpt-4o", inp=1_000_000)]
    recs = optimize.recommendations(records, default_catalog())
    ratio = optimize.savings_ratio(recs, records, default_catalog())
    assert 0.0 <= ratio <= 1.0


def test_total_savings_dedupes_per_model():
    """Caching and substituting the same model are alternatives, not additive."""
    recs = [optimize.Recommendation(kind="cache", title="t", detail="d",
                                    savings_usd=10.0, model="m"),
            optimize.Recommendation(kind="substitute", title="t", detail="d",
                                    savings_usd=6.0, model="m")]
    assert optimize.total_savings(recs) == 10.0     # the larger, not the sum


def test_savings_ratio_zero_spend():
    assert optimize.savings_ratio([], [], default_catalog()) == 0.0


def test_summarize_shape():
    records = [rec(model="gpt-4o", inp=1_000_000)]
    data = optimize.summarize(records, default_catalog())
    for key in ("recommendations", "total_savings_usd", "savings_ratio",
                "spent_usd", "quantified", "unquantified"):
        assert key in data
    assert data["spent_usd"] > 0
