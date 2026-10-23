"""Core-layer tests: models, pricing, utils, analytics (no I/O)."""

from __future__ import annotations

import pytest

from token_usage import analytics
from token_usage.models import Budget, ModelPricing, UsageRecord
from token_usage.pricing import PricingCatalog, default_catalog
from token_usage.utils import day_key, month_key, parse_iso, period_start, week_key, bar


def make_record(**overrides) -> UsageRecord:
    base = dict(model="gpt-4o", input_tokens=1_000_000, output_tokens=500_000,
                timestamp="2024-05-01T12:00:00", cached_input_tokens=0, project="demo")
    base.update(overrides)
    return UsageRecord(**base)


# -- models / pricing -------------------------------------------------------

def test_record_totals_and_billable():
    record = make_record(cached_input_tokens=200_000)
    assert (record.total_tokens, record.billable_input_tokens) == (1_500_000, 800_000)


@pytest.mark.parametrize("kwargs", [
    {"input_tokens": -1}, {"cached_input_tokens": -1},
    {"input_tokens": 100, "cached_input_tokens": 200}])
def test_record_rejects_bad_tokens(kwargs):
    with pytest.raises(ValueError):
        make_record(**kwargs)


def test_record_roundtrip():
    record = make_record(tags=["api", "batch"], note="hi")
    assert UsageRecord.from_dict(record.to_dict()) == record


def test_pricing_validation_and_per_1k():
    with pytest.raises(ValueError):
        ModelPricing(model="x", input_per_1m=-1, output_per_1m=1)
    with pytest.raises(ValueError):
        ModelPricing(model="", input_per_1m=1, output_per_1m=1)
    assert ModelPricing("m", 10.0, 20.0, 2.0).per_1k()["input_per_1k"] == 0.01


def test_budget_validation():
    with pytest.raises(ValueError):
        Budget(limit_usd=0)
    with pytest.raises(ValueError):
        Budget(limit_usd=1, period="yearly")


def test_cost_uses_cached_rate():
    catalog = PricingCatalog([ModelPricing("m", 10.0, 30.0, 1.0)])
    assert catalog.cost_for("m", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(31.0)


def test_catalog_fallback_and_roundtrip():
    catalog = default_catalog()
    assert catalog.get("made-up-model") is not None
    restored = PricingCatalog.from_json(catalog.to_json())
    assert restored.models() == catalog.models()
    assert restored.default_model == catalog.default_model


# -- utils ------------------------------------------------------------------

def test_parse_iso_variants():
    assert parse_iso("2024-05-01T00:00:00").year == 2024
    assert parse_iso("2024-05-01T00:00:00Z").hour == 0
    assert parse_iso("2024-05-01T08:00:00+08:00").hour == 0


def test_period_keys_and_start():
    dt = parse_iso("2024-05-17T12:00:00")
    assert day_key(dt) == "2024-05-17"
    assert month_key(dt) == "2024-05"
    assert week_key(dt).startswith("2024-W")
    assert day_key(period_start("monthly", now=dt)) == "2024-05-01"


def test_bar_clamps():
    assert bar(0, 0) == ""
    assert (len(bar(50, 100, width=10)), len(bar(999, 100, width=10))) == (5, 10)


def test_filter_and_totals():
    records = [make_record(project="a"),
               make_record(project="b", input_tokens=0, output_tokens=0)]
    kept = analytics.filter_records(records, project="a")
    total = analytics.totals(kept, default_catalog())
    assert (len(kept), total["total_tokens"]) == (1, 1_500_000)
    assert total["cost_usd"] > 0


def test_group_ordering():
    catalog = default_catalog()
    records = [make_record(model="gpt-4o", input_tokens=10_000_000, output_tokens=0),
               make_record(model="gpt-4o-mini", input_tokens=1_000, output_tokens=0)]
    assert list(analytics.by_model(records, catalog))[0] == "gpt-4o"
    days = analytics.by_day([make_record(timestamp="2024-05-02T00:00:00"),
                             make_record(timestamp="2024-05-01T00:00:00")], catalog)
    assert list(days) == ["2024-05-01", "2024-05-02"]


def test_cache_savings_positive():
    savings = analytics.cache_savings([make_record(cached_input_tokens=500_000)],
                                      default_catalog())
    assert savings["cached_input_tokens"] == 500_000 and savings["saved_usd"] > 0


def test_forecast_flat_and_empty():
    flat = analytics.linear_forecast(
        {f"2024-05-0{i}": {"cost_usd": 1.0} for i in range(1, 5)}, days_ahead=5)
    assert flat["projected_total"] == pytest.approx(5.0, abs=0.01)
    empty = analytics.linear_forecast({}, days_ahead=10)
    assert empty["samples"] == 0 and empty["projected_total"] == 0.0


def test_evaluate_budgets_status():
    results = analytics.evaluate_budgets(
        [make_record(input_tokens=10_000_000, output_tokens=0)],
        default_catalog(), [Budget(limit_usd=0.001, period="total")])
    assert results[0]["status"] == "exceeded" and results[0]["remaining_usd"] == 0.0


def test_unknown_grouping_raises():
    with pytest.raises(ValueError):
        analytics.group([], "nonsense", default_catalog())
