"""Tests for the iteration-11 period-end simulation (pure logic)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import simulation
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog

NOW = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)


def rec(inp=1_000_000, out=0, day=1):
    """A record on day ``day`` of June 2024 (1-based)."""
    return UsageRecord(model="gpt-4o", input_tokens=inp, output_tokens=out,
                       timestamp=f"2024-06-{day:02d}T12:00:00")


# -- daily series -----------------------------------------------------------

def test_daily_costs_zero_fills_missing_days():
    """A day with no calls is a real zero-cost day, not a gap to skip."""
    records = [rec(day=1), rec(day=3)]
    series = simulation.daily_costs(records, default_catalog(), now=NOW)
    assert len(series["days"]) == 10          # June 1..10
    assert series["zero_days"] == 8
    assert series["values"][1] == 0.0          # June 2 had no usage


def test_daily_costs_totals_match_spend():
    records = [rec(day=1), rec(day=2)]
    series = simulation.daily_costs(records, default_catalog(), now=NOW)
    assert series["spent_usd"] == pytest.approx(5.0)   # 2 x $2.50
    assert series["mean_usd"] == pytest.approx(0.5)    # spread over 10 days


def test_daily_costs_empty_is_all_zeros():
    series = simulation.daily_costs([], default_catalog(), now=NOW)
    assert series["spent_usd"] == 0.0 and series["zero_days"] == len(series["values"])


def test_daily_costs_ignores_previous_month():
    records = [UsageRecord(model="gpt-4o", input_tokens=1_000_000,
                           output_tokens=0, timestamp="2024-05-20T12:00:00")]
    series = simulation.daily_costs(records, default_catalog(), now=NOW)
    assert series["spent_usd"] == 0.0


# -- days left --------------------------------------------------------------

@pytest.mark.parametrize("period,expected", [
    ("daily", 1),        # never zero: there is always a day of spending ahead
    ("weekly", 6),       # Monday Jun 10 elapsed -> Tue..Sun remaining
    ("monthly", 20),     # 30 days, 10 elapsed (incl. today) -> Jun 11..30
])
def test_days_left(period, expected):
    assert simulation.days_left(period, now=NOW) == expected


def test_days_left_never_negative():
    """On the last day of the period the floor of 1 keeps the maths usable."""
    assert simulation.days_left("daily", now=NOW) >= 1


# -- percentile -------------------------------------------------------------

def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert simulation.percentile(values, 0) == 1.0
    assert simulation.percentile(values, 100) == 5.0
    assert simulation.percentile(values, 50) == 3.0
    assert simulation.percentile(values, 25) == 2.0


def test_percentile_edge_cases():
    assert simulation.percentile([], 90) == 0.0
    assert simulation.percentile([7.0], 90) == 7.0


# -- simulation -------------------------------------------------------------

def test_simulate_is_reproducible_with_same_seed():
    records = [rec(day=d) for d in range(1, 11)]
    a = simulation.simulate(records, default_catalog(), now=NOW, seed=42)
    b = simulation.simulate(records, default_catalog(), now=NOW, seed=42)
    assert a["p90_total_usd"] == b["p90_total_usd"]


def test_simulate_seed_changes_result():
    """Different seeds must sample differently -- otherwise the seed is ignored.

    A single percentile can coincide by chance on a quantised series, so compare
    the mean of all draws, which averages 2000 samples and cannot coincide to
    six decimal places unless the RNG was never reseeded.
    """
    records = [rec(inp=1_000_000 * d, day=d) for d in range(1, 11)]
    a = simulation.simulate(records, default_catalog(), now=NOW, seed=1)
    b = simulation.simulate(records, default_catalog(), now=NOW, seed=2)
    assert a["mean_total_usd"] != b["mean_total_usd"]


def test_simulate_percentiles_are_ordered():
    records = [rec(inp=1_000_000 * d, day=d) for d in range(1, 11)]
    data = simulation.simulate(records, default_catalog(), now=NOW)
    ordered = [data["percentiles"][p] for p in simulation.REPORTED_PERCENTILES]
    assert ordered == sorted(ordered)


def test_simulate_never_below_spend_so_far():
    """Future cost cannot be negative, so the floor is what is already spent."""
    records = [rec(day=d) for d in range(1, 11)]
    data = simulation.simulate(records, default_catalog(), now=NOW,
                               method="normal")
    assert data["percentiles"][5] >= data["spent_usd"] - 1e-9


def test_simulate_exceed_probability_without_limit_is_none():
    """No limit means "unknown", not "0% chance of overspending"."""
    data = simulation.simulate([rec(day=1)], default_catalog(), now=NOW)
    assert data["exceed_probability"] is None


def test_simulate_exceed_probability_bounds():
    records = [rec(day=d) for d in range(1, 11)]
    data = simulation.simulate(records, default_catalog(), now=NOW, limit_usd=50.0)
    assert 0.0 <= data["exceed_probability"] <= 1.0


@pytest.mark.parametrize("limit,expected", [
    (1e9, 0.0),      # limit far above any plausible outcome
    (0.01, 1.0),     # limit below what is already spent
])
def test_simulate_exceed_probability_extremes(limit, expected):
    data = simulation.simulate([rec(day=1)], default_catalog(), now=NOW,
                               limit_usd=limit)
    assert data["exceed_probability"] == expected


def test_simulate_no_data_is_a_single_point():
    data = simulation.simulate([], default_catalog(), now=NOW, limit_usd=10.0)
    assert data["insufficient_data"] is True
    assert data["p90_total_usd"] == 0.0
    assert data["exceed_probability"] == 0.0


@pytest.mark.parametrize("kwargs", [
    {"method": "nonsense"},
    {"iterations": 0},
])
def test_simulate_rejects_bad_options(kwargs):
    with pytest.raises(ValueError):
        simulation.simulate([rec(day=1)], default_catalog(), now=NOW, **kwargs)


def test_normal_method_counts_clamped_days():
    """A flat series with a high stdev relative to mean will draw negatives."""
    records = [rec(inp=1_000_000, day=1), rec(inp=1_000_000, day=2)]
    data = simulation.simulate(records, default_catalog(), now=NOW,
                               method="normal", seed=3)
    assert data["clamped_days"] >= 0
    assert isinstance(data["clamped_days"], int)


# -- method comparison ------------------------------------------------------

def test_compare_methods_reports_spread():
    records = [rec(inp=1_000_000 * d, day=d) for d in range(1, 11)]
    data = simulation.compare_methods(records, default_catalog(), now=NOW)
    assert set(data["runs"]) == set(simulation.METHODS)
    assert data["p90_spread_usd"] >= 0
    assert data["agreement"] in ("close", "divergent")


def test_compare_methods_flags_divergence_on_spiky_series():
    """One huge day among zeros makes bootstrap and normal disagree sharply."""
    records = [rec(inp=50_000_000, day=1)]
    data = simulation.compare_methods(records, default_catalog(), now=NOW)
    assert data["p90_spread_usd"] > 0


# -- required cut -----------------------------------------------------------

def test_required_cut_reports_overage():
    cut = simulation.required_cut(120.0, 100.0)
    assert cut["cut_usd"] == pytest.approx(20.0)
    assert cut["cut_pct"] == pytest.approx(16.67, abs=0.01)
    assert cut["feasible"] is True


@pytest.mark.parametrize("projected,limit,feasible", [
    (80.0, 100.0, True),     # already inside the limit: nothing to cut
    (300.0, 100.0, False),   # >50% cut is a capability change, not tuning
])
def test_required_cut_feasibility(projected, limit, feasible):
    assert simulation.required_cut(projected, limit)["feasible"] is feasible


def test_required_cut_without_limit():
    cut = simulation.required_cut(80.0, 0.0)
    assert cut["feasible"] is False and cut["note"] == "no limit supplied"


def test_required_cut_marks_impossible_targets():
    cut = simulation.required_cut(300.0, 100.0)
    assert "capability change" in cut["note"]


# -- summary ----------------------------------------------------------------

def test_summarize_adds_cut_and_methods():
    records = [rec(day=d) for d in range(1, 11)]
    data = simulation.summarize(records, default_catalog(), now=NOW, limit_usd=50.0)
    assert "cut" in data and "methods" in data
    assert data["methods"]["period"] == "monthly"
    assert data["percentiles"][90] == data["p90_total_usd"]
