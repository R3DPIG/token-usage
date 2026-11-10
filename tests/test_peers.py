"""Unit tests for the iteration-13 peer benchmarking module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import peers
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog

NOW = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
CATALOG = default_catalog()


def ts(hours_ago: int = 1) -> str:
    return (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def rec(model="gpt-4o", inp=1_000_000, out=0, cached=0, hours_ago=1) -> UsageRecord:
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       cached_input_tokens=cached, timestamp=ts(hours_ago),
                       project="api", tags=["api"])


def dataset(*, model="gpt-4o", inp=1_000_000, out=0, cached=0, calls=1,
            start=1) -> list:
    return [rec(model=model, inp=inp, out=out, cached=cached, hours_ago=start + i)
            for i in range(calls)]


def flat_row(name: str, **overrides) -> dict:
    """A complete metric row for ranking tests that never touch pricing."""
    row = {"peer": name, "cost_per_1m": 5.0, "cost_per_call": 1.0,
           "output_ratio": 0.5, "cache_ratio": 0.0, "cost_usd": 1.0, "tokens": 10}
    row.update(overrides)
    return row


def sample_rows():
    """Three peers on one model, differing only in how much they generate."""
    return peers.build_profiles({
        "cheap": dataset(out=0),
        "mid": dataset(out=500_000),
        "dear": dataset(out=1_000_000),
    }, CATALOG)


# -- profiling --------------------------------------------------------------

def test_profile_normalises_scale_into_efficiency():
    row = peers.profile("a", dataset(inp=1_000_000, out=1_000_000), CATALOG)
    assert row["tokens"] == 2_000_000
    # gpt-4o: $2.50/1M in + $10.00/1M out => $12.50 for this shape
    assert row["cost_per_1m"] == 6.25
    assert row["output_ratio"] == 0.5 and row["eligible"] is True


def test_profile_keeps_a_tiny_peer_but_marks_it_ineligible():
    row = peers.profile("tiny", dataset(inp=1_000), CATALOG)
    assert row["eligible"] is False
    assert row["cost_per_1m"] > 0            # still computed, just not trusted


def test_profile_of_no_records_is_all_zeros():
    row = peers.profile("empty", [], CATALOG)
    assert row["tokens"] == 0 and row["cost_per_1m"] == 0.0
    assert row["cost_per_call"] == 0.0 and row["eligible"] is False


def test_profile_reports_cache_ratio():
    row = peers.profile("a", dataset(inp=1_000_000, cached=250_000), CATALOG)
    assert row["cache_ratio"] == 0.25


def test_build_profiles_orders_by_spend_and_eligible_filters():
    rows = peers.build_profiles({
        "small": dataset(inp=1_000_000),
        "big": dataset(inp=5_000_000),
        "tiny": dataset(inp=100),
    }, CATALOG)
    assert [row["peer"] for row in rows] == ["big", "small", "tiny"]
    assert [row["peer"] for row in peers.eligible(rows)] == ["big", "small"]


# -- ranking ----------------------------------------------------------------

@pytest.mark.parametrize("metric,order", [
    ("cost_per_1m", ["cheap", "mid", "dear"]),      # lower is better
    ("output_ratio", ["dear", "mid", "cheap"]),     # higher is better
])
def test_rank_respects_metric_direction(metric, order):
    ranked = peers.rank(sample_rows(), metric)
    assert [row["peer"] for row in ranked] == order
    assert ranked[0]["percentile"] == 1.0     # best, whichever direction
    assert ranked[-1]["percentile"] == 0.0


def test_rank_gives_ties_the_same_rank():
    rows = [flat_row("a"), flat_row("b"), flat_row("c", cost_per_1m=9.0)]
    assert [row["rank"] for row in peers.rank(rows, "cost_per_1m")] == [1, 1, 3]


def test_rank_of_one_peer_is_trivially_best():
    ranked = peers.rank(peers.build_profiles({"only": dataset()}, CATALOG),
                        "cost_per_1m")
    assert ranked[0]["rank"] == 1 and ranked[0]["percentile"] == 1.0


def test_rank_rejects_an_unknown_metric():
    with pytest.raises(ValueError):
        peers.rank(sample_rows(), "nonsense")


# -- spread -----------------------------------------------------------------

def test_spread_reports_the_multiple_between_extremes():
    data = peers.spread(sample_rows(), "cost_per_1m")
    assert data["count"] == 3
    assert data["min"] < data["median"] < data["max"]
    assert data["multiple"] > 1.0
    assert data["best"] == "cheap" and data["worst"] == "dear"


def test_spread_of_identical_peers_has_zero_variation():
    data = peers.spread([flat_row("a"), flat_row("b"), flat_row("c")],
                        "cost_per_1m")
    assert data["multiple"] == 1.0 and data["cv"] == 0.0


def test_spread_of_nothing_is_empty_not_an_error():
    data = peers.spread([], "cost_per_1m")
    assert data["count"] == 0 and data["best"] is None


# -- savings ----------------------------------------------------------------

def test_savings_reprices_each_peer_at_the_best_rate():
    by_peer = {item["peer"]: item
               for item in peers.savings_vs_best(sample_rows())}
    assert by_peer["cheap"]["savings_usd"] == 0.0
    assert by_peer["dear"]["savings_usd"] > 0
    assert by_peer["dear"]["benchmark"] == "cheap"


def test_savings_never_goes_negative_and_states_its_assumption():
    items = peers.savings_vs_best(sample_rows())
    assert all(item["savings_usd"] >= 0.0 for item in items)
    assert all(item["savings_pct"] >= 0.0 for item in items)
    assert all("assumes" in item["note"] for item in items)


def test_savings_by_cost_per_call_works():
    rows = peers.build_profiles({"a": dataset(calls=10), "b": dataset(calls=10)},
                                CATALOG)
    items = peers.savings_vs_best(rows, metric="cost_per_call")
    assert len(items) == 2 and all(item["savings_usd"] == 0.0 for item in items)


def test_savings_rejects_a_metric_it_cannot_reprice():
    with pytest.raises(ValueError):
        peers.savings_vs_best(sample_rows(), metric="output_ratio")


# -- outliers ---------------------------------------------------------------

def test_outliers_need_at_least_three_peers():
    assert peers.outliers(sample_rows()[:2], "cost_per_1m") == []


def test_outliers_flag_the_extreme_peer_only():
    rows = [flat_row("a"), flat_row("b", cost_per_1m=5.1),
            flat_row("c", cost_per_1m=4.9), flat_row("z", cost_per_1m=400.0)]
    flagged = peers.outliers(rows, "cost_per_1m")
    assert [item["peer"] for item in flagged] == ["z"]
    assert flagged[0]["direction"] == "high" and flagged[0]["basis"] == "mad"


def test_outliers_on_identical_peers_finds_nothing():
    assert peers.outliers([flat_row("a"), flat_row("b"), flat_row("c")],
                          "cost_per_1m") == []


def test_outliers_fall_back_to_a_ratio_when_the_mad_is_zero():
    """Three peers on one price and one far above is the common real case.

    Only one value differs, so the MAD is zero and no z-score exists. Reporting
    "no outlier" there would hide the single peer worth investigating.
    """
    rows = [flat_row("a"), flat_row("b"), flat_row("c"),
            flat_row("z", cost_per_1m=90.0)]
    flagged = peers.outliers(rows, "cost_per_1m")
    assert [item["peer"] for item in flagged] == ["z"]
    assert flagged[0]["basis"] == "ratio" and flagged[0]["score"] == 18.0


# -- comparability ----------------------------------------------------------

def test_workload_warning_stays_quiet_for_like_for_like_peers():
    rows = peers.build_profiles({"a": dataset(out=800_000),
                                 "b": dataset(out=850_000)}, CATALOG)
    assert peers.workload_warning(rows) is None


def test_workload_warning_fires_when_output_ratios_diverge():
    rows = peers.build_profiles({
        "chat": dataset(out=1_000_000),
        "embed": dataset(model="text-embedding-3-small", out=0),
    }, CATALOG)
    warning = peers.workload_warning(rows)
    assert warning is not None and "not doing the same kind of work" in warning


def test_efficiency_note_says_when_the_spread_is_small():
    rows = peers.build_profiles({"a": dataset(out=500_000),
                                 "b": dataset(out=500_000)}, CATALOG)
    assert "within 20%" in peers.efficiency_note(rows)


def test_efficiency_note_points_at_both_causes_when_the_spread_is_wide():
    note = peers.efficiency_note(sample_rows())
    assert "model choice" in note and "cache" in note


# -- summary ----------------------------------------------------------------

def test_summarize_flags_incomparable_workloads():
    result = peers.summarize({
        "chat": dataset(out=1_000_000),
        "embed": dataset(model="text-embedding-3-small", out=0),
    }, CATALOG)
    assert result["comparable"] is False
    assert any("same kind of work" in text for text in result["warnings"])


def test_summarize_totals_cover_every_peer_but_benchmark_only_the_eligible():
    result = peers.summarize({"real": dataset(), "tiny": dataset(inp=100)},
                             CATALOG)
    assert result["total_peers"] == 2 and result["eligible_count"] == 1
    assert result["total_tokens"] == 1_000_100


def test_summarize_of_a_single_peer_reports_no_warnings():
    result = peers.summarize({"only": dataset(calls=3)}, CATALOG)
    assert result["warnings"] == [] and result["total_peers"] == 1
