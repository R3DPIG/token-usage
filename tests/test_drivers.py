"""Tests for the iteration-10 spend-driver decomposition (pure logic)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from token_usage import drivers
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog

NOW = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)


def rec(model="gpt-4o", inp=1_000_000, out=0, ts="2024-06-10T12:00:00"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts)


# -- period splitting -------------------------------------------------------

def test_split_periods_uses_equal_windows():
    """The previous window mirrors the elapsed part of the current one."""
    records = [rec(ts="2024-06-05T00:00:00"),     # current (>= 06-01)
               rec(ts="2024-05-20T00:00:00"),     # previous (05-16..06-01)
               rec(ts="2024-05-01T00:00:00")]     # too old
    cur, prev = drivers.split_periods(records, "monthly", now=NOW)
    assert len(cur) == 1 and len(prev) == 1


def test_split_periods_rejects_unknown_period():
    with pytest.raises(ValueError):
        drivers.split_periods([], "yearly", now=NOW)


# -- decomposition identity -------------------------------------------------

def test_decompose_balances_when_volume_grows():
    records = [rec(inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(inp=2_000_000, ts="2024-06-10T12:00:00")]
    data = drivers.decompose(records, default_catalog(), now=NOW)
    assert data["balanced"] is True
    assert data["drift_usd"] == 0.0
    assert data["drivers"]["volume"] > 0


def test_decompose_balances_when_mix_shifts():
    """Same token count, but the dearer model takes over."""
    records = [rec(model="gpt-4o-mini", inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(model="claude-3-opus", inp=1_000_000, ts="2024-06-10T12:00:00")]
    data = drivers.decompose(records, default_catalog(), now=NOW)
    assert data["balanced"] is True
    assert data["drivers"]["mix"] > 0


def test_decompose_is_flat_with_identical_periods():
    records = [rec(inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(inp=1_000_000, ts="2024-06-10T12:00:00")]
    data = drivers.decompose(records, default_catalog(), now=NOW)
    assert data["delta_usd"] == 0.0
    assert drivers.dominant_driver(data["drivers"]) == "flat"


def test_decompose_handles_empty_periods():
    data = drivers.decompose([], default_catalog(), now=NOW)
    assert data["delta_usd"] == 0.0 and data["balanced"] is True
    assert data["records"] == {"current": 0, "previous": 0}
    assert data["no_baseline"] is True


def test_no_baseline_flags_one_sided_comparison():
    """With no history the identity pushes everything into ``interaction``;
    that must be flagged rather than narrated as a real decomposition."""
    records = [rec(inp=1_000_000, ts="2024-06-10T12:00:00")]   # current only
    data = drivers.decompose(records, default_catalog(), now=NOW)
    assert data["no_baseline"] is True
    assert data["drivers"]["interaction"] == data["delta_usd"]
    assert data["balanced"] is True            # the arithmetic is still sound


def test_no_baseline_clears_when_both_periods_have_data():
    records = [rec(inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(inp=2_000_000, ts="2024-06-10T12:00:00")]
    assert drivers.decompose(records, default_catalog(), now=NOW)["no_baseline"] is False


@pytest.mark.parametrize("current,previous", [
    ([rec(inp=3_000_000, ts="2024-06-10T12:00:00")],
     [rec(inp=1_000_000, ts="2024-05-20T12:00:00")]),
    ([rec(model="gpt-4o-mini", inp=5_000_000, ts="2024-06-10T12:00:00")],
     [rec(model="claude-3-opus", inp=2_000_000, ts="2024-05-20T12:00:00")]),
    ([rec(model="claude-3-opus", inp=2_000_000, ts="2024-06-10T12:00:00")],
     [rec(model="gpt-4o-mini", inp=5_000_000, ts="2024-05-20T12:00:00")]),
])
def test_decompose_identity_holds(current, previous):
    """The four terms must always re-add to the observed delta."""
    data = drivers.decompose(current + previous, default_catalog(), now=NOW)
    total = sum(data["drivers"].values())
    assert total == pytest.approx(data["delta_usd"], abs=1e-6)
    assert data["balanced"] is True


# -- unit price -------------------------------------------------------------

def test_unit_prices_flags_assumed_when_model_missing():
    """A model present in one period only must be flagged, not silently priced."""
    current = {"gpt-4o": {"tokens": 1_000_000, "cost_usd": 2.5, "records": 1}}
    prices = drivers.unit_prices(current, {})
    assert prices["gpt-4o"]["assumed_price"] is True
    assert prices["gpt-4o"]["price_per_1m"] == pytest.approx(2.5)


def test_assumed_prices_surface_in_decompose():
    records = [rec(inp=1_000_000, ts="2024-06-10T12:00:00")]   # new model only
    data = drivers.decompose(records, default_catalog(), now=NOW)
    assert "gpt-4o" in data["assumed_prices"]


# -- dominant driver --------------------------------------------------------

@pytest.mark.parametrize("values,expected", [
    ({"volume": 5.0, "mix": 0.0, "unit_price": 1.0, "interaction": 0.0}, "volume"),
    ({"volume": 0.0, "mix": -3.0, "unit_price": 2.0, "interaction": 0.0}, "mix"),
    ({"volume": 0.0, "mix": 0.0, "unit_price": 0.0, "interaction": 0.0}, "flat"),
])
def test_dominant_driver(values, expected):
    assert drivers.dominant_driver(values) == expected


# -- model contributions ----------------------------------------------------

def test_model_contributions_sorted_and_labelled():
    records = [rec(model="gpt-4o", inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(model="gpt-4o", inp=1_000_000, ts="2024-06-10T12:00:00"),
               rec(model="claude-3-opus", inp=1_000_000, ts="2024-06-10T12:00:00")]
    rows = drivers.model_contributions(records, default_catalog(), now=NOW)
    deltas = [abs(row["delta_usd"]) for row in rows]
    assert deltas == sorted(deltas, reverse=True)
    statuses = {row["model"]: row["status"] for row in rows}
    assert statuses["gpt-4o"] == "ongoing"
    assert statuses["claude-3-opus"] == "new"


def test_model_contributions_respect_limit():
    records = [rec(model=m, inp=1_000_000, ts="2024-06-10T12:00:00")
               for m in ("gpt-4o", "gpt-4o-mini", "claude-3-opus")]
    rows = drivers.model_contributions(records, default_catalog(), now=NOW,
                                       limit=2)
    assert len(rows) == 2


# -- mix shift --------------------------------------------------------------

def test_mix_shift_reports_share_delta():
    records = [rec(model="gpt-4o", inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(model="gpt-4o-mini", inp=3_000_000, ts="2024-06-10T12:00:00")]
    rows = {row["model"]: row
            for row in drivers.mix_shift(records, default_catalog(), now=NOW)}
    assert rows["gpt-4o-mini"]["share_current"] == 1.0
    assert rows["gpt-4o"]["share_current"] == 0.0
    assert rows["gpt-4o"]["share_delta"] == -1.0


def test_mix_shift_is_token_based_not_dollar_based():
    """A single model that doubles in volume keeps 100% share."""
    records = [rec(inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(inp=2_000_000, ts="2024-06-10T12:00:00")]
    rows = drivers.mix_shift(records, default_catalog(), now=NOW)
    assert rows[0]["share_delta"] == 0.0


# -- explain ----------------------------------------------------------------

def test_explain_reports_flat_when_unchanged():
    data = drivers.decompose([], default_catalog(), now=NOW)
    assert drivers.explain(data) == ["spend is unchanged period over period"]


def test_explain_says_no_baseline_instead_of_narrating():
    """A one-sided comparison must not claim volume and price moved together."""
    records = [rec(inp=1_000_000, ts="2024-06-10T12:00:00")]
    lines = drivers.explain(drivers.decompose(records, default_catalog(), now=NOW))
    assert len(lines) == 1
    assert "new spend" in lines[0] and "no comparable baseline" in lines[0]
    assert not any("interaction" in line for line in lines)


def test_explain_reports_spend_disappearing():
    records = [rec(inp=1_000_000, ts="2024-05-20T12:00:00")]
    lines = drivers.explain(drivers.decompose(records, default_catalog(), now=NOW))
    assert "spend disappeared" in lines[0]


def test_explain_puts_the_direction_first():
    records = [rec(inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(inp=3_000_000, ts="2024-06-10T12:00:00")]
    lines = drivers.explain(drivers.decompose(records, default_catalog(), now=NOW))
    assert lines[0].startswith("spend is up")
    assert any("volume" in line for line in lines)
    assert not any("model mix" in line for line in lines)   # zero drivers omitted


def test_explain_warns_about_assumed_prices():
    records = [rec(model="gpt-4o-mini", inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(model="gpt-4o", inp=1_000_000, ts="2024-06-10T12:00:00")]
    lines = drivers.explain(drivers.decompose(records, default_catalog(), now=NOW))
    assert any("assumed" in line for line in lines)


# -- summary ----------------------------------------------------------------

def test_summarize_shape():
    records = [rec(inp=1_000_000, ts="2024-05-20T12:00:00"),
               rec(inp=2_000_000, ts="2024-06-10T12:00:00")]
    data = drivers.summarize(records, default_catalog(), now=NOW)
    for key in ("drivers", "delta_usd", "balanced", "dominant", "lines", "models"):
        assert key in data
    assert data["balanced"] is True and data["dominant"] == "volume"
