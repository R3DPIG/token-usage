"""Tests for the iteration-2 insights layer and its CLI plugins."""

from __future__ import annotations

import json

import pytest

from token_usage import cli, insights
from token_usage.models import Budget, UsageRecord
from token_usage.pricing import default_catalog
from token_usage.storage import JsonStore


def rec(model="gpt-4o", inp=1_000_000, out=500_000, ts="2024-05-01T12:00:00",
        cached=0, project="demo", tags=None):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, cached_input_tokens=cached, project=project,
                       tags=tags or [])


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


# -- insights ---------------------------------------------------------------

def test_detect_anomalies_flags_spike():
    catalog = default_catalog()
    records = [rec(ts=f"2024-05-0{i}T00:00:00", inp=1_000_000, out=0) for i in range(1, 6)]
    records.append(rec(ts="2024-05-06T00:00:00", inp=50_000_000, out=0))  # spike
    flagged = insights.detect_anomalies(records, catalog, z_threshold=1.5)
    assert flagged and flagged[0]["day"] == "2024-05-06"
    assert flagged[0]["direction"] == "spike"


def test_detect_anomalies_needs_min_samples():
    catalog = default_catalog()
    records = [rec(ts="2024-05-01T00:00:00"), rec(ts="2024-05-02T00:00:00")]
    assert insights.detect_anomalies(records, catalog, min_samples=5) == []


def test_detect_anomalies_zero_variance_returns_empty():
    catalog = default_catalog()
    records = [rec(ts=f"2024-05-0{i}T00:00:00", inp=1_000_000, out=0) for i in range(1, 7)]
    assert insights.detect_anomalies(records, catalog) == []


def test_compare_periods_rejects_bad_period():
    with pytest.raises(ValueError):
        insights.compare_periods([], default_catalog(), period="yearly")


def test_compare_periods_direction():
    from datetime import datetime, timezone
    catalog = default_catalog()
    now = datetime(2024, 5, 15, tzinfo=timezone.utc)
    records = [rec(ts="2024-05-10T00:00:00", inp=10_000_000, out=0),   # current
               rec(ts="2024-04-10T00:00:00", inp=1_000_000, out=0)]    # previous
    result = insights.compare_periods(records, catalog, period="monthly", now=now)
    assert result["direction"] == "up" and result["delta_usd"] > 0


def test_model_efficiency_sorted_cheapest_first():
    catalog = default_catalog()
    records = [rec(model="gpt-4o", inp=1_000_000, out=0),
               rec(model="gpt-4o-mini", inp=1_000_000, out=0)]
    rows = insights.model_efficiency(records, catalog)
    assert rows[0]["model"] == "gpt-4o-mini"
    assert rows[0]["cost_per_1k_tokens"] < rows[1]["cost_per_1k_tokens"]


def test_concentration_share():
    catalog = default_catalog()
    records = [rec(model="gpt-4o", inp=10_000_000, out=0),
               rec(model="gpt-4o-mini", inp=1_000, out=0)]
    result = insights.concentration(records, catalog, dimension="model", top=1)
    assert result["top"][0]["name"] == "gpt-4o"
    assert 0.9 < result["top_share"] <= 1.0


def test_concentration_empty_is_safe():
    result = insights.concentration([], default_catalog())
    assert result["total_usd"] == 0.0 and result["top_share"] == 0.0


def test_waste_estimate_flags_low_output():
    catalog = default_catalog()
    records = [rec(inp=1_000_000, out=0),          # 0% output -> waste
               rec(inp=1_000_000, out=1_000_000)]  # 50% output -> fine
    result = insights.waste_estimate(records, catalog)
    assert result["count"] == 1 and result["wasted_usd"] > 0


# -- CLI plugins ------------------------------------------------------------

def test_cli_import_dry_run(store, tmp_path, capsys):
    export = tmp_path / "export.json"
    export.write_text(json.dumps([{"model": "gpt-4o", "input_tokens": 100,
                                   "output_tokens": 50}]), encoding="utf-8")
    code = cli.main(["--home", str(store.home), "import",
                     "--file", str(export), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0 and "dry run" in out
    assert store.count_records() == 0  # nothing written


def test_cli_import_writes_records(store, tmp_path, capsys):
    export = tmp_path / "export.json"
    export.write_text(json.dumps([{"model": "gpt-4o", "input_tokens": 100,
                                   "output_tokens": 50}]), encoding="utf-8")
    assert cli.main(["--home", str(store.home), "import", "--file", str(export)]) == 0
    assert store.count_records() == 1


def test_cli_import_missing_file(store, capsys):
    assert cli.main(["--home", str(store.home), "import", "--file", "/nope.json"]) == 2


def test_cli_compare_and_anomaly(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "sample", "--days", "10", "--per-day", "3"])
    capsys.readouterr()
    assert cli.main(["--home", home, "compare", "--period", "monthly"]) == 0
    assert "comparison" in capsys.readouterr().out
    assert cli.main(["--home", home, "anomaly"]) == 0
    assert "scanned" in capsys.readouterr().out


def test_cli_tags(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "sample", "--days", "3", "--per-day", "2"])
    capsys.readouterr()
    assert cli.main(["--home", home, "tags"]) == 0
    assert "distinct tag" in capsys.readouterr().out


def test_cli_doctor_clean_and_with_problems(store, capsys):
    home = str(store.home)
    # Empty workspace -> doctor reports no records and exits 1.
    assert cli.main(["--home", home, "doctor"]) == 1
    capsys.readouterr()

    cli.main(["--home", home, "sample", "--days", "10", "--per-day", "4"])
    capsys.readouterr()
    code = cli.main(["--home", home, "doctor"])
    out = capsys.readouterr().out
    assert "token-usage doctor" in out
    assert code in (0, 1)


def test_cli_doctor_flags_exceeded_budget(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "record", "--model", "gpt-4o",
              "--input", "10000000", "--output", "0"])
    store.save_budgets([Budget(limit_usd=0.001, period="total", label="tiny")])
    capsys.readouterr()
    code = cli.main(["--home", home, "doctor"])
    out = capsys.readouterr().out
    assert code == 1 and "exceeded" in out


def test_new_plugins_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("import", "compare", "anomaly", "tags", "doctor"):
        assert registry.get(name) is not None
