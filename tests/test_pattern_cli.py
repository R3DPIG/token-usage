"""CLI tests for the iteration-7 pattern plugin."""

from __future__ import annotations

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore


def rec(inp=1_000_000, out=0, ts="2024-05-06T12:00:00", model="gpt-4o",
        project="demo"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def seed_week(store):
    """One record per day for a full week, Monday 2024-05-06 onward."""
    store.append_records([rec(ts=f"2024-05-{day:02d}T12:00:00")
                          for day in range(6, 13)])


def run(store, *args):
    return cli.main(["--home", str(store.home), "pattern", *args])


# -- views ------------------------------------------------------------------

def test_cli_pattern_no_records(store, capsys):
    assert run(store) == 0
    assert "nothing to analyse" in capsys.readouterr().out


def test_cli_pattern_summary(store, capsys):
    seed_week(store)
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "temporal pattern" in out
    assert "trend" in out and "variability" in out and "seasonality" in out


@pytest.mark.parametrize("view,expected", [
    ("weekday", ["cost by weekday", "Mon", "Sun"]),
    ("hourly", ["cost by hour (UTC)", "12:00"]),
    ("trend", ["moving average", "direction"]),
    ("weekend", ["weekday vs weekend", "weekdays", "weekends"]),
    ("streaks", ["longest above"]),
    ("burst", ["classification", "cv"]),
])
def test_cli_pattern_views(store, capsys, view, expected):
    seed_week(store)
    assert run(store, "--view", view) == 0
    out = capsys.readouterr().out
    for fragment in expected:
        assert fragment in out


def test_cli_pattern_trend_window(store, capsys):
    seed_week(store)
    assert run(store, "--view", "trend", "--window", "3") == 0
    assert "moving average" in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--view", "trend", "--window", "0"),
])
def test_cli_pattern_rejects_bad_args(store, capsys, args):
    seed_week(store)
    assert run(store, *args) == 2


# -- filtering --------------------------------------------------------------

@pytest.mark.parametrize("flag,value", [
    ("--project", "a"),
    ("--model", "gpt-4o"),
])
def test_cli_pattern_filters(store, capsys, flag, value):
    store.append_records([rec(project="a", model="gpt-4o"),
                          rec(project="b", model="gpt-4o-mini")])
    assert run(store, flag, value) == 0
    assert "1 day(s)" in capsys.readouterr().out


def test_cli_pattern_since_filter(store, capsys):
    store.append_records([rec(ts="2024-05-01T12:00:00"),
                          rec(ts="2024-05-10T12:00:00")])
    assert run(store, "--since", "2024-05-05") == 0
    assert "1 day(s)" in capsys.readouterr().out


# -- behaviour --------------------------------------------------------------

def test_cli_pattern_weekend_warns_on_one_sided_data(store, capsys):
    store.append_records([rec(ts="2024-05-06T12:00:00")])  # Monday only
    assert run(store, "--view", "weekend") == 0
    assert "only one side" in capsys.readouterr().out


def test_cli_pattern_trend_reports_fit(store, capsys):
    # Alternating high/low produces a steep but poorly-fitting slope.
    store.append_records([rec(inp=1_000_000, ts="2024-05-06T12:00:00"),
                          rec(inp=9_000_000, ts="2024-05-07T12:00:00"),
                          rec(inp=1_000_000, ts="2024-05-08T12:00:00"),
                          rec(inp=9_000_000, ts="2024-05-09T12:00:00"),
                          rec(inp=1_000_000, ts="2024-05-10T12:00:00")])
    assert run(store, "--view", "trend") == 0
    assert "r2=" in capsys.readouterr().out


def test_cli_pattern_streaks_warns_on_long_run(store, capsys):
    # Six expensive days after one cheap day.
    records = [rec(inp=100_000, ts="2024-05-06T12:00:00")]
    records += [rec(inp=5_000_000, ts=f"2024-05-{d:02d}T12:00:00")
                for d in range(7, 13)]
    store.append_records(records)
    assert run(store, "--view", "streaks") == 0
    assert "consecutive expensive days" in capsys.readouterr().out


def test_cli_pattern_burst_mentions_anomaly(store, capsys):
    store.append_records([rec(inp=100_000, ts="2024-05-06T12:00:00"),
                          rec(inp=100_000, ts="2024-05-07T12:00:00"),
                          rec(inp=100_000, ts="2024-05-08T12:00:00"),
                          rec(inp=50_000_000, ts="2024-05-09T12:00:00")])
    assert run(store, "--view", "burst") == 0
    assert "anomaly" in capsys.readouterr().out


# -- regression -------------------------------------------------------------

def test_existing_commands_still_work(store, capsys):
    seed_week(store)
    for command in ("report", "stats", "anomaly", "plan"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    assert registry.get("pattern") is not None
    assert registry.get("when") is not None   # alias
