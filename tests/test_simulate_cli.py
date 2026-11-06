"""CLI tests for the iteration-11 simulate plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore


def _ts(days_ago: int) -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0)
            - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def rec(inp=1_000_000, out=0, days_ago=1, project="default"):
    return UsageRecord(model="gpt-4o", input_tokens=inp, output_tokens=out,
                       timestamp=_ts(days_ago), project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def run(store, *args):
    return cli.main(["--home", str(store.home), "simulate", *args])


@pytest.fixture
def seeded(store):
    """Three days of recent usage inside the current daily window."""
    store.append_records([rec(days_ago=d) for d in range(0, 3)])
    return store


# -- summary ----------------------------------------------------------------

def test_cli_simulate_no_records(store, capsys):
    assert run(store) == 0
    assert "nothing to simulate" in capsys.readouterr().out


def test_cli_simulate_summary(seeded, capsys):
    assert run(seeded) == 0
    out = capsys.readouterr().out
    assert "period-end risk" in out
    assert "spent so far" in out and "days left" in out
    assert "P90 end" in out


def test_cli_simulate_summary_hints_limit(seeded, capsys):
    assert run(seeded) == 0
    assert "pass --limit" in capsys.readouterr().out


def test_cli_simulate_summary_with_limit(seeded, capsys):
    assert run(seeded, "--limit", "50") == 0
    out = capsys.readouterr().out
    assert "overspend risk" in out and "limit" in out


# -- views ------------------------------------------------------------------

@pytest.mark.parametrize("view,expected", [
    ("distribution", "simulated period-end distribution"),
    ("methods", "bootstrap vs normal"),
])
def test_cli_simulate_views(seeded, capsys, view, expected):
    assert run(seeded, "--view", view) == 0
    assert expected in capsys.readouterr().out


def test_cli_simulate_distribution_lists_percentiles(seeded, capsys):
    assert run(seeded, "--view", "distribution", "--limit", "1000") == 0
    out = capsys.readouterr().out
    assert "P50" in out and "P90" in out and "P95" in out
    assert "<- limit" in out


def test_cli_simulate_methods_reports_agreement(seeded, capsys):
    assert run(seeded, "--view", "methods") == 0
    out = capsys.readouterr().out
    assert "P90 spread" in out
    assert "close" in out or "divergent" in out


def test_cli_simulate_budget_requires_limit(seeded, capsys):
    assert run(seeded, "--view", "budget") == 2


def test_cli_simulate_budget_view(seeded, capsys):
    assert run(seeded, "--view", "budget", "--limit", "1000") == 0
    out = capsys.readouterr().out
    assert "overspend risk" in out
    assert "most simulated futures" in out


def test_cli_simulate_budget_flags_likely_breach(capsys, tmp_path):
    store = JsonStore(tmp_path / "breach")
    store.append_records([rec(inp=1_000_000) for _ in range(5)])
    assert run(store, "--view", "budget", "--limit", "0.01") == 0
    assert "expected outcome" in capsys.readouterr().out


def test_cli_simulate_budget_reports_required_cut(capsys, tmp_path):
    store = JsonStore(tmp_path / "cut")
    store.append_records([rec(inp=1_000_000) for _ in range(5)])
    assert run(store, "--view", "budget", "--limit", "1") == 0
    assert "cut" in capsys.readouterr().out


# -- options ----------------------------------------------------------------

def test_cli_simulate_is_reproducible(seeded, capsys):
    assert run(seeded, "--seed", "99") == 0
    first = capsys.readouterr().out
    assert run(seeded, "--seed", "99") == 0
    assert capsys.readouterr().out == first


@pytest.mark.parametrize("period", ["daily", "weekly", "monthly"])
def test_cli_simulate_periods(seeded, capsys, period):
    assert run(seeded, "--period", period) == 0
    assert period in capsys.readouterr().out


def test_cli_simulate_method_option(seeded, capsys):
    assert run(seeded, "--method", "normal", "--limit", "1000") == 0
    assert "normal" in capsys.readouterr().out


def test_cli_simulate_iterations_option(seeded, capsys):
    assert run(seeded, "--iterations", "100") == 0
    assert "100" in capsys.readouterr().out


# -- validation -------------------------------------------------------------

@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--period", "yearly"),
    ("--method", "nonsense"),
    ("--iterations", "0"),
])
def test_cli_simulate_rejects_bad_args(seeded, capsys, args):
    assert run(seeded, *args) == 2


# -- filtering and regression -----------------------------------------------

def test_cli_simulate_project_filter(store, capsys):
    store.append_records([rec(days_ago=1, project="api"),
                          rec(days_ago=1, project="web")])
    assert run(store, "--project", "api") == 0
    assert "period-end risk" in capsys.readouterr().out


def test_existing_commands_still_work(store, capsys):
    store.append_records([rec()])
    for command in ("report", "stats", "pattern", "optimize", "chargeback",
                    "drivers", "plan", "doctor"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    assert registry.get("simulate") is not None
    assert registry.get("risk") is not None           # alias
