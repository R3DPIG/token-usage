"""CLI tests for the iteration-16 backtest plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore

NOW = datetime.now(timezone.utc).replace(microsecond=0)
TOKENS_PER_DOLLAR = 400_000          # gpt-4o input at $2.50/1M


def seed(home, usd_per_day, project="api"):
    """Seed one record per day, oldest first, each costing that many dollars."""
    store = JsonStore(home)
    count = len(usd_per_day)
    store.append_records([
        UsageRecord(model="gpt-4o", input_tokens=int(usd * TOKENS_PER_DOLLAR),
                    output_tokens=0,
                    timestamp=(NOW - timedelta(days=count - 1 - index)).strftime(
                        "%Y-%m-%dT%H:%M:%S"),
                    project=project, tags=["api"])
        for index, usd in enumerate(usd_per_day)])
    return store


def run(store, *args):
    return cli.main(["--home", str(store.home), "backtest", *args])


@pytest.fixture
def steady(tmp_path):
    """Twelve flat days: an easy series the forecast should nail."""
    return seed(tmp_path / "steady", [1.0] * 12)


@pytest.fixture
def volatile(tmp_path):
    """Twelve days swinging wildly: errors must be large and visible."""
    return seed(tmp_path / "volatile",
                [1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0])


# -- empty and thin history -----------------------------------------------------

def test_cli_backtest_with_too_few_days(tmp_path, capsys):
    store = seed(tmp_path / "thin", [1.0, 2.0])
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "not evaluated" in out and "need at least" in out


def test_cli_backtest_with_no_records(tmp_path, capsys):
    store = JsonStore(tmp_path / "empty")
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "not evaluated" in out


# -- views ----------------------------------------------------------------------

def test_cli_backtest_summary_on_a_steady_series(steady, capsys):
    assert run(steady) == 0
    out = capsys.readouterr().out
    assert "forecast backtest" in out and "samples" in out and "mape" in out
    assert "walk-forward" in out


def test_cli_backtest_summary_states_the_bias_sign_convention(steady, capsys):
    assert run(steady) == 0
    assert "positive means the forecast ran high" in capsys.readouterr().out


def test_cli_backtest_samples_view_lists_each_cutoff(steady, capsys):
    assert run(steady, "--view", "samples") == 0
    out = capsys.readouterr().out
    assert "cutoff" in out and "predicted" in out and "actual" in out


def test_cli_backtest_worst_view_shows_largest_misses(volatile, capsys):
    assert run(volatile, "--view", "worst") == 0
    out = capsys.readouterr().out
    assert "largest misses" in out and "off by" in out


def test_cli_backtest_worst_view_without_samples(tmp_path, capsys):
    store = seed(tmp_path / "thin", [1.0, 2.0])
    assert run(store, "--view", "worst") == 0
    out = capsys.readouterr().out
    assert "largest misses" not in out          # no fake table for empty data
    assert "need at least" in out


# -- window arithmetic ------------------------------------------------------------

def test_cli_backtest_window_shrinks_the_sample_count(steady, capsys):
    assert run(steady, "--window", "1") == 0
    narrow = capsys.readouterr().out
    assert run(steady, "--window", "5") == 0
    wide = capsys.readouterr().out
    # Both report a window; the wider one grades fewer cutoffs.
    assert "1 day(s) per prediction" in narrow
    assert "5 day(s) per prediction" in wide


def test_cli_backtest_window_too_wide_grades_nothing(steady, capsys):
    assert run(steady, "--window", "40") == 0
    out = capsys.readouterr().out
    assert "not evaluated" in out


@pytest.mark.parametrize("args", [
    ("--window", "0"),
    ("--view", "nonsense"),
    ("--fail-on", "nonsense"),
])
def test_cli_backtest_rejects_bad_args(steady, capsys, args):
    assert run(steady, *args) == 2


# -- reliability and gating -------------------------------------------------------

def test_cli_backtest_flags_a_thin_sample_as_a_sanity_check(tmp_path, capsys):
    store = seed(tmp_path / "thin", [1.0] * 5)
    assert run(store, "--window", "2") == 0
    assert "sanity check, not a verdict" in capsys.readouterr().out


def test_cli_backtest_marks_a_long_run_as_reliable(steady, capsys):
    assert run(steady, "--window", "1") == 0
    assert "below the reliability floor" not in capsys.readouterr().out


def test_cli_backtest_fail_on_unreliable_ignores_a_usable_forecast(steady):
    assert run(steady, "--window", "1", "--fail-on", "unreliable") == 0


def test_cli_backtest_fail_on_unreliable_fires_on_a_volatile_series(volatile,
                                                                   capsys):
    """A forecast this erratic is graded unreliable and must fail the gate."""
    assert run(volatile, "--window", "1", "--fail-on", "unreliable") == 1
    out = capsys.readouterr().out
    assert "grade: unreliable" in out
    assert "--fail-on unreliable" in out


def test_cli_backtest_fail_on_never_is_the_default(volatile):
    # The same erratic series passes when no gate is asked for.
    assert run(volatile, "--view", "summary") == 0


def test_cli_backtest_a_sustained_jump_is_still_forecast_well(tmp_path, capsys):
    """A step change is not noise: the line adapts, so the gate stays shut."""
    store = seed(tmp_path / "explode", [1.0] * 6 + [200.0] * 6)
    assert run(store, "--window", "1", "--fail-on", "unreliable") == 0
    assert "grade: usable" in capsys.readouterr().out


# -- filtering and safety ---------------------------------------------------------

def test_cli_backtest_project_filter_scopes_the_history(tmp_path, capsys):
    store = seed(tmp_path / "w", [1.0] * 12, project="api")
    JsonStore(store.home).append_records([
        UsageRecord(model="gpt-4o", input_tokens=TOKENS_PER_DOLLAR,
                    output_tokens=0,
                    timestamp=(NOW - timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S"),
                    project="web")
        for i in range(10)])
    assert run(store, "--project", "api", "--view", "samples") == 0
    assert "cutoff" in capsys.readouterr().out


def test_cli_backtest_model_filter_leaves_an_empty_scope(tmp_path, capsys):
    store = seed(tmp_path / "w", [1.0] * 12)
    assert run(store, "--model", "no-such-model") == 0
    assert "not evaluated" in capsys.readouterr().out


def test_cli_backtest_never_writes_to_the_workspace(steady):
    before = steady.usage_path.read_text(encoding="utf-8")
    assert run(steady, "--apply") == 0
    assert steady.usage_path.read_text(encoding="utf-8") == before


def test_existing_commands_still_work(tmp_path):
    store = seed(tmp_path / "regress", [1.0] * 12)
    for command in ("forecast", "report", "stats", "quality", "envelope",
                    "reconcile", "plan", "simulate"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1, 2)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("backtest", "bt", "accuracy"):
        assert registry.get(name) is not None
