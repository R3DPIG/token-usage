"""CLI tests for the iteration-6 plan/quota plugins.

The CLI reads the real clock, so fixtures must use timestamps relative to *now*
rather than fixed dates -- otherwise the period filter drops them.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from token_usage import cli
from token_usage.models import Budget, UsageRecord
from token_usage.storage import JsonStore


def _today_iso(hour: int = 12) -> str:
    """A timestamp inside the current month, so period filters keep it."""
    now = datetime.now(timezone.utc)
    return now.replace(hour=hour, minute=0, second=0, microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%S")


def rec(model="gpt-4o", inp=1_000_000, out=0, ts=None, project="demo"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts or _today_iso(), project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def seed(store, records):
    store.append_records(records)


# -- plan -------------------------------------------------------------------

def test_cli_plan_no_records(store, capsys):
    assert cli.main(["--home", str(store.home), "plan"]) == 0
    assert "nothing to plan" in capsys.readouterr().out


def test_cli_plan_with_limit(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "plan", "--limit", "100"]) == 0
    out = capsys.readouterr().out
    assert "projected total" in out and "cli --limit" in out


def test_cli_plan_rejects_bad_period(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "plan", "--period", "yearly"]) == 2


def test_cli_plan_uses_stored_budgets(store, capsys):
    seed(store, [rec()])
    store.save_budgets([Budget(limit_usd=100.0, period="monthly", label="main")])
    assert cli.main(["--home", str(store.home), "plan"]) == 0
    assert "main" in capsys.readouterr().out


def test_cli_plan_exceeded_exits_nonzero(store, capsys):
    seed(store, [rec(inp=100_000_000)])
    store.save_budgets([Budget(limit_usd=0.001, period="monthly", label="tiny")])
    code = cli.main(["--home", str(store.home), "plan"])
    assert code == 1
    assert "[OVER]" in capsys.readouterr().out


def test_cli_plan_no_budgets_hint(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "plan"]) == 0
    assert "no budgets configured" in capsys.readouterr().out


def test_cli_plan_what_if(store, capsys):
    seed(store, [rec(model="gpt-4o", inp=10_000_000)])
    assert cli.main(["--home", str(store.home), "plan",
                     "--what-if", "gpt-4o", "--change", "-30"]) == 0
    out = capsys.readouterr().out
    assert "what-if" in out and "adjusted total" in out


def test_cli_plan_what_if_unknown_model(store, capsys):
    seed(store, [rec(model="gpt-4o")])
    assert cli.main(["--home", str(store.home), "plan",
                     "--what-if", "not-used"]) == 0
    assert "no spend recorded" in capsys.readouterr().out


# -- quota ------------------------------------------------------------------

def test_cli_quota_requires_total(store, capsys):
    seed(store, [rec()])
    code = cli.main(["--home", str(store.home), "quota"])
    assert code == 2 and "--total" in capsys.readouterr().out


def test_cli_quota_splits_by_project(store, capsys):
    seed(store, [rec(project="a", inp=3_000_000), rec(project="b", inp=1_000_000)])
    assert cli.main(["--home", str(store.home), "quota", "--total", "100"]) == 0
    out = capsys.readouterr().out
    assert "allocating" in out and "allocated" in out


def test_cli_quota_by_model(store, capsys):
    seed(store, [rec(model="gpt-4o"), rec(model="gpt-4o-mini")])
    assert cli.main(["--home", str(store.home), "quota", "--total", "50",
                     "--dimension", "model"]) == 0
    out = capsys.readouterr().out
    assert "gpt-4o" in out


def test_cli_quota_rejects_bad_dimension(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "quota", "--total", "10",
                     "--dimension", "hour"]) == 2


def test_cli_quota_no_records(store, capsys):
    assert cli.main(["--home", str(store.home), "quota", "--total", "10"]) == 0
    assert "no records in scope" in capsys.readouterr().out


def test_cli_quota_floor(store, capsys):
    seed(store, [rec(project="a", inp=9_000_000), rec(project="b", inp=1_000_000)])
    assert cli.main(["--home", str(store.home), "quota", "--total", "100",
                     "--floor", "20"]) == 0
    assert "floor of" in capsys.readouterr().out


def test_cli_quota_warns_on_concentration(store, capsys):
    seed(store, [rec(project="a", inp=9_000_000), rec(project="b", inp=1_000)])
    assert cli.main(["--home", str(store.home), "quota", "--total", "100"]) == 0
    assert "takes" in capsys.readouterr().out


# -- regression -------------------------------------------------------------

def test_existing_commands_still_work(store, capsys):
    seed(store, [rec()])
    for command in ("report", "stats", "forecast", "doctor"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1)


def test_new_plugins_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("plan", "quota"):
        assert registry.get(name) is not None
