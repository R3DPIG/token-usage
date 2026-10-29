"""CLI tests for the iteration-5 alert plugin."""

from __future__ import annotations

import json

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore


def rec(model="gpt-4o", inp=1_000_000, out=0, ts="2024-05-01T12:00:00", project="demo"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


# -- rule management --------------------------------------------------------

def test_cli_alert_init_and_list(store, capsys):
    home = str(store.home)
    assert cli.main(["--home", home, "alert", "--action", "init"]) == 0
    assert "installed 3" in capsys.readouterr().out

    assert cli.main(["--home", home, "alert", "--action", "list"]) == 0
    out = capsys.readouterr().out
    assert "monthly-spend" in out and "low-cache-hit" in out


def test_cli_alert_init_refuses_without_force(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "alert", "--action", "init"])
    capsys.readouterr()
    assert cli.main(["--home", home, "alert", "--action", "init"]) == 1
    assert cli.main(["--home", home, "alert", "--action", "init", "--force"]) == 0


def test_cli_alert_add_and_remove(store, capsys):
    home = str(store.home)
    assert cli.main(["--home", home, "alert", "--action", "add",
                     "--name", "big-day", "--metric", "cost", "--operator", "gt",
                     "--threshold", "5", "--dimension", "day"]) == 0
    assert "saved rule" in capsys.readouterr().out
    assert len(store.load_alerts()) == 1

    assert cli.main(["--home", home, "alert", "--action", "remove",
                     "--name", "big-day"]) == 0
    assert store.load_alerts() == []


def test_cli_alert_add_requires_fields(store, capsys):
    code = cli.main(["--home", str(store.home), "alert", "--action", "add",
                     "--name", "x"])
    assert code == 2 and "requires" in capsys.readouterr().out


def test_cli_alert_add_rejects_bad_metric(store, capsys):
    code = cli.main(["--home", str(store.home), "alert", "--action", "add",
                     "--name", "x", "--metric", "nonsense", "--operator", "gt",
                     "--threshold", "1"])
    assert code == 2


def test_cli_alert_remove_missing(store, capsys):
    assert cli.main(["--home", str(store.home), "alert", "--action", "remove",
                     "--name", "nope"]) == 1


def test_cli_alert_enable_disable(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "alert", "--action", "add", "--name", "r",
              "--metric", "cost", "--operator", "gt", "--threshold", "1"])
    capsys.readouterr()

    assert cli.main(["--home", home, "alert", "--action", "disable",
                     "--name", "r"]) == 0
    assert store.load_alerts()[0].enabled is False

    assert cli.main(["--home", home, "alert", "--action", "enable",
                     "--name", "r"]) == 0
    assert store.load_alerts()[0].enabled is True


def test_cli_alert_metrics(store, capsys):
    assert cli.main(["--home", str(store.home), "alert", "--action", "metrics"]) == 0
    out = capsys.readouterr().out
    assert "cache_ratio" in out and "operators" in out


# -- checking ---------------------------------------------------------------

def test_cli_alert_check_no_rules(store, capsys):
    assert cli.main(["--home", str(store.home), "alert", "--action", "check"]) == 0
    assert "nothing to check" in capsys.readouterr().out


def test_cli_alert_check_fires_and_exits_nonzero(store, capsys):
    home = str(store.home)
    store.append_records([rec(inp=10_000_000)])
    cli.main(["--home", home, "alert", "--action", "add", "--name", "spend",
              "--metric", "cost", "--operator", "gt", "--threshold", "0.001",
              "--severity", "critical"])
    capsys.readouterr()

    code = cli.main(["--home", home, "alert", "--action", "check"])
    out = capsys.readouterr().out
    assert code == 1 and "[FIRE]" in out


def test_cli_alert_check_fail_on_never(store, capsys):
    home = str(store.home)
    store.append_records([rec(inp=10_000_000)])
    cli.main(["--home", home, "alert", "--action", "add", "--name", "spend",
              "--metric", "cost", "--operator", "gt", "--threshold", "0.001",
              "--severity", "critical"])
    capsys.readouterr()

    assert cli.main(["--home", home, "alert", "--action", "check",
                     "--fail-on", "never"]) == 0


def test_cli_alert_check_info_does_not_fail_critical_gate(store, capsys):
    home = str(store.home)
    store.append_records([rec(inp=10_000_000)])
    cli.main(["--home", home, "alert", "--action", "add", "--name", "spend",
              "--metric", "cost", "--operator", "gt", "--threshold", "0.001",
              "--severity", "info"])
    capsys.readouterr()

    assert cli.main(["--home", home, "alert", "--action", "check",
                     "--fail-on", "critical"]) == 0


def test_cli_alert_check_rejects_bad_fail_on(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "alert", "--action", "init"])
    capsys.readouterr()
    assert cli.main(["--home", home, "alert", "--action", "check",
                     "--fail-on", "sometimes"]) == 2


def test_cli_alert_check_reports_unknown_for_empty_scope(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "alert", "--action", "add", "--name", "r",
              "--metric", "cost", "--operator", "gt", "--threshold", "1",
              "--project", "nope"])
    capsys.readouterr()
    assert cli.main(["--home", home, "alert", "--action", "check"]) == 0
    assert "[ ?? ]" in capsys.readouterr().out


# -- resilience -------------------------------------------------------------

def test_corrupt_alerts_file_is_tolerated(store, capsys):
    store.append_records([rec()])
    store.alerts_path.write_text("{ not json", encoding="utf-8")
    assert cli.main(["--home", str(store.home), "alert", "--action", "list"]) == 0
    assert "no alert rules" in capsys.readouterr().out


def test_alerts_file_with_one_bad_rule_keeps_the_rest(store, capsys):
    store.alerts_path.write_text(json.dumps([
        {"name": "good", "metric": "cost", "operator": "gt", "threshold": 1},
        {"name": "broken", "metric": "nonsense", "operator": "gt", "threshold": 1},
    ]), encoding="utf-8")
    assert cli.main(["--home", str(store.home), "alert", "--action", "list"]) == 0
    out = capsys.readouterr().out
    assert "good" in out
    assert "broken" not in out          # the malformed rule was dropped
    assert len(store.load_alerts()) == 1


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    assert registry.get("alert") is not None
