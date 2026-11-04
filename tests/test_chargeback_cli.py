"""CLI tests for the iteration-9 chargeback plugin."""

from __future__ import annotations

import pytest

from token_usage import cli
from token_usage.attribution import OwnerRule
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore


def rec(model="gpt-4o", inp=1_000_000, out=0, project="api",
        ts="2024-05-06T12:00:00"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def run(store, *args):
    return cli.main(["--home", str(store.home), "chargeback", *args])


# -- summary ----------------------------------------------------------------

def test_cli_chargeback_no_records(store, capsys):
    assert run(store) == 0
    assert "nothing to attribute" in capsys.readouterr().out


def test_cli_chargeback_summary_without_rules(store, capsys):
    store.append_records([rec()])
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "0.0% coverage" in out and "no rules defined" in out


def test_cli_chargeback_summary_with_rules(store, capsys):
    store.append_records([rec(project="api")])
    store.save_owners([OwnerRule(owner="team-api", project="api")])
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "100.0% coverage" in out
    assert "team-api" in out and "balances" in out


# -- views ------------------------------------------------------------------

@pytest.mark.parametrize("view,expected", [
    ("owners", "spend by owner"),
    ("orphans", "records no rule claimed"),
    ("suggest", "proposed rules"),
    ("reconcile", "attribution reconciliation"),
])
def test_cli_chargeback_views(store, capsys, view, expected):
    store.append_records([rec(project="api")])
    assert run(store, "--view", view) == 0
    assert expected in capsys.readouterr().out


def test_cli_chargeback_orphans_lists_records(store, capsys):
    store.append_records([rec(project="web")])
    assert run(store, "--view", "orphans") == 0
    out = capsys.readouterr().out
    assert "web" in out and "gpt-4o" in out


def test_cli_chargeback_suggest_prints_command(store, capsys):
    store.append_records([rec(project="web")])
    assert run(store, "--view", "suggest") == 0
    out = capsys.readouterr().out
    assert "--action add" in out and "team-web" in out


def test_cli_chargeback_invoice(store, capsys):
    store.append_records([rec(project="api", model="gpt-4o"),
                          rec(project="api", model="gpt-4o-mini")])
    store.save_owners([OwnerRule(owner="team-api", project="api")])
    assert run(store, "--view", "invoice", "--owner", "team-api") == 0
    out = capsys.readouterr().out
    assert "invoice for team-api" in out
    assert "by project" in out and "by model" in out


def test_cli_chargeback_invoice_unknown_owner(store, capsys):
    store.append_records([rec()])
    assert run(store, "--view", "invoice", "--owner", "ghost") == 0
    assert "no rule assigns" in capsys.readouterr().out


def test_cli_chargeback_reconcile_balanced(store, capsys):
    store.append_records([rec(project="api")])
    store.save_owners([OwnerRule(owner="a", project="api")])
    assert run(store, "--view", "reconcile") == 0
    assert "balanced" in capsys.readouterr().out


# -- rule management --------------------------------------------------------

def test_cli_chargeback_rules_empty(store, capsys):
    assert run(store, "--view", "rules") == 0
    assert "no rules defined" in capsys.readouterr().out


def test_cli_chargeback_rules_add_and_list(store, capsys):
    store.append_records([rec(project="api")])
    assert run(store, "--view", "rules", "--action", "add",
               "--owner", "team-api", "--project", "api") == 0
    out = capsys.readouterr().out
    assert "added rule" in out and "claims 1 record" in out

    assert run(store, "--view", "rules") == 0
    assert "team-api" in capsys.readouterr().out
    assert len(store.load_owners()) == 1


def test_cli_chargeback_rules_add_warns_when_matching_nothing(store, capsys):
    store.append_records([rec(project="api")])
    assert run(store, "--view", "rules", "--action", "add",
               "--owner", "ghost", "--project", "nope") == 0
    assert "matches no stored record" in capsys.readouterr().out


def test_cli_chargeback_rules_remove(store, capsys):
    store.save_owners([OwnerRule(owner="a", project="api"),
                       OwnerRule(owner="b", project="web")])
    assert run(store, "--view", "rules", "--action", "remove",
               "--index", "0") == 0
    assert "removed rule" in capsys.readouterr().out
    assert [r.owner for r in store.load_owners()] == ["b"]


# -- validation -------------------------------------------------------------

@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--view", "rules", "--action", "nonsense"),
    ("--view", "rules", "--action", "add"),
    ("--view", "rules", "--action", "remove", "--index", "5"),
    ("--view", "invoice"),
])
def test_cli_chargeback_rejects_bad_args(store, capsys, args):
    store.append_records([rec()])
    assert run(store, *args) == 2


# -- filtering and regression -----------------------------------------------

def test_cli_chargeback_project_filter(store, capsys):
    store.append_records([rec(project="api"), rec(project="web")])
    assert run(store, "--view", "orphans", "--project", "web") == 0
    assert "1 call(s)" in capsys.readouterr().out


def test_existing_commands_still_work(store, capsys):
    store.append_records([rec()])
    for command in ("report", "stats", "pattern", "optimize", "plan", "doctor"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    assert registry.get("chargeback") is not None
    assert registry.get("owners") is not None      # alias
