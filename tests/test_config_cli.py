"""CLI tests for the iteration-4 config/preset plugins.

The important behaviours: defaults must not override explicit flags, presets
must sit between defaults and explicit flags, and a broken config must never
break a command.
"""

from __future__ import annotations

import json

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore


def rec(project="demo", model="gpt-4o", inp=1000, out=500, ts="2024-05-01T12:00:00"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


# -- config plugin ----------------------------------------------------------

def test_cli_config_set_list_unset(store, capsys):
    home = str(store.home)
    assert cli.main(["--home", home, "config", "--action", "set",
                     "--key", "project", "--value", "demo"]) == 0
    assert "project = 'demo'" in capsys.readouterr().out

    assert cli.main(["--home", home, "config", "--action", "list"]) == 0
    assert "project" in capsys.readouterr().out

    assert cli.main(["--home", home, "config", "--action", "unset",
                     "--key", "project"]) == 0
    assert store.load_config().defaults == {}


def test_cli_config_rejects_unknown_key(store, capsys):
    code = cli.main(["--home", str(store.home), "config", "--action", "set",
                     "--key", "projct", "--value", "demo"])
    assert code == 2 and "unknown setting" in capsys.readouterr().out


def test_cli_config_rejects_bad_enum(store, capsys):
    code = cli.main(["--home", str(store.home), "config", "--action", "set",
                     "--key", "group", "--value", "hour"])
    assert code == 2


def test_cli_config_clear(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "config", "--action", "set", "--key", "project", "--value", "a"])
    cli.main(["--home", home, "config", "--action", "set", "--key", "limit", "--value", "5"])
    capsys.readouterr()
    assert cli.main(["--home", home, "config", "--action", "clear"]) == 0
    assert "cleared 2" in capsys.readouterr().out


def test_cli_config_keys(store, capsys):
    assert cli.main(["--home", str(store.home), "config", "--action", "keys"]) == 0
    assert "project" in capsys.readouterr().out


# -- defaults actually apply ------------------------------------------------

def test_default_project_applies_to_list(store, capsys):
    home = str(store.home)
    store.append_records([rec(project="demo"), rec(project="other")])
    cli.main(["--home", home, "config", "--action", "set", "--key", "project", "--value", "demo"])
    capsys.readouterr()

    assert cli.main(["--home", home, "list"]) == 0
    out = capsys.readouterr().out
    assert "showing 1 of 1" in out  # only the demo record


def test_explicit_flag_overrides_default(store, capsys):
    home = str(store.home)
    store.append_records([rec(project="demo"), rec(project="other")])
    cli.main(["--home", home, "config", "--action", "set", "--key", "project", "--value", "demo"])
    capsys.readouterr()

    assert cli.main(["--home", home, "list", "--project", "other"]) == 0
    assert "showing 1 of 1" in capsys.readouterr().out


def test_default_group_applies_to_report(store, capsys):
    home = str(store.home)
    store.append_records([rec()])
    cli.main(["--home", home, "config", "--action", "set", "--key", "group", "--value", "day"])
    capsys.readouterr()
    assert cli.main(["--home", home, "report"]) == 0
    assert "grouped by day" in capsys.readouterr().out


# -- presets ----------------------------------------------------------------

def test_cli_preset_save_list_show_delete(store, capsys):
    home = str(store.home)
    assert cli.main(["--home", home, "preset", "--action", "save",
                     "--name", "weekly", "--group", "day", "--period", "weekly"]) == 0
    assert "saved preset" in capsys.readouterr().out

    assert cli.main(["--home", home, "preset", "--action", "list"]) == 0
    assert "weekly" in capsys.readouterr().out

    assert cli.main(["--home", home, "preset", "--action", "show", "--name", "weekly"]) == 0
    assert "--preset weekly" in capsys.readouterr().out

    assert cli.main(["--home", home, "preset", "--action", "delete",
                     "--name", "weekly"]) == 0
    assert store.load_config().preset_names() == []


def test_cli_preset_save_requires_options(store, capsys):
    code = cli.main(["--home", str(store.home), "preset", "--action", "save",
                     "--name", "empty"])
    assert code == 2 and "no recognised options" in capsys.readouterr().out


def test_cli_preset_replay(store, capsys):
    home = str(store.home)
    store.append_records([rec()])
    cli.main(["--home", home, "preset", "--action", "save",
              "--name", "daily", "--group", "day"])
    capsys.readouterr()

    assert cli.main(["--home", home, "report", "--preset", "daily"]) == 0
    assert "grouped by day" in capsys.readouterr().out


def test_cli_preset_explicit_flag_beats_preset(store, capsys):
    home = str(store.home)
    store.append_records([rec()])
    cli.main(["--home", home, "preset", "--action", "save",
              "--name", "daily", "--group", "day"])
    capsys.readouterr()

    assert cli.main(["--home", home, "report", "--preset", "daily",
                     "--group", "model"]) == 0
    assert "grouped by model" in capsys.readouterr().out


def test_cli_missing_preset_warns_but_runs(store, capsys):
    home = str(store.home)
    store.append_records([rec()])
    assert cli.main(["--home", home, "report", "--preset", "nope"]) == 0
    captured = capsys.readouterr()
    assert "not found" in captured.err
    assert "usage summary" in captured.out


def test_cli_preset_delete_missing(store, capsys):
    assert cli.main(["--home", str(store.home), "preset", "--action", "delete",
                     "--name", "nope"]) == 1


# -- resilience -------------------------------------------------------------

def test_corrupt_config_does_not_break_commands(store, capsys):
    store.append_records([rec()])
    store.config_path.write_text("{ not valid json", encoding="utf-8")
    assert cli.main(["--home", str(store.home), "list"]) == 0
    assert "showing 1 of 1" in capsys.readouterr().out


def test_config_with_invalid_default_is_ignored(store, capsys):
    store.append_records([rec()])
    store.config_path.write_text(
        json.dumps({"defaults": {"group": "hour", "project": "demo"}}), encoding="utf-8")
    assert cli.main(["--home", str(store.home), "list"]) == 0
    assert "showing 1 of 1" in capsys.readouterr().out


def test_new_plugins_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("config", "preset"):
        assert registry.get(name) is not None
