"""CLI tests for the iteration-3 lifecycle plugins.

Focus: destructive commands must be dry-run by default, and archives must
round-trip without data loss.
"""

from __future__ import annotations

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore


def rec(model="gpt-4o", inp=1000, out=500, ts="2024-05-01T12:00:00", project="demo"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def seed(store, records):
    store.append_records(records)


# -- dedupe -----------------------------------------------------------------

def test_cli_dedupe_dry_run_writes_nothing(store, capsys):
    seed(store, [rec(), rec()])
    code = cli.main(["--home", str(store.home), "dedupe"])
    out = capsys.readouterr().out
    assert code == 0 and "dry run" in out
    assert store.count_records() == 2  # untouched


def test_cli_dedupe_apply_removes(store, capsys):
    seed(store, [rec(), rec(), rec(ts="2024-05-02T12:00:00")])
    assert cli.main(["--home", str(store.home), "dedupe", "--apply"]) == 0
    assert store.count_records() == 2


def test_cli_dedupe_rejects_bad_keep(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "dedupe", "--keep", "middle"]) == 2


# -- prune ------------------------------------------------------------------

def test_cli_prune_requires_criteria(store, capsys):
    seed(store, [rec()])
    code = cli.main(["--home", str(store.home), "prune"])
    assert code == 2 and "refusing" in capsys.readouterr().out
    assert store.count_records() == 1


def test_cli_prune_dry_run_then_apply(store, capsys):
    seed(store, [rec(project="scratch"), rec(project="prod")])
    assert cli.main(["--home", str(store.home), "prune", "--project", "scratch"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert store.count_records() == 2

    assert cli.main(["--home", str(store.home), "prune",
                     "--project", "scratch", "--apply"]) == 0
    assert store.count_records() == 1
    assert store.load_records()[0].project == "prod"


# -- edit -------------------------------------------------------------------

def test_cli_edit_dry_run_then_apply(store, capsys):
    seed(store, [rec(project="demo")])
    assert cli.main(["--home", str(store.home), "edit",
                     "--index", "0", "--project", "prod"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert store.load_records()[0].project == "demo"

    assert cli.main(["--home", str(store.home), "edit",
                     "--index", "0", "--project", "prod", "--apply"]) == 0
    assert store.load_records()[0].project == "prod"


def test_cli_edit_rejects_bad_index(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "edit",
                     "--index", "9", "--project", "x"]) == 2


def test_cli_edit_rejects_no_changes(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "edit", "--index", "0"]) == 2


def test_cli_edit_rejects_invalid_edit(store, capsys):
    seed(store, [rec(inp=100)])
    code = cli.main(["--home", str(store.home), "edit", "--index", "0",
                     "--cached", "500", "--apply"])
    assert code == 2  # cached > input rejected by the model


# -- merge ------------------------------------------------------------------

def test_cli_merge_from_other_workspace(store, tmp_path, capsys):
    seed(store, [rec()])
    other = JsonStore(tmp_path / "other")
    other.append_records([rec(ts="2024-05-02T12:00:00"), rec(ts="2024-05-03T12:00:00")])

    assert cli.main(["--home", str(store.home), "merge",
                     "--from", str(other.home)]) == 0
    assert "dry run" in capsys.readouterr().out
    assert store.count_records() == 1

    assert cli.main(["--home", str(store.home), "merge",
                     "--from", str(other.home), "--apply"]) == 0
    assert store.count_records() == 3


def test_cli_merge_requires_source(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "merge"]) == 2


def test_cli_merge_missing_workspace(store, tmp_path, capsys):
    seed(store, [rec()])
    missing = tmp_path / "does-not-exist"
    assert cli.main(["--home", str(store.home), "merge", "--from", str(missing)]) == 2


# -- archive ----------------------------------------------------------------

def test_cli_archive_create_list_restore_roundtrip(store, capsys):
    home = str(store.home)
    seed(store, [rec(ts="2023-06-01T00:00:00"), rec(ts="2024-06-01T00:00:00")])

    # create (dry-run first)
    assert cli.main(["--home", home, "archive", "--action", "create",
                     "--label", "old", "--before", "2024-01-01"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert store.count_records() == 2

    # create for real
    assert cli.main(["--home", home, "archive", "--action", "create",
                     "--label", "old", "--before", "2024-01-01", "--apply"]) == 0
    assert store.count_records() == 1          # old record moved out
    assert len(store.read_archive("old")) == 1

    # list
    assert cli.main(["--home", home, "archive", "--action", "list"]) == 0
    assert "old" in capsys.readouterr().out

    # restore
    assert cli.main(["--home", home, "archive", "--action", "restore",
                     "--label", "old", "--apply"]) == 0
    assert store.count_records() == 2          # back to two


def test_cli_archive_create_requires_label_and_before(store, capsys):
    seed(store, [rec()])
    assert cli.main(["--home", str(store.home), "archive",
                     "--action", "create", "--label", "x"]) == 2


def test_cli_archive_drop(store, capsys):
    home = str(store.home)
    seed(store, [rec(ts="2023-06-01T00:00:00")])
    cli.main(["--home", home, "archive", "--action", "create",
              "--label", "old", "--before", "2024-01-01", "--apply"])
    capsys.readouterr()

    assert cli.main(["--home", home, "archive", "--action", "drop",
                     "--label", "old"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert store.archive_path("old").exists()

    assert cli.main(["--home", home, "archive", "--action", "drop",
                     "--label", "old", "--apply"]) == 0
    assert not store.archive_path("old").exists()


def test_cli_archive_unknown_action(store, capsys):
    assert cli.main(["--home", str(store.home), "archive", "--action", "nope"]) == 2


def test_new_plugins_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("dedupe", "prune", "edit", "merge", "archive"):
        assert registry.get(name) is not None
