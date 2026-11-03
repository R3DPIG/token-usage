"""CLI tests for the iteration-8 optimize plugin."""

from __future__ import annotations

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore


def rec(model="gpt-4o", inp=1_000_000, out=0, cached=0, ts="2024-05-06T12:00:00",
        project="demo"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, cached_input_tokens=cached, project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def run(store, *args):
    return cli.main(["--home", str(store.home), "optimize", *args])


# -- summary ----------------------------------------------------------------

def test_cli_optimize_no_records(store, capsys):
    assert run(store) == 0
    assert "nothing to optimise" in capsys.readouterr().out


def test_cli_optimize_summary(store, capsys):
    store.append_records([rec(inp=10_000_000)])
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "optimisation review" in out
    assert "current spend" in out and "identified saving" in out
    assert "[cache]" in out or "[waste]" in out
    assert "confidence" in out and "effort" in out


def test_cli_optimize_no_savings_message(store, capsys):
    # A tiny call on the cheapest model with caching already on.
    store.append_records([rec(model="gpt-4o-mini", inp=100, out=100, cached=100)])
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "no savings found" in out or "recommendations" in out


# -- views ------------------------------------------------------------------

@pytest.mark.parametrize("view,expected", [
    ("substitute", "model substitution"),
    ("cache", "prompt caching"),
    ("repeat", "repeated prompt"),
    ("waste", "almost no output"),
    ("batch", "small-call batching"),
])
def test_cli_optimize_views(store, capsys, view, expected):
    store.append_records([rec(inp=10_000_000)])
    assert run(store, "--view", view) == 0
    assert expected in capsys.readouterr().out


def test_cli_optimize_substitute_candidates(store, capsys):
    # claude-3-opus and gpt-4-turbo are both premium, so a same-tier swap exists.
    store.append_records([rec(model="claude-3-opus", inp=10_000_000)])
    assert run(store, "--view", "substitute") == 0
    assert "claude-3-opus ->" in capsys.readouterr().out

    # With a two-band drop the swap is still offered, but never to an embedding
    # model -- those bill no output and cannot stand in for a generator.
    assert run(store, "--view", "substitute", "--tier-drop", "2") == 0
    out = capsys.readouterr().out
    assert "claude-3-opus ->" in out and "text-embedding" not in out


def test_cli_optimize_substitute_hint_without_tier_drop(store, capsys):
    # gpt-4o is mid-tier with no cheaper mid-tier model in the catalog.
    store.append_records([rec(model="gpt-4o", inp=10_000_000)])
    assert run(store, "--view", "substitute") == 0
    assert "tier-drop" in capsys.readouterr().out


@pytest.mark.parametrize("view,records,expected", [
    ("cache", [rec(model="gpt-4o", inp=5_000_000)], "0 cached"),
    ("repeat", [rec(model="gpt-4o", inp=10_000) for _ in range(4)],
     "sent 4 times"),
    ("waste", [rec(inp=1_000_000, out=0)], "out-ratio"),
    ("batch", [rec(inp=100, out=50) for _ in range(5)], "no dollar figure"),
])
def test_cli_optimize_view_details(store, capsys, view, records, expected):
    store.append_records(records)
    assert run(store, "--view", view) == 0
    assert expected in capsys.readouterr().out


# -- validation -------------------------------------------------------------

@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--tier-drop", "9"),
    ("--tier-drop", "-1"),
])
def test_cli_optimize_rejects_bad_args(store, capsys, args):
    store.append_records([rec()])
    assert run(store, *args) == 2


def test_cli_optimize_min_savings_filters(store, capsys):
    store.append_records([rec(inp=1000)])
    assert run(store, "--view", "substitute", "--min-savings", "1000") == 0
    assert "no cheaper model" in capsys.readouterr().out


# -- filtering --------------------------------------------------------------

@pytest.mark.parametrize("args,records", [
    (("--project", "b"), [rec(project="a", inp=10_000_000),
                          rec(project="b", inp=100)]),
    (("--since", "2024-05-05"), [rec(ts="2024-05-01T12:00:00"),
                                 rec(ts="2024-05-10T12:00:00")]),
])
def test_cli_optimize_filters(store, capsys, args, records):
    store.append_records(records)
    assert run(store, *args) == 0
    assert "1 record(s)" in capsys.readouterr().out


# -- regression -------------------------------------------------------------

def test_existing_commands_still_work(store, capsys):
    store.append_records([rec()])
    for command in ("report", "stats", "pattern", "plan", "doctor"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    assert registry.get("optimize") is not None
    assert registry.get("savings") is not None   # alias
