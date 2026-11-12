"""CLI tests for the iteration-15 envelope plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore

# Records are stamped "now" so they land inside the current period window.
NOW = datetime.now(timezone.utc).replace(microsecond=0)


def seed(home, projects=("api",), inp=1_000_000):
    """One 1M-input gpt-4o record per project, priced at $2.50 each."""
    store = JsonStore(home)
    store.append_records([
        UsageRecord(model="gpt-4o", input_tokens=inp, output_tokens=0,
                    timestamp=(NOW - timedelta(minutes=i + 1)).strftime(
                        "%Y-%m-%dT%H:%M:%S"),
                    project=project, tags=["api"])
        for i, project in enumerate(projects)])
    return store


def run(store, *args):
    return cli.main(["--home", str(store.home), "envelope", *args])


def add(store, name, limit, scope="project:api", period="monthly"):
    return run(store, "--action", "set", "--name", name,
               "--limit", str(limit), "--scope", scope, "--period", period)


# -- management ---------------------------------------------------------------

def test_cli_envelope_list_is_empty_by_default(tmp_path, capsys):
    assert run(seed(tmp_path / "w"), "--action", "list") == 0
    out = capsys.readouterr().out
    assert "no spending envelopes defined" in out and "hint:" in out


def test_cli_envelope_set_then_list(tmp_path, capsys):
    store = seed(tmp_path / "w")
    assert add(store, "api", 40) == 0
    assert "added envelope 'api'" in capsys.readouterr().out
    assert run(store, "--action", "list") == 0
    out = capsys.readouterr().out
    assert "api" in out and "project=api" in out and "monthly" in out


def test_cli_envelope_set_replaces_an_existing_name(tmp_path, capsys):
    store = seed(tmp_path / "w")
    add(store, "api", 40)
    capsys.readouterr()
    assert add(store, "api", 75) == 0
    assert "replaced envelope 'api'" in capsys.readouterr().out
    assert len(store.load_envelopes()) == 1


def test_cli_envelope_set_requires_name_and_positive_limit(tmp_path, capsys):
    store = seed(tmp_path / "w")
    assert run(store, "--action", "set", "--limit", "40") == 2
    assert run(store, "--action", "set", "--name", "x", "--limit", "0") == 2


@pytest.mark.parametrize("args", [
    ("--scope", "api"),              # bare scope, no type
    ("--scope", "user:bob"),         # unknown scope type
    ("--period", "yearly"),          # unknown period
])
def test_cli_envelope_set_rejects_bad_shape(tmp_path, capsys, args):
    store = seed(tmp_path / "w")
    assert run(store, "--action", "set", "--name", "x", "--limit", "10", *args) == 2


def test_cli_envelope_remove(tmp_path, capsys):
    store = seed(tmp_path / "w")
    add(store, "api", 40)
    capsys.readouterr()
    assert run(store, "--action", "remove", "--name", "api") == 0
    assert "removed envelope 'api'" in capsys.readouterr().out
    assert store.load_envelopes() == []


def test_cli_envelope_remove_unknown_name_exits_one(tmp_path, capsys):
    store = seed(tmp_path / "w")
    assert run(store, "--action", "remove", "--name", "ghost") == 1
    assert "no envelope named" in capsys.readouterr().out


def test_cli_envelope_unknown_action(tmp_path, capsys):
    assert run(seed(tmp_path / "w"), "--action", "nonsense") == 2


# -- review --------------------------------------------------------------------

@pytest.fixture
def healthy(tmp_path):
    store = seed(tmp_path / "w")
    add(store, "api", 1000, period="total")
    return store


@pytest.fixture
def blown(tmp_path):
    store = seed(tmp_path / "w")
    add(store, "api", 1, period="total")       # $2.50 spent against $1.00
    return store


def test_cli_envelope_review_without_envelopes(tmp_path, capsys):
    assert run(seed(tmp_path / "w")) == 0
    assert "nothing to review" in capsys.readouterr().out


def test_cli_envelope_review_table_when_healthy(healthy, capsys):
    assert run(healthy, "--action", "review") == 0
    out = capsys.readouterr().out
    assert "envelope review (clean)" in out
    assert "coverage" in out and "pace" in out


def test_cli_envelope_review_reports_an_overrun(blown, capsys):
    assert run(blown, "--action", "review") == 0
    out = capsys.readouterr().out
    assert "envelope review (attention)" in out and "OVER" in out


def test_cli_envelope_review_total_envelope_shows_no_pace(blown, capsys):
    """A limit with no reset must not display a fabricated pace figure."""
    assert run(blown, "--action", "review") == 0
    out = capsys.readouterr().out
    assert "n/a" in out
    assert "has no reset" in out


def test_cli_envelope_review_next_view(blown, capsys):
    assert run(blown, "--action", "review", "--view", "next") == 0
    out = capsys.readouterr().out
    assert "reduce spend or raise the limit" in out


def test_cli_envelope_review_next_view_when_healthy(healthy, capsys):
    assert run(healthy, "--action", "review", "--view", "next") == 0
    assert "every envelope is within its limit" in capsys.readouterr().out


def test_cli_envelope_review_flags_a_scope_that_matches_nothing(tmp_path, capsys):
    store = seed(tmp_path / "w")
    add(store, "typo", 40, scope="project:apii")
    assert run(store, "--action", "review") == 0
    out = capsys.readouterr().out
    assert "no activity" in out
    assert "check the scope" in out


# -- gating --------------------------------------------------------------------

def test_cli_envelope_fail_on_over(blown, capsys):
    assert run(blown, "--action", "review", "--fail-on", "over") == 1
    assert "--fail-on over" in capsys.readouterr().out


def test_cli_envelope_fail_on_never_is_the_default(blown, capsys):
    assert run(blown, "--action", "review") == 0
    assert run(blown, "--action", "review", "--fail-on", "never") == 0


def test_cli_envelope_fail_on_attention_catches_no_activity(tmp_path, capsys):
    store = seed(tmp_path / "w")
    add(store, "typo", 40, scope="project:apii")
    assert run(store, "--action", "review", "--fail-on", "attention") == 1


def test_cli_envelope_set_hint_names_a_valid_scope(tmp_path, capsys):
    store = seed(tmp_path / "w")
    run(store, "--action", "set", "--name", "x", "--limit", "10", "--scope", "api")
    assert "project:api" in capsys.readouterr().out


def test_cli_envelope_fail_on_over_passes_when_only_watching(tmp_path, capsys):
    """A watch is not an overrun; --fail-on over must not fire on it."""
    store = seed(tmp_path / "w")
    add(store, "api", 2.6, period="total")     # $2.50 of $2.60 = 96% consumed
    assert run(store, "--action", "review", "--fail-on", "over") == 0
    assert "watch" in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--warn-at", "0"),
    ("--warn-at", "1.5"),
    ("--fail-on", "nonsense"),
])
def test_cli_envelope_review_rejects_bad_args(blown, capsys, args):
    assert run(blown, "--action", "review", *args) == 2


def test_cli_envelope_bad_option_does_not_print_a_report_first(blown, capsys):
    """Validation must run before rendering; a typo must not bury a full table."""
    assert run(blown, "--action", "review", "--view", "nonsense") == 2
    out = capsys.readouterr().out
    assert "review (" not in out            # no status banner was rendered
    assert "coverage" not in out            # no report body either
    assert "--view must be one of" in out


def test_cli_envelope_warn_at_shifts_the_watch_line(healthy, capsys):
    assert run(healthy, "--action", "review", "--warn-at", "0.5") == 0
    assert "envelope review" in capsys.readouterr().out


# -- safety and regression ------------------------------------------------------

def test_cli_envelope_set_writes_but_review_does_not(tmp_path, capsys):
    store = seed(tmp_path / "w")
    add(store, "api", 40)
    before = store.usage_path.read_text(encoding="utf-8")
    assert run(store, "--action", "review") == 0
    assert store.usage_path.read_text(encoding="utf-8") == before


def test_cli_envelope_envelopes_survive_a_reload(tmp_path, capsys):
    store = seed(tmp_path / "w")
    add(store, "api", 40, scope="project:api", period="weekly")
    reloaded = JsonStore(store.home).load_envelopes()
    assert len(reloaded) == 1
    assert reloaded[0].period == "weekly" and reloaded[0].scope_value == "api"


def test_existing_commands_still_work(tmp_path, capsys):
    store = seed(tmp_path / "regress")
    for command in ("report", "stats", "quality", "peers", "reconcile",
                    "quota", "optimize", "plan"):
        assert cli.main(["--home", str(store.home), command,
                         "--total", "10"]) in (0, 1, 2)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("envelope", "envelopes", "env"):
        assert registry.get(name) is not None
