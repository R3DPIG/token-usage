"""CLI tests for the iteration-10 drivers plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore
from token_usage.utils import ISO_FORMAT, period_start


def _window_stamps(period: str = "monthly"):
    """One timestamp inside the current window and one inside the previous.

    The previous window mirrors the elapsed part of the current period, so it is
    usually only a few days wide at the start of a month. Fixed offsets like
    "40 days ago" would fall outside it; deriving the midpoint cannot.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = period_start(period, now=now)
    elapsed = now - start
    if elapsed <= timedelta(seconds=2):        # guard the midnight boundary
        elapsed = timedelta(seconds=2)
    previous_start = start - elapsed
    current = start + elapsed / 2
    previous = previous_start + elapsed / 2
    return current.strftime(ISO_FORMAT), previous.strftime(ISO_FORMAT)


CUR_TS, PREV_TS = _window_stamps()


def rec(model="gpt-4o", inp=1_000_000, out=0, ts=None, project="default"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts or CUR_TS, project=project)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def run(store, *args):
    return cli.main(["--home", str(store.home), "drivers", *args])


# -- summary ----------------------------------------------------------------

def test_cli_drivers_no_records(store, capsys):
    assert run(store) == 0
    assert "nothing to explain" in capsys.readouterr().out


def test_cli_drivers_flat_when_unchanged(store, capsys):
    store.append_records([rec(ts=PREV_TS), rec(ts=CUR_TS)])
    assert run(store) == 0
    assert "unchanged" in capsys.readouterr().out


def test_cli_drivers_summary_reports_volume(store, capsys):
    store.append_records([rec(inp=1_000_000, ts=PREV_TS),
                          rec(inp=3_000_000, ts=CUR_TS)])
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "spend is up" in out
    assert "volume" in out and "dominant driver" in out


def test_cli_drivers_summary_shows_reconciliation(store, capsys):
    store.append_records([rec(inp=1_000_000, ts=PREV_TS),
                          rec(inp=2_000_000, ts=CUR_TS)])
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "reconciliation" in out and "drift" in out


def test_cli_drivers_summary_lists_top_movers(store, capsys):
    store.append_records([rec(model="gpt-4o", inp=1_000_000, ts=PREV_TS),
                          rec(model="claude-3-opus", inp=1_000_000, ts=CUR_TS)])
    assert run(store) == 0
    assert "top movers" in capsys.readouterr().out


# -- views ------------------------------------------------------------------

@pytest.mark.parametrize("view,expected", [
    ("models", "per-model contribution"),
    ("mix", "token share by model"),
    ("price", "blended unit price"),
])
def test_cli_drivers_views(store, capsys, view, expected):
    store.append_records([rec(inp=1_000_000, ts=PREV_TS),
                          rec(inp=2_000_000, ts=CUR_TS)])
    assert run(store, "--view", view) == 0
    assert expected in capsys.readouterr().out


def test_cli_drivers_models_shows_status(store, capsys):
    store.append_records([rec(model="gpt-4o", inp=1_000_000, ts=PREV_TS),
                          rec(model="gpt-4o-mini", inp=1_000_000, ts=CUR_TS)])
    assert run(store, "--view", "models") == 0
    out = capsys.readouterr().out
    assert "new" in out and "dropped" in out
    assert "only one period" in out          # assumed-price warning


def test_cli_drivers_mix_shows_percentage_points(store, capsys):
    store.append_records([rec(model="gpt-4o", inp=1_000_000, ts=PREV_TS),
                          rec(model="gpt-4o-mini", inp=1_000_000, ts=CUR_TS)])
    assert run(store, "--view", "mix") == 0
    assert "pp" in capsys.readouterr().out


def test_cli_drivers_price_states_no_movement(store, capsys):
    store.append_records([rec(inp=1_000_000, ts=PREV_TS),
                          rec(inp=2_000_000, ts=CUR_TS)])
    assert run(store, "--view", "price") == 0
    assert "did not move" in capsys.readouterr().out


def test_cli_drivers_price_reports_movement_when_mix_shifts(store, capsys):
    """A 100x blended-price jump must not be reported as "did not move".

    The isolated ``unit_price`` driver is zero here (the change is attributed to
    mix), so a view that only inspects that driver would deny a movement that is
    plainly visible in the two blended prices.
    """
    store.append_records([rec(model="gpt-4o-mini", inp=1_000_000, ts=PREV_TS),
                          rec(model="claude-3-opus", inp=1_000_000, ts=CUR_TS)])
    assert run(store, "--view", "price") == 0
    out = capsys.readouterr().out
    assert "movement" in out and "did not move" not in out
    assert "the rest of the movement is mix" in out


def test_cli_drivers_price_handles_one_sided_data(store, capsys):
    store.append_records([rec(inp=1_000_000, ts=CUR_TS)])
    assert run(store, "--view", "price") == 0
    assert "only one period has data" in capsys.readouterr().out


# -- period options ---------------------------------------------------------

@pytest.mark.parametrize("period", ["daily", "weekly", "monthly"])
def test_cli_drivers_periods(store, capsys, period):
    store.append_records([rec(ts=_window_stamps(period)[1])])
    assert run(store, "--period", period) == 0
    assert period in capsys.readouterr().out


# -- validation -------------------------------------------------------------

@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--period", "yearly"),
])
def test_cli_drivers_rejects_bad_args(store, capsys, args):
    store.append_records([rec()])
    assert run(store, *args) == 2


# -- filtering and regression -----------------------------------------------

def test_cli_drivers_project_filter(store, capsys):
    """Filtering to one project must exclude the other project's records."""
    store.append_records([rec(inp=1_000_000, ts=CUR_TS, project="api"),
                          rec(inp=5_000_000, ts=PREV_TS, project="web")])
    assert run(store, "--project", "api") == 0
    out = capsys.readouterr().out
    # Only the api record survives: $0.0000 -> $2.5000. Had the 5M web record
    # leaked in, the previous side would be non-zero.
    assert "$0.0000 -> $2.5000" in out


def test_existing_commands_still_work(store, capsys):
    store.append_records([rec()])
    for command in ("report", "stats", "pattern", "optimize", "chargeback",
                    "compare", "doctor"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    assert registry.get("drivers") is not None
    assert registry.get("why") is not None           # alias
