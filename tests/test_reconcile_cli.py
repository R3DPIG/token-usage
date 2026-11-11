"""CLI tests for the iteration-14 reconcile plugin."""

from __future__ import annotations

import json

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore

# gpt-4o: $2.50/1M input -> one 1M-input record prices at exactly $2.50.


def seed(home, models=("gpt-4o",)):
    store = JsonStore(home)
    store.append_records([
        UsageRecord(model=model, input_tokens=1_000_000, output_tokens=0,
                    timestamp="2026-11-01T00:00:00", project="api")
        for model in models])
    return store


def write_bill(tmp_path, name, rows):
    path = tmp_path / name
    if name.endswith(".csv"):
        header = "model,amount_usd,input_tokens,output_tokens"
        lines = [header] + [f"{r['model']},{r['amount_usd']},"
                            f"{r.get('input_tokens', '')},"
                            f"{r.get('output_tokens', '')}" for r in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def run(store, *args):
    return cli.main(["--home", str(store.home), "reconcile", *args])


# -- argument and error paths ----------------------------------------------

def test_cli_reconcile_requires_a_bill(tmp_path, capsys):
    assert run(seed(tmp_path / "w")) == 2
    assert "--bill is required" in capsys.readouterr().out


def test_cli_reconcile_rejects_a_missing_bill_file(tmp_path, capsys):
    store = seed(tmp_path / "w")
    assert run(store, "--bill", str(tmp_path / "nope.csv")) == 2
    assert "not found" in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--tolerance", "-1"),
])
def test_cli_reconcile_rejects_bad_args(tmp_path, capsys, args):
    store = seed(tmp_path / "w")
    bill = write_bill(tmp_path, "bill.csv",
                      [{"model": "gpt-4o", "amount_usd": 2.50}])
    assert run(store, "--bill", str(bill), *args) == 2


def test_cli_reconcile_with_no_usable_lines(tmp_path, capsys):
    store = seed(tmp_path / "w")
    bill = write_bill(tmp_path, "empty.csv", [])
    assert run(store, "--bill", str(bill)) == 0
    assert "no usable bill lines" in capsys.readouterr().out


def test_cli_reconcile_reports_bad_rows_but_keeps_going(tmp_path, capsys):
    store = seed(tmp_path / "w")
    bill = write_bill(tmp_path, "mixed.csv", [
        {"model": "gpt-4o", "amount_usd": 2.50},
        {"model": "no-price", "amount_usd": "n/a"},
    ])
    assert run(store, "--bill", str(bill)) == 0
    out = capsys.readouterr().out
    assert "invalid amount_usd" in out and "bill reconciliation (clean)" in out


# -- views -------------------------------------------------------------------

@pytest.fixture
def clean(tmp_path):
    store = seed(tmp_path / "w")
    bill = write_bill(tmp_path, "bill.csv",
                      [{"model": "gpt-4o", "amount_usd": 2.50}])
    return store, bill


@pytest.fixture
def gapped(tmp_path):
    """Bill overcharges gpt-4o, bills an unrecorded model, misses a local one."""
    store = seed(tmp_path / "w", models=("gpt-4o", "gpt-4o-mini"))
    bill = write_bill(tmp_path, "bill.json", [
        {"model": "gpt-4o", "amount_usd": 4.00,
         "input_tokens": 1_000_000, "output_tokens": 0},
        {"model": "claude-3-opus", "amount_usd": 15.0},
    ])
    return store, bill


def test_cli_reconcile_summary_when_clean(clean, capsys):
    store, bill = clean
    assert run(store, "--bill", str(bill)) == 0
    out = capsys.readouterr().out
    assert "bill reconciliation (clean)" in out
    assert "coverage   100.0%" in out
    assert "nothing to chase" in out


def test_cli_reconcile_models_view_lists_every_model(gapped, capsys):
    store, bill = gapped
    assert run(store, "--bill", str(bill), "--view", "models") == 0
    out = capsys.readouterr().out
    assert "gpt-4o" in out and "claude-3-opus" in out and "gpt-4o-mini" in out
    assert "MISMATCH" in out and "not local" in out and "not on bill" in out


def test_cli_reconcile_gaps_view_narrows_the_cause(gapped, capsys):
    store, bill = gapped
    assert run(store, "--bill", str(bill), "--view", "gaps") == 0
    out = capsys.readouterr().out
    assert "tokens: price -- token counts agree" in out
    assert "implied $4.0000/1M vs catalog $2.5000/1M" in out


def test_cli_reconcile_gaps_view_says_not_evaluated_without_tokens(tmp_path,
                                                                   capsys):
    store = seed(tmp_path / "w")
    bill = write_bill(tmp_path, "b.csv",
                      [{"model": "gpt-4o", "amount_usd": 9.00}])
    assert run(store, "--bill", str(bill), "--view", "gaps") == 0
    out = capsys.readouterr().out
    assert "tokens: not evaluated -- the bill carries no token counts" in out


def test_cli_reconcile_explain_view_states_its_limits(gapped, capsys):
    store, bill = gapped
    assert run(store, "--bill", str(bill), "--view", "explain") == 0
    out = capsys.readouterr().out
    assert "is consistent with:" in out
    assert "not conclusions" in out


def test_cli_reconcile_warns_about_low_coverage(gapped, capsys):
    store, bill = gapped
    assert run(store, "--bill", str(bill)) == 0
    assert "unreliable" in capsys.readouterr().out


def test_cli_reconcile_gaps_view_when_clean(clean, capsys):
    store, bill = clean
    assert run(store, "--bill", str(bill), "--view", "gaps") == 0
    assert "every model agrees" in capsys.readouterr().out


# -- behaviour knobs -----------------------------------------------------------

def test_cli_reconcile_strict_fails_on_gaps_but_not_when_clean(tmp_path, capsys):
    clean_store, clean_bill = seed(tmp_path / "a"), None
    clean_bill = write_bill(tmp_path / "a", "b.csv",
                            [{"model": "gpt-4o", "amount_usd": 2.50}])
    assert run(clean_store, "--bill", str(clean_bill), "--strict") == 0
    capsys.readouterr()

    gap_store = seed(tmp_path / "b")
    gap_bill = write_bill(tmp_path / "b", "b.csv",
                          [{"model": "gpt-4o", "amount_usd": 40.0}])
    assert run(gap_store, "--bill", str(gap_bill), "--strict") == 1
    assert "--strict: unexplained gaps remain" in capsys.readouterr().out


def test_cli_reconcile_tolerance_absorbs_small_gaps(tmp_path, capsys):
    store = seed(tmp_path / "w")
    bill = write_bill(tmp_path, "b.csv",
                      [{"model": "gpt-4o", "amount_usd": 3.40}])
    assert run(store, "--bill", str(bill)) == 0
    assert "(clean)" in capsys.readouterr().out
    capsys.readouterr()
    # A tighter tolerance exposes the same 90-cent gap.
    assert run(store, "--bill", str(bill), "--tolerance", "0.25") == 0
    assert "(gap)" in capsys.readouterr().out


def test_cli_reconcile_model_filter_scopes_the_workspace(tmp_path, capsys):
    store = seed(tmp_path / "w", models=("gpt-4o", "gpt-4o-mini"))
    bill = write_bill(tmp_path, "b.csv",
                      [{"model": "gpt-4o", "amount_usd": 2.50}])
    assert run(store, "--bill", str(bill), "--model", "gpt-4o",
               "--view", "models") == 0
    out = capsys.readouterr().out
    assert "gpt-4o" in out and "gpt-4o-mini" not in out


# -- safety and regression -----------------------------------------------------

def test_cli_reconcile_never_writes_to_the_workspace(tmp_path, capsys):
    store = seed(tmp_path / "w")
    before = store.usage_path.read_text(encoding="utf-8")
    bill = write_bill(tmp_path, "b.csv",
                      [{"model": "gpt-4o", "amount_usd": 99.0}])
    assert run(store, "--bill", str(bill), "--apply") == 0
    assert store.usage_path.read_text(encoding="utf-8") == before


def test_existing_commands_still_work(tmp_path, capsys):
    store = seed(tmp_path / "regress")
    for command in ("report", "stats", "quality", "peers", "optimize", "plan",
                    "doctor", "import"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1, 2)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("reconcile", "bill", "audit-bill"):
        assert registry.get(name) is not None
