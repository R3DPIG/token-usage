"""CLI tests for the iteration-12 quality plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def ts(days_ago: int = 1) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def rec(inp=1_000_000, out=500, days_ago=1, model="gpt-4o", project="api",
        tags=("api",)) -> UsageRecord:
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts(days_ago), project=project, tags=list(tags))


def ahead(record: UsageRecord, days: int) -> UsageRecord:
    record.timestamp = (NOW + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    return record


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def run(store, *args):
    return cli.main(["--home", str(store.home), "quality", *args])


@pytest.fixture
def clean(store):
    store.append_records([rec(days_ago=d) for d in range(1, 5)])
    return store


@pytest.fixture
def dirty(store):
    """One of every critical / warning defect, plus notices."""
    records = [rec(days_ago=d) for d in range(1, 7)]
    records.append(records[0])                                   # duplicate
    records.append(rec(inp=0, out=0, days_ago=2))                # zero tokens
    records.append(rec(model="mystery-1", days_ago=3))           # unknown model
    records.append(ahead(rec(days_ago=8), 30))                   # future stamp
    drifted = rec(days_ago=4)
    drifted.cost_usd = 42.0
    records.append(drifted)                                      # override drift
    records.append(rec(days_ago=9, project="default", tags=()))   # notices
    store.append_records(records)
    return store


# -- empty and clean --------------------------------------------------------

def test_cli_quality_no_records(store, capsys):
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "nothing to audit" in out and "sample" in out


def test_cli_quality_clean_dataset(clean, capsys):
    assert run(clean) == 0
    out = capsys.readouterr().out
    assert "100.0 / 100" in out and "excellent" in out and "safe to report" in out


# -- dirty dataset ----------------------------------------------------------

def test_cli_quality_summary_reports_findings(dirty, capsys):
    assert run(dirty) == 1
    out = capsys.readouterr().out
    assert "data quality audit" in out and "/ 100" in out and "#" in out
    assert "worst findings" in out and "critical" in out


def test_cli_quality_checks_lists_every_check(dirty, capsys):
    assert run(dirty, "--view", "checks") == 1
    out = capsys.readouterr().out
    for check_id in ("duplicates", "future_timestamps", "unknown_models",
                     "untagged", "day_coverage"):
        assert check_id in out


def test_cli_quality_checks_min_severity_filter(dirty, capsys):
    assert run(dirty, "--view", "checks", "--min-severity", "critical") == 1
    out = capsys.readouterr().out
    assert "[CRIT]" in out and "[note]" not in out
    # Hiding checks must not improve the score, and the output says so.
    assert "the score still counts all of them" in out


def test_cli_quality_detail_requires_a_check(dirty, capsys):
    assert run(dirty, "--view", "detail") == 2
    assert "requires --check" in capsys.readouterr().out


def test_cli_quality_detail_shows_evidence(dirty, capsys):
    assert run(dirty, "--view", "detail", "--check", "duplicates") == 1
    out = capsys.readouterr().out
    assert "max deduction" in out and "dedupe --apply" in out


def test_cli_quality_detail_on_passing_check_returns_zero(clean, capsys):
    assert run(clean, "--view", "detail", "--check", "duplicates") == 0
    assert "status    pass" in capsys.readouterr().out


def test_cli_quality_next_ranks_critical_first(dirty, capsys):
    assert run(dirty, "--view", "next") == 1
    out = capsys.readouterr().out
    assert "ranked worklist" in out and "recovers" in out
    if "[note]" in out:
        assert out.index("[CRIT]") < out.index("[note]")


def test_cli_quality_next_respects_top(dirty, capsys):
    run(dirty, "--view", "next", "--top", "1")
    out = capsys.readouterr().out
    assert "1. " in out and "2. " not in out


# -- exit policy ------------------------------------------------------------

def test_cli_quality_strict_fails_on_notices_only(store, capsys):
    store.append_records([rec(days_ago=d, tags=(), project="default")
                          for d in range(1, 6)])
    assert run(store) == 0
    capsys.readouterr()
    assert run(store, "--strict") == 1


# -- validation, filtering, regression --------------------------------------

@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--check", "nonsense"),
    ("--min-severity", "nonsense"),
])
def test_cli_quality_rejects_bad_args(dirty, capsys, args):
    assert run(dirty, *args) == 2


def test_cli_quality_project_filter(store, capsys):
    store.append_records([rec(project="api"), rec(project="web", days_ago=2)])
    assert run(store, "--project", "api") == 0
    assert "records   1" in capsys.readouterr().out


def test_cli_quality_filter_matching_nothing_is_not_an_error(store, capsys):
    store.append_records([rec()])
    assert run(store, "--project", "absent") == 0
    assert "no records in scope" in capsys.readouterr().out


def test_existing_commands_still_work(store, capsys):
    store.append_records([rec()])
    for command in ("report", "stats", "pattern", "optimize", "chargeback",
                    "drivers", "simulate", "plan", "doctor"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("quality", "qa", "audit"):
        assert registry.get(name) is not None
