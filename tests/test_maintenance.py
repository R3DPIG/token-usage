"""Tests for the iteration-3 maintenance layer (pure functions)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from token_usage import maintenance
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog


def rec(model="gpt-4o", inp=1000, out=500, ts="2024-05-01T12:00:00",
        cached=0, project="demo", tags=None, note=""):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, cached_input_tokens=cached, project=project,
                       tags=tags or [], note=note)


# -- duplicates -------------------------------------------------------------

def test_fingerprint_includes_timestamp_by_default():
    a, b = rec(), rec(ts="2024-05-02T12:00:00")
    assert maintenance.fingerprint(a) != maintenance.fingerprint(b)
    assert maintenance.fingerprint(a, ignore_timestamp=True) == \
        maintenance.fingerprint(b, ignore_timestamp=True)


def test_find_duplicates_groups_only_repeats():
    records = [rec(), rec(), rec(ts="2024-05-02T12:00:00")]
    groups = maintenance.find_duplicates(records)
    assert groups == [[0, 1]]


def test_dedupe_keeps_first_and_returns_removed():
    records = [rec(note="a"), rec(note="b"), rec(ts="2024-05-02T12:00:00")]
    kept, removed, groups = maintenance.dedupe(records)
    assert len(kept) == 2 and len(removed) == 1
    assert kept[0].note == "a"          # first survives
    assert removed[0].note == "b"
    assert groups == [[0, 1]]


def test_dedupe_keep_last():
    records = [rec(note="a"), rec(note="b")]
    kept, removed, _ = maintenance.dedupe(records, keep="last")
    assert kept[0].note == "b" and removed[0].note == "a"


def test_dedupe_rejects_bad_keep():
    with pytest.raises(ValueError):
        maintenance.dedupe([rec()], keep="middle")


def test_dedupe_ignore_timestamp_merges_clock_skew():
    records = [rec(ts="2024-05-01T12:00:00"), rec(ts="2024-05-01T12:00:01")]
    assert len(maintenance.dedupe(records)[0]) == 2          # distinct by default
    assert len(maintenance.dedupe(records, ignore_timestamp=True)[0]) == 1


# -- pruning ----------------------------------------------------------------

def test_prune_by_project():
    records = [rec(project="a"), rec(project="b")]
    kept, removed = maintenance.prune(records, project="a")
    assert len(kept) == 1 and kept[0].project == "b"
    assert len(removed) == 1 and removed[0].project == "a"


def test_prune_before_is_exclusive():
    records = [rec(ts="2024-04-30T00:00:00"), rec(ts="2024-05-01T00:00:00")]
    kept, removed = maintenance.prune(records, before="2024-05-01")
    assert len(removed) == 1 and removed[0].timestamp.startswith("2024-04-30")
    assert len(kept) == 1


def test_prune_after_is_inclusive():
    """``--after X`` deletes X and everything later, so X is removed."""
    records = [rec(ts="2024-04-30T00:00:00"), rec(ts="2024-05-01T00:00:00")]
    kept, removed = maintenance.prune(records, after="2024-05-01")
    assert len(kept) == 1 and kept[0].timestamp.startswith("2024-04-30")
    assert len(removed) == 1 and removed[0].timestamp.startswith("2024-05-01")


def test_prune_after_and_before_form_a_range():
    """Together they delete the half-open interval [after, before)."""
    records = [rec(ts="2024-04-01T00:00:00"), rec(ts="2024-05-01T00:00:00"),
               rec(ts="2024-06-01T00:00:00")]
    kept, removed = maintenance.prune(records, after="2024-05-01", before="2024-06-01")
    assert len(removed) == 1 and removed[0].timestamp.startswith("2024-05-01")
    assert len(kept) == 2


def test_prune_combines_criteria_with_and():
    records = [rec(project="a", model="gpt-4o"), rec(project="a", model="gpt-4o-mini")]
    kept, removed = maintenance.prune(records, project="a", model="gpt-4o")
    assert len(removed) == 1 and removed[0].model == "gpt-4o"


def test_prune_by_tag():
    records = [rec(tags=["scratch"]), rec(tags=["prod"])]
    _, removed = maintenance.prune(records, tag="scratch")
    assert len(removed) == 1


def test_split_by_age():
    records = [rec(ts="2023-12-31T00:00:00"), rec(ts="2024-06-01T00:00:00")]
    recent, old = maintenance.split_by_age(records, "2024-01-01")
    assert len(old) == 1 and len(recent) == 1


# -- editing ----------------------------------------------------------------

def test_apply_edits_changes_fields():
    updated = maintenance.apply_edits(rec(), {"project": "prod", "input_tokens": 42})
    assert updated.project == "prod" and updated.input_tokens == 42


def test_apply_edits_parses_tag_string():
    updated = maintenance.apply_edits(rec(), {"tags": "a, b ,c"})
    assert updated.tags == ["a", "b", "c"]


def test_apply_edits_rejects_unknown_field():
    with pytest.raises(ValueError):
        maintenance.apply_edits(rec(), {"cost_usd": 999})


def test_apply_edits_revalidates_invariants():
    # cached > input must be rejected by the model, not silently accepted.
    with pytest.raises(ValueError):
        maintenance.apply_edits(rec(inp=100), {"cached_input_tokens": 500})


def test_diff_records_reports_changes():
    before = rec()
    after = maintenance.apply_edits(before, {"project": "prod"})
    diff = maintenance.diff_records(before, after)
    assert diff == [("project", "demo", "prod")]


# -- merging ----------------------------------------------------------------

def test_merge_records_dedupes_by_default():
    a = [rec(note="x")]
    b = [rec(note="x"), rec(ts="2024-05-02T12:00:00")]
    merged, stats = maintenance.merge_records(a, b)
    assert len(merged) == 2 and stats["duplicates_removed"] == 1


def test_merge_records_can_skip_dedupe():
    merged, stats = maintenance.merge_records([rec()], [rec()], dedupe_after=False)
    assert len(merged) == 2 and stats["duplicates_removed"] == 0


# -- integrity --------------------------------------------------------------

def test_scan_integrity_flags_zero_tokens():
    issues = maintenance.scan_integrity([rec(inp=0, out=0)], default_catalog())
    assert any(i["kind"] == "zero_tokens" for i in issues)


def test_scan_integrity_flags_future_timestamp():
    now = datetime(2024, 5, 1, tzinfo=timezone.utc)
    issues = maintenance.scan_integrity([rec(ts="2030-01-01T00:00:00")],
                                        default_catalog(), now=now)
    assert any(i["kind"] == "future_timestamp" for i in issues)


def test_scan_integrity_clean_record_has_no_issues():
    now = datetime(2024, 5, 2, tzinfo=timezone.utc)
    assert maintenance.scan_integrity([rec()], default_catalog(), now=now) == []


def test_summarize_issues_counts_by_kind():
    issues = [{"kind": "zero_tokens"}, {"kind": "zero_tokens"},
              {"kind": "unknown_model"}]
    assert maintenance.summarize_issues(issues) == {"zero_tokens": 2, "unknown_model": 1}


def test_estimate_value():
    value = maintenance.estimate_value([rec(inp=1_000_000, out=0)], default_catalog())
    assert value["records"] == 1 and value["tokens"] == 1_000_000
    assert value["cost_usd"] > 0
