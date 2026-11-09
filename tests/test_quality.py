"""Unit tests for the iteration-12 quality scoring module."""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

import pytest

from token_usage import quality
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog

NOW = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
CATALOG = default_catalog()


def ts(days_ago: int = 1) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def rec(model="gpt-4o", inp=1_000_000, out=0, days_ago=1, project="api",
        tags=("api",)) -> UsageRecord:
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts(days_ago), project=project, tags=list(tags))


def at(record: UsageRecord, days_ahead: int) -> UsageRecord:
    record.timestamp = (NOW + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%S")
    return record


# -- score, grade, structure ------------------------------------------------

@pytest.mark.parametrize("score,label", [
    (100, "excellent"), (90, "excellent"), (89.9, "good"), (74.9, "fair"),
    (59.9, "poor"), (39.9, "unusable"),
])
def test_grade(score, label):
    assert quality.grade(score) == label


def test_audit_empty_is_flagged_rather_than_clean():
    result = quality.audit([], CATALOG, now=NOW)
    assert result["empty"] is True and result["checks"] == []
    assert result["score"] == 100.0


def test_clean_dataset_scores_full_marks():
    result = quality.audit([rec(days_ago=d) for d in range(1, 6)], CATALOG, now=NOW)
    assert result["score"] == 100.0 and result["failing"] == []


def test_run_checks_emits_every_documented_id_in_order():
    result = quality.audit([rec()], CATALOG, now=NOW)
    assert [c["id"] for c in result["checks"]] == list(quality.CHECK_IDS)


def test_score_is_one_hundred_minus_the_penalty_total():
    result = quality.audit([rec(), rec(inp=0, out=0, days_ago=2)], CATALOG, now=NOW)
    assert result["score"] == round(100.0 - result["penalty_total"], 1)
    assert result["warnings"] >= 1


# -- money-weighted scoring -------------------------------------------------

def test_duplicates_are_weighted_by_money_not_by_row_count():
    copy = rec()
    check = quality.check_duplicates(
        [copy, copy, rec(model="gpt-4o-mini", days_ago=2)], CATALOG)
    assert check["affected"] == 1
    assert check["share"] > 0.4                  # 33% of rows, ~50% of spend
    assert check["severity"] == "critical"


def test_untagged_and_default_project_are_measured_in_money():
    records = [rec(), rec(days_ago=2, tags=(), project="default")]
    for check in (quality.check_untagged(records, CATALOG),
                  quality.check_default_project(records, CATALOG)):
        assert check["affected"] == 1 and check["share"] == 0.5


def test_a_failing_check_is_never_worth_zero_points():
    """A critical defect on a sliver of spend still costs MIN_PENALTY."""
    cheap = rec(model="gpt-4o-mini", inp=100, out=10, days_ago=2)
    check = quality.check_duplicates([rec(days_ago=d) for d in range(1, 20)]
                                     + [cheap, cheap], CATALOG)
    assert check["ok"] is False and check["share"] < 0.01
    assert check["penalty"] == quality.MIN_PENALTY


def test_penalty_is_capped_per_check():
    check = quality.check_zero_tokens([rec(inp=0, out=0) for _ in range(4)])
    assert check["share"] == 1.0
    assert check["penalty"] == quality.MAX_PENALTY["warning"]


def test_zero_tokens_penalty_tracks_the_affected_share():
    check = quality.check_zero_tokens([rec(), rec(inp=0, out=0, days_ago=2)])
    assert check["penalty"] == round(quality.MAX_PENALTY["warning"] * 0.5, 1)


# -- structural checks ------------------------------------------------------

def test_unknown_models_are_named_in_the_summary():
    records = [rec(model="mystery-a"), rec(model="mystery-a", days_ago=2),
               rec(model="mystery-b", days_ago=3), rec()]
    check = quality.check_unknown_models(records, CATALOG)
    assert check["affected"] == 3
    assert "mystery-a" in check["summary"] and "mystery-b" in check["summary"]


def test_future_timestamps_tolerate_a_day_of_clock_skew():
    skew = rec()                       # 20 hours ahead sits inside the allowance
    skew.timestamp = (NOW + timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%S")
    check = quality.check_future_timestamps([skew, at(rec(days_ago=2), 5)], now=NOW)
    assert check["affected"] == 1 and check["severity"] == "critical"


def test_cost_override_drift_ignores_pins_inside_tolerance():
    pinned, drifted = rec(), rec(days_ago=2)
    pinned.cost_usd = CATALOG.cost_for_record(pinned) * 1.005       # within 1%
    drifted.cost_usd = 99.0
    check = quality.check_cost_override_drift([pinned, drifted], CATALOG)
    assert check["affected"] == 1 and check["samples"][0]["stored_usd"] == 99.0


def test_extreme_tokens_declines_to_judge_a_small_sample():
    check = quality.check_extreme_tokens([rec(days_ago=i % 5) for i in range(9)])
    assert check["ok"] is True and "not evaluated" in check["summary"]


def test_extreme_tokens_flags_one_giant_record_among_many():
    check = quality.check_extreme_tokens([rec(inp=1_000) for _ in range(500)]
                                         + [rec(inp=1_000_000)])
    assert check["affected"] == 1 and check["severity"] == "notice"


def test_extreme_tokens_survives_an_outlier_that_hides_from_a_mean_test():
    """Small samples let one huge row inflate mean+stdev past itself.

    With about twenty records the 5-sigma threshold lands *above* the outlier, so
    the mean-based test used elsewhere in this package reports the dataset clean.
    The median/MAD rule still isolates it, which is why this check does not reuse
    the naive z-score.
    """
    records = [rec(inp=1_000 * n) for n in range(1, 21)] + [rec(inp=900_000_000)]
    values = [r.total_tokens for r in records]
    assert max(values) < statistics.mean(values) + 5 * statistics.pstdev(values)
    assert quality.check_extreme_tokens(records)["affected"] == 1


def test_extreme_tokens_falls_back_when_the_median_is_flat():
    check = quality.check_extreme_tokens([rec(inp=1_000) for _ in range(10)]
                                         + [rec(inp=5_000_000)])
    assert check["affected"] == 1 and "too flat" in check["summary"]


# -- coverage and hygiene ---------------------------------------------------

def test_day_coverage_counts_the_unrecorded_days():
    check = quality.check_day_coverage([rec(days_ago=1), rec(days_ago=5)])
    assert check["affected"] == 3 and check["share"] == round(3 / 5, 4)


def test_retention_counts_records_past_the_horizon():
    check = quality.check_retention([rec(days_ago=10), rec(days_ago=400)],
                                    now=NOW, max_age_days=365)
    assert check["affected"] == 1


# -- worklist and filtering -------------------------------------------------

def test_next_steps_puts_critical_findings_first():
    result = quality.audit([rec(inp=0, out=0), at(rec(days_ago=4, tags=()), 9)],
                           CATALOG, now=NOW)
    severities = [step["severity"] for step in quality.next_steps(result)]
    assert severities == sorted(severities, key=quality.severity_rank)
    assert severities[0] == "critical"


def test_next_steps_respects_the_limit():
    result = quality.audit([rec(inp=0, out=0), rec(days_ago=2, tags=())],
                           CATALOG, now=NOW)
    assert len(quality.next_steps(result, limit=1)) == 1


def test_filtering_checks_does_not_improve_the_score():
    result = quality.audit([rec(inp=0, out=0)], CATALOG, now=NOW)
    kept = quality.filter_checks(result["checks"], min_severity="critical")
    assert all(check["severity"] == "critical" for check in kept)
    assert result["score"] < 100.0


def test_filter_checks_rejects_an_unknown_severity():
    with pytest.raises(ValueError):
        quality.filter_checks([], min_severity="bogus")
