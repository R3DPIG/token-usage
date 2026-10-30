"""Tests for the iteration-6 planning layer (pure logic)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from token_usage import planning
from token_usage.models import Budget, UsageRecord
from token_usage.pricing import default_catalog


def rec(model="gpt-4o", inp=1_000_000, out=0, ts="2024-05-10T12:00:00", project="demo"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, project=project)


# 2024-05-15 is a Wednesday; May has 31 days.
NOW = datetime(2024, 5, 15, 12, 0, tzinfo=timezone.utc)


# -- period arithmetic ------------------------------------------------------

def test_period_bounds_monthly():
    start, end = planning.period_bounds("monthly", now=NOW)
    assert start.day == 1 and start.month == 5
    assert end.day == 1 and end.month == 6


def test_period_bounds_december_rollover():
    december = datetime(2024, 12, 15, tzinfo=timezone.utc)
    start, end = planning.period_bounds("monthly", now=december)
    assert (start.year, start.month) == (2024, 12)
    assert (end.year, end.month) == (2025, 1)


def test_period_bounds_weekly_starts_monday():
    start, end = planning.period_bounds("weekly", now=NOW)
    assert start.weekday() == 0          # Monday
    assert (end - start).days == 7


def test_period_bounds_daily():
    start, end = planning.period_bounds("daily", now=NOW)
    assert (end - start).days == 1


def test_period_bounds_rejects_unknown():
    with pytest.raises(ValueError):
        planning.period_bounds("yearly", now=NOW)


def test_days_elapsed_and_remaining_sum_to_period_length():
    elapsed = planning.days_elapsed("monthly", now=NOW)
    remaining = planning.days_remaining("monthly", now=NOW)
    assert elapsed == 15                 # May 1..15 inclusive
    assert elapsed + remaining == 31     # May has 31 days


def test_days_remaining_never_zero_on_last_day():
    last_day = datetime(2024, 5, 31, 23, 0, tzinfo=timezone.utc)
    assert planning.days_remaining("monthly", now=last_day) >= 1


# -- projection -------------------------------------------------------------

def test_project_period_end_extrapolates_pace():
    # $0.01/day for 15 days -> ~$0.31 by month end.
    records = [rec(inp=4_000_000, ts=f"2024-05-{day:02d}T12:00:00")
               for day in range(1, 16)]
    projection = planning.project_period_end(records, default_catalog(),
                                             period="monthly", now=NOW)
    assert projection["days_elapsed"] == 15
    assert projection["days_remaining"] == 16
    assert projection["projected_total_usd"] > projection["spent_usd"]
    assert projection["daily_rate_usd"] > 0


def test_project_period_end_ignores_other_periods():
    records = [rec(ts="2024-04-10T12:00:00"), rec(ts="2024-05-10T12:00:00")]
    projection = planning.project_period_end(records, default_catalog(),
                                             period="monthly", now=NOW)
    assert projection["records"] == 1    # April record excluded


def test_project_period_end_empty_is_zero():
    projection = planning.project_period_end([], default_catalog(),
                                             period="monthly", now=NOW)
    assert projection["spent_usd"] == 0.0
    assert projection["projected_total_usd"] == 0.0


# -- budget outcome ---------------------------------------------------------

def test_budget_outcome_on_track():
    records = [rec(inp=1_000_000, ts="2024-05-10T12:00:00")]
    outcome = planning.budget_outcome(records, default_catalog(), 100.0,
                                      period="monthly", now=NOW)
    assert outcome["verdict"] == "on_track"
    assert outcome["daily_allowance_usd"] > 0
    assert outcome["overage_usd"] == 0.0


def test_budget_outcome_at_risk():
    """Projected to overshoot, but not yet over: spend is still under the limit."""
    # 10k input tokens on gpt-4o = $0.025/day; 15 days = $0.375 spent, and the
    # projection lands near $0.78 against a $0.50 limit.
    records = [rec(inp=10_000, ts=f"2024-05-{day:02d}T12:00:00")
               for day in range(1, 16)]
    outcome = planning.budget_outcome(records, default_catalog(), 0.5,
                                      period="monthly", now=NOW)
    assert outcome["spent_usd"] < 0.5          # not yet exceeded
    assert outcome["projected_total_usd"] > 0.5
    assert outcome["verdict"] == "at_risk"
    assert outcome["overage_usd"] > 0
    assert outcome["required_reduction_pct"] > 0


def test_budget_outcome_exceeded():
    records = [rec(inp=100_000_000, ts="2024-05-10T12:00:00")]
    outcome = planning.budget_outcome(records, default_catalog(), 0.01,
                                      period="monthly", now=NOW)
    assert outcome["verdict"] == "exceeded"
    assert outcome["headroom_usd"] == 0.0


def test_budget_outcome_no_budget():
    outcome = planning.budget_outcome([rec()], default_catalog(), 0.0,
                                      period="monthly", now=NOW)
    assert outcome["verdict"] == "no_budget"


# -- quota allocation -------------------------------------------------------

def test_allocate_quota_sums_to_total():
    records = [rec(project="a", inp=3_000_000), rec(project="b", inp=1_000_000)]
    quotas = planning.allocate_quota(100.0, records, default_catalog(),
                                     dimension="project")
    assert len(quotas) == 2
    assert sum(q["quota_usd"] for q in quotas) == pytest.approx(100.0, abs=1e-4)


def test_allocate_quota_proportional_to_history():
    records = [rec(project="big", inp=9_000_000), rec(project="small", inp=1_000_000)]
    quotas = planning.allocate_quota(100.0, records, default_catalog(),
                                     dimension="project")
    by_name = {q["bucket"]: q for q in quotas}
    assert by_name["big"]["quota_usd"] > by_name["small"]["quota_usd"]
    assert by_name["big"]["share"] == pytest.approx(0.9, abs=0.01)


def test_allocate_quota_floor_is_honoured():
    records = [rec(project="big", inp=9_000_000), rec(project="small", inp=1_000_000)]
    quotas = planning.allocate_quota(100.0, records, default_catalog(),
                                     dimension="project", floor_usd=20.0)
    assert all(q["quota_usd"] >= 20.0 for q in quotas)
    assert sum(q["quota_usd"] for q in quotas) == pytest.approx(100.0, abs=1e-4)


def test_allocate_quota_floor_exceeding_total_falls_back_to_equal_split():
    records = [rec(project="a"), rec(project="b")]
    quotas = planning.allocate_quota(10.0, records, default_catalog(),
                                     dimension="project", floor_usd=100.0)
    assert all(q["quota_usd"] == pytest.approx(5.0) for q in quotas)


def test_allocate_quota_empty_records():
    assert planning.allocate_quota(100.0, [], default_catalog()) == []


def test_allocate_quota_rejects_negative_total():
    with pytest.raises(ValueError):
        planning.allocate_quota(-1.0, [rec()], default_catalog())


# -- scenario ---------------------------------------------------------------

def test_scenario_shift_reduces_projection():
    records = [rec(model="gpt-4o", inp=10_000_000, ts=f"2024-05-{d:02d}T12:00:00")
               for d in range(1, 16)]
    result = planning.scenario_shift(records, default_catalog(), "gpt-4o", -30.0,
                                     period="monthly", now=NOW)
    assert result["delta_usd"] < 0
    assert result["adjusted_total_usd"] < result["projected_total_usd"]


def test_scenario_shift_increases_projection():
    records = [rec(model="gpt-4o", inp=10_000_000, ts=f"2024-05-{d:02d}T12:00:00")
               for d in range(1, 16)]
    result = planning.scenario_shift(records, default_catalog(), "gpt-4o", 50.0,
                                     period="monthly", now=NOW)
    assert result["delta_usd"] > 0


def test_scenario_shift_unknown_model_is_noop():
    records = [rec(model="gpt-4o")]
    result = planning.scenario_shift(records, default_catalog(), "not-used", -50.0,
                                     period="monthly", now=NOW)
    assert result["delta_usd"] == 0.0
    assert result["note"]


# -- summary ----------------------------------------------------------------

def test_plan_summary_includes_budgets():
    records = [rec(inp=1_000_000, ts="2024-05-10T12:00:00")]
    budgets = [Budget(limit_usd=100.0, period="monthly", label="main")]
    summary = planning.plan_summary(records, default_catalog(), budgets,
                                    period="monthly", now=NOW)
    assert summary["overall"]["spent_usd"] > 0
    assert summary["budgets"][0]["label"] == "main"


def test_plan_summary_scopes_project_budget():
    records = [rec(project="a", inp=1_000_000), rec(project="b", inp=1_000_000)]
    budgets = [Budget(limit_usd=100.0, period="monthly", project="a")]
    summary = planning.plan_summary(records, default_catalog(), budgets,
                                    period="monthly", now=NOW)
    # The project-scoped budget must only see project a's records.
    assert summary["budgets"][0]["records"] == 1


def test_plan_summary_without_budgets():
    summary = planning.plan_summary([rec()], default_catalog(), [],
                                    period="monthly", now=NOW)
    assert summary["budgets"] == []


# -- trend note -------------------------------------------------------------

def test_trend_note_mentions_projection():
    projection = planning.project_period_end([rec()], default_catalog(),
                                             period="monthly", now=NOW)
    note = planning.trend_note(projection)
    assert "projected" in note


def test_trend_note_at_risk_mentions_cut():
    """A projection that overshoots but has not yet breached must advise a cut."""
    records = [rec(inp=10_000, ts=f"2024-05-{d:02d}T12:00:00")
               for d in range(1, 16)]
    outcome = planning.budget_outcome(records, default_catalog(), 0.5,
                                      period="monthly", now=NOW)
    assert outcome["verdict"] == "at_risk"
    note = planning.trend_note(planning.project_period_end(
        records, default_catalog(), period="monthly", now=NOW), outcome)
    assert "cut" in note


def test_trend_note_exceeded_mentions_limit():
    records = [rec(inp=100_000_000, ts="2024-05-10T12:00:00")]
    outcome = planning.budget_outcome(records, default_catalog(), 0.01,
                                      period="monthly", now=NOW)
    note = planning.trend_note(planning.project_period_end(
        records, default_catalog(), period="monthly", now=NOW), outcome)
    assert "already exceeded" in note
