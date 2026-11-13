"""Unit tests for the iteration-16 forecast backtesting module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import backtest
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog

CATALOG = default_catalog()
NOW = datetime(2026, 11, 30, 12, 0, tzinfo=timezone.utc)
# gpt-4o input costs $2.50/1M, so 400k input tokens is exactly $1.00.
TOKENS_PER_DOLLAR = 400_000


def day(index: int, usd: float, model="gpt-4o") -> UsageRecord:
    """One record dated ``index`` days before NOW, costing ``usd``."""
    stamp = (NOW - timedelta(days=index)).strftime("%Y-%m-%dT%H:%M:%S")
    return UsageRecord(model=model, input_tokens=int(usd * TOKENS_PER_DOLLAR),
                       output_tokens=0, timestamp=stamp, project="api")


def series(*usd_per_day: float):
    """Build records from a list where index 0 is the oldest day."""
    count = len(usd_per_day)
    return [day(count - 1 - index, usd) for index, usd in enumerate(usd_per_day)]


# -- daily series ---------------------------------------------------------------

def test_daily_series_keeps_only_days_that_exist():
    records = [day(3, 1.0), day(1, 2.0)]          # a gap on day 2
    rows = backtest._daily_series(records, CATALOG)
    assert len(rows) == 2
    assert [row["cost_usd"] for row in rows] == [1.0, 2.0]


def test_daily_series_is_chronological():
    rows = backtest._daily_series([day(5, 1.0), day(1, 3.0)], CATALOG)
    assert rows[0]["day"] < rows[1]["day"]


def test_fit_needs_at_least_two_days():
    assert backtest._fit([{"day": "2026-11-01", "cost_usd": 1.0}]) is None


def test_fit_reuses_the_shipped_predictor():
    """The backtest must measure the product's forecast, not a private copy."""
    rows = [{"day": "2026-11-01", "cost_usd": 1.0},
            {"day": "2026-11-02", "cost_usd": 2.0},
            {"day": "2026-11-03", "cost_usd": 3.0}]
    assert backtest._fit(rows)["projected_daily"] == pytest.approx(4.0)


# -- input validation -----------------------------------------------------------

@pytest.mark.parametrize("kwargs", [{"window_days": 0}, {"min_history_days": 1}])
def test_backtest_rejects_impossible_arguments(kwargs):
    with pytest.raises(ValueError):
        backtest.backtest([], CATALOG, **kwargs)


def test_backtest_reports_insufficient_history():
    result = backtest.backtest(series(1.0, 2.0), CATALOG, window_days=1)
    assert result["samples"] == [] and result["metrics"] is None
    assert "need at least" in result["reason"]
    assert result["reliable"] is False


# -- no look-ahead ---------------------------------------------------------------

def test_the_first_prediction_uses_at_least_the_minimum_history():
    """The earliest cutoff must sit after the minimum history, never before it."""
    records = series(*[1.0] * 10)
    earliest_day = min(record.timestamp[:10] for record in records)
    result = backtest.backtest(records, CATALOG, window_days=1, min_history_days=3)
    first_cutoff = result["samples"][0]["cutoff"]
    # Three days of history means the first cutoff is the third calendar day.
    assert first_cutoff > earliest_day
    assert len(result["samples"]) >= 1


def test_predictions_never_see_the_days_they_score():
    """A flat series must predict flat; peeking would only add noise."""
    result = backtest.backtest(series(*[1.0] * 10), CATALOG, window_days=1)
    for row in result["samples"]:
        assert row["predicted_usd"] == pytest.approx(1.0, abs=0.01)
        assert row["actual_usd"] == pytest.approx(1.0)


def test_a_clean_trend_is_predicted_accurately():
    """A perfectly linear series should show almost no error."""
    result = backtest.backtest(series(*[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                               CATALOG, window_days=1)
    assert result["metrics"]["mape"] < 15.0
    assert backtest.accuracy_grade(result["metrics"]) in ("close", "usable")


def test_a_future_spike_never_leaks_into_an_earlier_prediction():
    """Changing data after a cutoff must not move that cutoff's prediction.

    This is the whole no-look-ahead guarantee: if appending a huge future day
    could change a prediction made before it, the model was peeking.
    """
    base = series(*[1.0] * 8)
    baseline = backtest.backtest(base, CATALOG, window_days=1)
    # Append a much later, very expensive day. Earlier cutoffs must be unmoved.
    spiked = base + [day(-5, 500.0)]
    after = backtest.backtest(spiked, CATALOG, window_days=1)
    first_cutoff = baseline["samples"][0]["cutoff"]
    matching = [row for row in after["samples"] if row["cutoff"] == first_cutoff]
    assert matching, "the earlier cutoff must still be graded"
    assert matching[0]["predicted_usd"] == baseline["samples"][0]["predicted_usd"]


def test_partial_windows_are_dropped_not_scored():
    """The final days have no full forward window, so they cannot be graded."""
    result = backtest.backtest(series(*[1.0] * 6), CATALOG, window_days=3,
                               min_history_days=2)
    for row in result["samples"]:
        assert row["actual_usd"] > 0


def test_widening_the_window_reduces_the_sample_count():
    narrow = backtest.backtest(series(*[1.0] * 12), CATALOG, window_days=1)
    wide = backtest.backtest(series(*[1.0] * 12), CATALOG, window_days=4)
    assert len(wide["samples"]) < len(narrow["samples"])


# -- metrics --------------------------------------------------------------------

def test_metrics_is_none_without_predictions():
    assert backtest.metrics([]) is None


def test_metrics_separates_bias_from_random_error():
    """Always-high predictions are bias; alternating misses are noise."""
    biased = backtest.metrics([
        {"error_usd": 2.0, "abs_error_usd": 2.0, "error_pct": 20.0, "actual_usd": 10.0},
        {"error_usd": 2.0, "abs_error_usd": 2.0, "error_pct": 20.0, "actual_usd": 10.0},
    ])
    noisy = backtest.metrics([
        {"error_usd": 2.0, "abs_error_usd": 2.0, "error_pct": 20.0, "actual_usd": 10.0},
        {"error_usd": -2.0, "abs_error_usd": 2.0, "error_pct": -20.0, "actual_usd": 10.0},
    ])
    assert biased["bias_pct"] == pytest.approx(20.0)
    assert noisy["bias_pct"] == pytest.approx(0.0)
    assert biased["mape"] == noisy["mape"] == pytest.approx(20.0)


def test_metrics_counts_sign_in_bias_and_ignores_it_in_mae():
    result = backtest.metrics([
        {"error_usd": -3.0, "abs_error_usd": 3.0, "error_pct": -30.0, "actual_usd": 10.0},
        {"error_usd": -1.0, "abs_error_usd": 1.0, "error_pct": -10.0, "actual_usd": 10.0},
    ])
    assert result["bias_usd"] == pytest.approx(-2.0)
    assert result["mae_usd"] == pytest.approx(2.0)
    assert result["worst_abs_usd"] == pytest.approx(3.0)


def test_metrics_reports_coverage_when_some_actuals_are_zero():
    result = backtest.metrics([
        {"error_usd": 1.0, "abs_error_usd": 1.0, "error_pct": None, "actual_usd": 0.0},
        {"error_usd": 1.0, "abs_error_usd": 1.0, "error_pct": 10.0, "actual_usd": 10.0},
    ])
    assert result["coverage"] == pytest.approx(0.5)
    assert result["mape"] == pytest.approx(10.0)   # computed over gradable rows only


def test_metrics_with_only_zero_actuals_has_no_percentage():
    result = backtest.metrics([
        {"error_usd": 1.0, "abs_error_usd": 1.0, "error_pct": None, "actual_usd": 0.0},
    ])
    assert result["bias_pct"] is None and result["mape"] is None
    assert result["mae_usd"] == pytest.approx(1.0)   # dollars still work


# -- grading --------------------------------------------------------------------

@pytest.mark.parametrize("mape,grade", [
    (5.0, "close"), (10.0, "close"),
    (11.0, "usable"), (25.0, "usable"),
    (26.0, "loose"), (50.0, "loose"),
    (51.0, "unreliable"),
])
def test_accuracy_grade_thresholds(mape, grade):
    assert backtest.accuracy_grade({"mape": mape}) == grade


@pytest.mark.parametrize("metrics_result", [None, {}, {"mape": None}])
def test_accuracy_grade_refuses_to_grade_without_a_percentage(metrics_result):
    assert backtest.accuracy_grade(metrics_result) == "not evaluated"


# -- bias characterisation ------------------------------------------------------

def test_bias_note_detects_a_correctable_forecast():
    """Directional error with a tight spread is bias, and bias is fixable."""
    note = backtest.bias_note({"bias_pct": 15.0, "mape": 18.0})
    assert note["kind"] == "bias"
    assert "consistently high" in note["note"]
    assert "correction factor" in note["note"]


def test_bias_note_detects_noise():
    note = backtest.bias_note({"bias_pct": 2.0, "mape": 40.0})
    assert note["kind"] == "noise"
    assert "cancel out" in note["note"]


def test_bias_note_detects_both_faults():
    note = backtest.bias_note({"bias_pct": 35.0, "mape": 60.0})
    assert note["kind"] == "both"
    assert "will not be enough" in note["note"]


def test_bias_note_clears_a_healthy_forecast():
    note = backtest.bias_note({"bias_pct": 3.0, "mape": 8.0})
    assert note["kind"] == "clean"


def test_bias_note_has_no_opinion_without_percentages():
    note = backtest.bias_note({"bias_pct": None, "mape": None})
    assert note["kind"] == "unknown" and "cannot be told apart" in note["note"]


# -- worst cases and summary ----------------------------------------------------

def test_worst_cases_are_ordered_by_absolute_miss():
    result = backtest.backtest(series(*[1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0]),
                               CATALOG, window_days=1)
    cases = backtest.worst_cases(result, limit=2)
    assert len(cases) == 2
    assert cases[0]["abs_error_usd"] >= cases[1]["abs_error_usd"]


def test_worst_cases_are_empty_without_samples():
    assert backtest.worst_cases({"samples": []}) == []


def test_summarize_warns_when_the_sample_is_thin():
    result = backtest.summarize(series(*[1.0] * 5), CATALOG, window_days=2)
    assert any("sanity check, not a verdict" in text for text in result["warnings"])


def test_summarize_warns_when_nothing_can_be_graded():
    result = backtest.summarize(series(1.0, 2.0), CATALOG, window_days=1)
    assert result["grade"] == "not evaluated"
    assert result["warnings"]


def test_summarize_attaches_grade_and_bias_character():
    result = backtest.summarize(series(*[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                                CATALOG, window_days=1)
    assert result["grade"] in ("close", "usable", "loose")
    assert result["bias"]["kind"] in ("clean", "bias", "noise", "both")


def test_summarize_marks_a_long_run_reliable():
    result = backtest.summarize(series(*[1.0] * 12), CATALOG, window_days=1)
    assert result["reliable"] is True
    assert len(result["samples"]) >= backtest.MIN_SAMPLES
