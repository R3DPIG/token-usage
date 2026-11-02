"""Tests for the iteration-7 temporal pattern layer (pure logic)."""

from __future__ import annotations

import pytest

from token_usage import patterns
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog


def rec(inp=1_000_000, out=0, ts="2024-05-06T12:00:00", model="gpt-4o"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts)


def daily_map(**costs):
    """Build a ``by_day``-shaped dict from ``day=cost`` kwargs."""
    return {day.replace("_", "-"): {"cost_usd": cost, "records": 1}
            for day, cost in costs.items()}


# -- profiles ---------------------------------------------------------------

def test_hourly_profile_always_24_buckets():
    profile = patterns.hourly_profile([rec(ts="2024-05-06T14:00:00")],
                                      default_catalog())
    assert len(profile) == 24
    assert profile[14]["records"] == 1
    assert profile[0]["records"] == 0


def test_hourly_profile_zero_filled_and_accumulates():
    empty = patterns.hourly_profile([], default_catalog())
    assert len(empty) == 24 and all(b["cost_usd"] == 0.0 for b in empty)
    assert empty[0]["label"] == "00:00"

    profile = patterns.hourly_profile(
        [rec(ts="2024-05-06T09:00:00"), rec(ts="2024-05-07T09:30:00")],
        default_catalog())
    assert profile[9]["records"] == 2


def test_weekday_profile_always_7_buckets_monday_first():
    profile = patterns.weekday_profile([rec()], default_catalog())
    assert len(profile) == 7
    assert [b["label"] for b in profile] == list(patterns.WEEKDAY_NAMES)


@pytest.mark.parametrize("timestamp,index,label", [
    ("2024-05-06T12:00:00", 0, "Mon"),   # Monday
    ("2024-05-12T12:00:00", 6, "Sun"),   # Sunday
])
def test_weekday_profile_places_days(timestamp, index, label):
    profile = patterns.weekday_profile([rec(ts=timestamp)], default_catalog())
    assert profile[index]["records"] == 1
    assert profile[index]["label"] == label


# -- peak / quiet -----------------------------------------------------------

def test_peak_buckets_ignores_empty_and_orders_by_cost():
    profile = patterns.weekday_profile(
        [rec(inp=1_000_000, ts="2024-05-06T12:00:00"),
         rec(inp=5_000_000, ts="2024-05-07T12:00:00")],
        default_catalog())
    peaks = patterns.peak_buckets(profile, top=5)
    assert len(peaks) == 2                      # only populated buckets
    assert peaks[0]["label"] == "Tue"           # the more expensive day


def test_quiet_buckets_excludes_empty():
    profile = patterns.weekday_profile([rec()], default_catalog())
    assert len(patterns.quiet_buckets(profile, top=3)) == 1


# -- weekend split ----------------------------------------------------------

def test_weekend_split_separates_days():
    records = [rec(ts="2024-05-06T12:00:00"),    # Monday
               rec(ts="2024-05-11T12:00:00")]    # Saturday
    split = patterns.weekend_split(records, default_catalog())
    assert split["weekday_days"] == 1 and split["weekend_days"] == 1
    assert split["weekday_records"] == 1 and split["weekend_records"] == 1


def test_weekend_split_normalises_per_day():
    """Raw totals would always favour weekdays; per-day averages must not."""
    records = [rec(inp=400_000, ts="2024-05-06T12:00:00"),
               rec(inp=400_000, ts="2024-05-07T12:00:00"),
               rec(inp=1_200_000, ts="2024-05-11T12:00:00")]
    split = patterns.weekend_split(records, default_catalog())
    assert split["weekend_per_day_usd"] > split["weekday_per_day_usd"]
    assert split["weekend_ratio"] > 1.0


def test_weekend_split_empty():
    split = patterns.weekend_split([], default_catalog())
    assert split["weekday_days"] == 0 and split["weekend_days"] == 0
    assert split["weekend_ratio"] == 0.0


# -- moving average ---------------------------------------------------------

def test_moving_average_keeps_full_length_and_trails():
    series = patterns.moving_average(daily_map(d1=1.0, d2=2.0, d3=3.0), window=2)
    assert len(series) == 3
    assert series[0]["average_usd"] == pytest.approx(1.0)
    assert series[1]["average_usd"] == pytest.approx(1.5)
    assert series[2]["average_usd"] == pytest.approx(2.5)


def test_moving_average_partial_window_at_start():
    series = patterns.moving_average(daily_map(d1=2.0, d2=4.0), window=7)
    assert series[0]["samples"] == 1 and series[1]["samples"] == 2


def test_moving_average_rejects_bad_window():
    with pytest.raises(ValueError):
        patterns.moving_average(daily_map(d1=1.0), window=0)


# -- trend ------------------------------------------------------------------

@pytest.mark.parametrize("costs,expected", [
    (dict(d1=1.0, d2=2.0, d3=3.0, d4=4.0, d5=5.0), "rising"),
    (dict(d1=5.0, d2=4.0, d3=3.0, d4=2.0, d5=1.0), "falling"),
    (dict(d1=2.0, d2=2.0, d3=2.0), "stable"),
    (dict(d1=100.0, d2=100.1, d3=100.2), "stable"),   # drift under 1%/day
])
def test_trend_direction(costs, expected):
    assert patterns.trend_direction(daily_map(**costs))["direction"] == expected


def test_trend_perfect_fit_has_r_squared_one():
    result = patterns.trend_direction(daily_map(d1=1.0, d2=2.0, d3=3.0, d4=4.0))
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["slope_usd_per_day"] > 0


def test_trend_r_squared_low_for_noise():
    daily = daily_map(d1=1.0, d2=9.0, d3=2.0, d4=8.0, d5=3.0)
    assert patterns.trend_direction(daily)["r_squared"] < 0.5


def test_trend_insufficient_and_empty():
    assert patterns.trend_direction(daily_map(d1=5.0))["samples"] == 1
    empty = patterns.trend_direction({})
    assert empty["samples"] == 0 and empty["direction"] == "stable"


def test_trend_window_limits_samples():
    daily = daily_map(d1=1.0, d2=2.0, d3=3.0, d4=4.0, d5=5.0)
    assert patterns.trend_direction(daily, window=3)["samples"] == 3


# -- streaks ----------------------------------------------------------------

def test_streaks_longest_above():
    result = patterns.streaks(daily_map(d1=1.0, d2=10.0, d3=10.0, d4=10.0, d5=1.0))
    assert result["longest_above"] == 3


def test_streaks_current_run():
    result = patterns.streaks(daily_map(d1=1.0, d2=1.0, d3=10.0, d4=10.0))
    assert result["current_direction"] == "above"
    assert result["current_above"] == 2


def test_streaks_empty_and_all_equal():
    assert patterns.streaks({})["current_direction"] == "none"
    flat = patterns.streaks(daily_map(d1=5.0, d2=5.0, d3=5.0))
    assert flat["longest_above"] == 0 and flat["longest_below"] == 0


# -- burstiness -------------------------------------------------------------

@pytest.mark.parametrize("costs,expected", [
    (dict(d1=10.0, d2=10.0, d3=10.0, d4=10.0), "steady"),
    (dict(d1=1.0, d2=1.0, d3=1.0, d4=100.0), "bursty"),
    (dict(d1=5.0), "insufficient-data"),
])
def test_burstiness_classification(costs, expected):
    assert patterns.burstiness(daily_map(**costs))["classification"] == expected


def test_burstiness_steady_has_zero_cv():
    assert patterns.burstiness(daily_map(d1=10.0, d2=10.0))["cv"] == 0.0


def test_burstiness_empty():
    assert patterns.burstiness({})["samples"] == 0


# -- seasonality ------------------------------------------------------------

def test_seasonality_strong():
    profile = patterns.weekday_profile(
        [rec(inp=10_000_000, ts="2024-05-06T12:00:00"),
         rec(inp=1_000_000, ts="2024-05-07T12:00:00")],
        default_catalog())
    result = patterns.seasonality_strength(profile)
    assert result["strength"] == "strong"
    assert result["strongest"] == "Mon"


def test_seasonality_weak_when_uniform():
    profile = patterns.weekday_profile(
        [rec(inp=1_000_000, ts="2024-05-06T12:00:00"),
         rec(inp=1_000_000, ts="2024-05-07T12:00:00")],
        default_catalog())
    assert patterns.seasonality_strength(profile)["strength"] == "weak"


def test_seasonality_insufficient():
    profile = patterns.weekday_profile([rec()], default_catalog())
    assert patterns.seasonality_strength(profile)["strength"] == "insufficient-data"


# -- summarize --------------------------------------------------------------

def test_summarize_shape():
    records = [rec(ts=f"2024-05-{day:02d}T12:00:00") for day in range(1, 9)]
    data = patterns.summarize(records, default_catalog())
    for key in ("days", "weekday_profile", "hourly_profile", "weekend",
                "trend", "streaks", "burstiness", "seasonality",
                "peak_weekdays", "peak_hours"):
        assert key in data
    assert data["days"] == 8
    assert len(data["hourly_profile"]) == 24
    assert len(data["weekday_profile"]) == 7


def test_summarize_empty():
    data = patterns.summarize([], default_catalog())
    assert data["days"] == 0 and data["peak_weekdays"] == []
