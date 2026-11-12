"""Unit tests for the iteration-15 envelope review module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import envelope
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog

CATALOG = default_catalog()
# A fixed mid-month clock so pace arithmetic is deterministic.
NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
# Late in the month, where the ``watch`` band actually lives: at the halfway
# point any even-spend projection is already double what has been spent, so
# ``heading_over`` always wins and ``watch`` is structurally unreachable.
NOW_LATE = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def record(model="gpt-4o", inp=1_000_000, out=0, project="api",
           tags=None, hours_ago=1, now=None):
    base = now or NOW
    stamp = (base - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%S")
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=stamp, project=project, tags=tags or [])


def env(name="e", limit=40.0, period="monthly", scope_type="project",
        scope_value="api", note=""):
    return envelope.Envelope(name=name, limit_usd=limit, period=period,
                             scope_type=scope_type, scope_value=scope_value,
                             note=note)


# -- the dataclass -------------------------------------------------------------

@pytest.mark.parametrize("kwargs,message", [
    ({"name": ""}, "name"),
    ({"limit_usd": 0}, "limit_usd"),
    ({"period": "yearly"}, "period"),
    ({"scope_type": "user"}, "scope_type"),
    ({"scope_value": ""}, "scope_value"),
])
def test_envelope_validates_its_shape(kwargs, message):
    base = {"name": "e", "limit_usd": 10.0, "scope_type": "project",
            "scope_value": "api"}
    with pytest.raises(ValueError, match=message):
        envelope.Envelope(**{**base, **kwargs})


def test_scope_all_needs_no_value():
    assert envelope.Envelope(name="e", limit_usd=1.0, scope_type="all").scope_label() == "all"


@pytest.mark.parametrize("scope_type,value", [
    ("project", "api"), ("model", "gpt-4o"), ("tag", "nightly"),
])
def test_scope_label_names_the_scope(scope_type, value):
    assert env(scope_type=scope_type, scope_value=value).scope_label() == \
        f"{scope_type}={value}"


def test_envelope_round_trips_through_dict():
    original = env(name="api", limit=40.0, note="team budget")
    restored = envelope.Envelope.from_dict(original.to_dict())
    assert restored == original


# -- persistence helpers --------------------------------------------------------

def test_load_envelopes_skips_malformed_entries_and_accepts_a_bare_list():
    payload = {"envelopes": [
        {"name": "good", "limit_usd": 10},
        {"name": "bad", "limit_usd": -5},        # negative limit
        {"limit_usd": 10},                       # missing name
        "not a dict",
    ]}
    assert [item.name for item in envelope.load_envelopes(payload)] == ["good"]
    assert len(envelope.load_envelopes([{"name": "a", "limit_usd": 5}])) == 1


@pytest.mark.parametrize("payload", ["nonsense", {"envelopes": "nope"}])
def test_load_envelopes_tolerates_garbage(payload):
    assert envelope.load_envelopes(payload) == []


def test_dump_envelopes_wraps_in_a_named_key():
    payload = envelope.dump_envelopes([env(name="a")])
    assert list(payload) == ["envelopes"] and payload["envelopes"][0]["name"] == "a"


# -- scope matching -------------------------------------------------------------

@pytest.mark.parametrize("scope_type,value,expected", [
    ("all", "", True),
    ("project", "api", True),
    ("project", "web", False),
    ("model", "gpt-4o", True),
    ("model", "gpt-4o-mini", False),
])
def test_matches_honours_each_scope_type(scope_type, value, expected):
    item = envelope.Envelope(name="e", limit_usd=10.0, scope_type=scope_type,
                             scope_value=value)
    assert envelope.matches(item, record(project="api")) is expected


def test_matches_finds_a_tag_but_not_a_missing_one():
    item = env(scope_type="tag", scope_value="nightly")
    assert envelope.matches(item, record(tags=["nightly", "batch"])) is True
    assert envelope.matches(item, record(tags=["batch"])) is False


def test_in_scope_returns_only_governed_records():
    item = env(scope_type="project", scope_value="api")
    scoped = envelope.in_scope(item, [record(project="api"), record(project="web")])
    assert len(scoped) == 1 and scoped[0].project == "api"


# -- period windows -------------------------------------------------------------

def test_total_period_has_no_pace():
    """A limit with no reset must not be given an invented schedule."""
    row = envelope.review_envelope(env(period="total"), [record()], CATALOG, now=NOW)
    assert row["pace_usd"] is None
    assert row["projection_usd"] is None
    assert row["status"] == envelope.STATUS_OK


def test_monthly_envelope_ignores_last_months_spend():
    """Lifetime spend must not be charged to a fresh monthly envelope."""
    old = record(hours_ago=24 * 45)              # mid-September
    row = envelope.review_envelope(env(limit=40.0, period="monthly"),
                                   [old], CATALOG, now=NOW)
    assert row["status"] == envelope.STATUS_NO_ACTIVITY
    assert row["spent_usd"] == 0.0


def test_total_envelope_keeps_every_record():
    old = record(hours_ago=24 * 45)
    row = envelope.review_envelope(env(limit=40.0, period="total"),
                                   [old], CATALOG, now=NOW)
    assert row["spent_usd"] == pytest.approx(2.50)


def test_in_window_respects_period_boundaries():
    assert envelope.in_window("monthly", record(), NOW) is True
    assert envelope.in_window("monthly", record(hours_ago=24 * 45), NOW) is False
    assert envelope.in_window("total", record(hours_ago=24 * 400), NOW) is True


# -- statuses -------------------------------------------------------------------

def test_ok_when_well_inside_the_limit():
    row = envelope.review_envelope(env(limit=100.0), [record()], CATALOG, now=NOW)
    assert row["status"] == envelope.STATUS_OK
    assert row["consumed"] == pytest.approx(0.025)


def test_watch_when_most_of_the_limit_is_gone():
    """Late in the month: 83% consumed and the projection still inside the limit.

    Day 25 of 30 makes the projection 3.00 for 2.50 spent, so a $3.05 limit is
    the narrow window where ``watch`` -- not ``heading_over`` -- applies.
    """
    late = record(now=NOW_LATE)
    row = envelope.review_envelope(env(limit=3.05), [late], CATALOG, now=NOW_LATE)
    assert row["consumed"] == pytest.approx(0.8197, abs=0.001)
    assert row["projection_usd"] <= 3.05
    assert row["status"] == envelope.STATUS_WATCH


def test_over_when_the_limit_is_already_breached():
    row = envelope.review_envelope(env(limit=2.0), [record()], CATALOG, now=NOW)
    assert row["status"] == envelope.STATUS_OVER
    assert row["remaining_usd"] == 0.0


def test_heading_over_when_the_projection_breaches_the_limit():
    """Halfway through the month, half the budget spent projects to double."""
    row = envelope.review_envelope(env(limit=4.0), [record()], CATALOG, now=NOW)
    assert row["status"] == envelope.STATUS_HEADING_OVER
    assert row["projection_usd"] > 4.0


def test_no_activity_is_reported_separately_from_zero_consumption():
    """A scope typo looks exactly like perfect restraint; it must not read as ok."""
    missing = env(scope_type="project", scope_value="typo")
    row = envelope.review_envelope(missing, [record()], CATALOG, now=NOW)
    assert row["status"] == envelope.STATUS_NO_ACTIVITY
    assert "not the same as an unused budget" in row["note"]


def test_warn_at_controls_the_watch_band():
    """The same late-month row flips between watch and ok on --warn-at alone."""
    late = record(now=NOW_LATE)
    strict = envelope.review_envelope(env(limit=2.9), [late], CATALOG,
                                      now=NOW_LATE, warn_at=0.95)
    loose = envelope.review_envelope(env(limit=2.9), [late], CATALOG,
                                     now=NOW_LATE, warn_at=0.5)
    assert strict["consumed"] < 0.95 < 1.0
    assert loose["status"] != envelope.STATUS_OK


def test_pace_delta_is_negative_when_spending_is_behind_schedule():
    row = envelope.review_envelope(env(limit=1000.0), [record()], CATALOG, now=NOW)
    assert row["pace_delta_usd"] < 0


def test_pace_matches_an_even_spender():
    """$31 spent on day 15 of 30 should be about half of a $31 limit."""
    row = envelope.review_envelope(env(limit=5.0), [record()], CATALOG, now=NOW)
    assert row["pace_usd"] == pytest.approx(2.50, abs=0.20)


# -- whole-allocation review ----------------------------------------------------

def test_review_sorts_by_limit_descending():
    items = [env(name="small", limit=5.0), env(name="big", limit=500.0)]
    result = envelope.review(items, [record()], CATALOG, now=NOW)
    assert [row["envelope"] for row in result["rows"]] == ["big", "small"]


def test_review_counts_every_status():
    items = [
        env(name="ok", limit=1000.0),
        env(name="over", limit=1.0),
        env(name="dud", limit=10.0, scope_value="typo"),
    ]
    result = envelope.review(items, [record()], CATALOG, now=NOW)
    assert result["counts"]["ok"] == 1
    assert result["counts"]["over"] == 1
    assert result["counts"]["no_activity"] == 1
    assert result["status"] == "attention"


def test_review_is_clean_when_every_envelope_is_healthy():
    result = envelope.review([env(limit=1000.0)], [record()], CATALOG, now=NOW)
    assert result["status"] == "clean"


def test_review_coverage_counts_nested_scopes_once():
    """An ``all`` envelope plus a narrower one must not double-count spend."""
    items = [env(name="all", limit=100.0, scope_type="all", scope_value=""),
             env(name="api", limit=50.0)]
    result = envelope.review(items, [record()], CATALOG, now=NOW)
    assert result["governed_usd"] == pytest.approx(2.50)
    assert result["coverage"] == pytest.approx(1.0)


def test_review_reports_partial_coverage():
    items = [env(name="api", limit=50.0)]
    records = [record(), record(project="web")]
    result = envelope.review(items, records, CATALOG, now=NOW)
    assert result["coverage"] == pytest.approx(0.5)


def test_review_warns_when_nothing_is_defined():
    result = envelope.review([], [record()], CATALOG, now=NOW)
    assert any("nothing is being governed" in text for text in result["warnings"])


def test_review_warns_about_a_scope_that_matches_nothing():
    result = envelope.review([env(scope_value="typo")], [record()], CATALOG, now=NOW)
    assert any("check the scope" in text for text in result["warnings"])


def test_review_warns_when_coverage_is_low():
    items = [env(name="api", limit=50.0)]
    records = [record()] + [record(project="web") for _ in range(9)]
    result = envelope.review(items, records, CATALOG, now=NOW)
    assert any("no allocation to be reviewed against" in text
               for text in result["warnings"])


# -- overlaps -------------------------------------------------------------------

@pytest.mark.parametrize("second", [
    env(name="b", scope_type="all", scope_value=""),
    env(name="b"),
])
def test_overlaps_detects_shared_records(second):
    first = env(name="a", scope_type="all", scope_value="")
    assert len(envelope.overlaps([first, second])) == 1


def test_overlaps_explains_why_each_pair_collides():
    all_pair = envelope.overlaps([env(name="a", scope_type="all", scope_value=""),
                                  env(name="b", scope_type="all", scope_value="")])
    same_scope = envelope.overlaps([env(name="a"), env(name="b")])
    assert "every record" in all_pair[0]["why"]
    assert same_scope[0]["why"] == "same scope, different limits"


def test_overlaps_ignores_disjoint_scopes():
    items = [env(name="api", scope_value="api"),
             env(name="web", scope_value="web")]
    assert envelope.overlaps(items) == []


def test_review_warns_that_overlapping_limits_do_not_add_up():
    items = [env(name="all", scope_type="all", scope_value=""), env(name="api")]
    result = envelope.review(items, [record()], CATALOG, now=NOW)
    assert any("do not add up" in text for text in result["warnings"])


# -- next steps -----------------------------------------------------------------

def test_next_steps_only_lists_rows_needing_action():
    items = [env(name="fine", limit=1000.0), env(name="blown", limit=1.0)]
    result = envelope.review(items, [record()], CATALOG, now=NOW)
    steps = envelope.next_steps(result)
    assert [step["envelope"] for step in steps] == ["blown"]
    assert steps[0]["over_usd"] == pytest.approx(1.50)


def test_next_steps_is_empty_when_everything_is_healthy():
    result = envelope.review([env(limit=1000.0)], [record()], CATALOG, now=NOW)
    assert envelope.next_steps(result) == []


@pytest.mark.parametrize("status", [
    envelope.STATUS_OVER, envelope.STATUS_HEADING_OVER,
    envelope.STATUS_WATCH, envelope.STATUS_NO_ACTIVITY,
])
def test_every_problem_status_carries_an_action(status):
    row = {"envelope": "e", "scope": "all", "status": status,
           "spent_usd": 1.0, "limit_usd": 1.0}
    steps = envelope.next_steps({"rows": [row]})
    assert steps and steps[0]["action"]
