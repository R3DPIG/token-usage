"""Tests for the iteration-5 alert rule engine (pure logic)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from token_usage import alerts
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog


def rec(model="gpt-4o", inp=1_000_000, out=0, ts="2024-05-01T12:00:00",
        cached=0, project="demo", tags=None):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, cached_input_tokens=cached, project=project,
                       tags=tags or [])


# -- rule validation --------------------------------------------------------

def test_rule_rejects_unknown_metric():
    with pytest.raises(alerts.AlertError):
        alerts.Rule(name="x", metric="nonsense", operator="gt", threshold=1)


def test_rule_rejects_unknown_operator():
    with pytest.raises(alerts.AlertError):
        alerts.Rule(name="x", metric="cost", operator="equals", threshold=1)


def test_rule_rejects_unknown_dimension_and_severity():
    with pytest.raises(alerts.AlertError):
        alerts.Rule(name="x", metric="cost", operator="gt", threshold=1, dimension="hour")
    with pytest.raises(alerts.AlertError):
        alerts.Rule(name="x", metric="cost", operator="gt", threshold=1, severity="fatal")


def test_rule_rejects_empty_name():
    with pytest.raises(alerts.AlertError):
        alerts.Rule(name="", metric="cost", operator="gt", threshold=1)


def test_rule_roundtrip():
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=5.0,
                       dimension="day", scope={"project": "p"}, severity="critical")
    assert alerts.Rule.from_dict(rule.to_dict()) == rule


def test_rule_render():
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=5.0,
                       dimension="day", scope={"project": "p"})
    rendered = rule.render()
    assert "cost gt 5" in rendered and "day" in rendered and "project=p" in rendered


# -- metric evaluation ------------------------------------------------------

def test_cost_metric_fires():
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=0.001)
    results = alerts.evaluate_rule(rule, [rec()], default_catalog())
    assert results[0]["status"] == "firing"


def test_cost_metric_ok_when_below():
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=9999)
    assert alerts.evaluate_rule(rule, [rec()], default_catalog())[0]["status"] == "ok"


def test_lt_operator():
    rule = alerts.Rule(name="r", metric="records", operator="lt", threshold=5)
    assert alerts.evaluate_rule(rule, [rec()], default_catalog())[0]["status"] == "firing"


def test_cache_ratio_metric():
    rule = alerts.Rule(name="r", metric="cache_ratio", operator="lt", threshold=0.5)
    records = [rec(inp=1_000_000, cached=100_000)]  # 10% cached
    assert alerts.evaluate_rule(rule, records, default_catalog())[0]["status"] == "firing"


def test_cache_ratio_undefined_without_input():
    rule = alerts.Rule(name="r", metric="cache_ratio", operator="lt", threshold=0.5)
    records = [rec(inp=0, out=100)]
    assert alerts.evaluate_rule(rule, records, default_catalog())[0]["status"] == "unknown"


def test_output_ratio_metric():
    rule = alerts.Rule(name="r", metric="output_ratio", operator="lt", threshold=0.1)
    records = [rec(inp=1_000_000, out=0)]  # 0% output
    assert alerts.evaluate_rule(rule, records, default_catalog())[0]["status"] == "firing"


def test_cost_per_1k_metric():
    rule = alerts.Rule(name="r", metric="cost_per_1k", operator="gt", threshold=0.001)
    assert alerts.evaluate_rule(rule, [rec()], default_catalog())[0]["status"] == "firing"


def test_spend_change_pct_metric():
    """``spend_change_pct`` compares the current month against the previous one.

    ``compare_periods`` uses equal-length windows, so with ``now`` mid-May the
    previous window is mid-April to May 1st -- the fixture dates must fall
    inside those windows.
    """
    now = datetime(2024, 5, 15, tzinfo=timezone.utc)
    rule = alerts.Rule(name="r", metric="spend_change_pct", operator="gt", threshold=10)
    records = [rec(ts="2024-05-10T00:00:00", inp=10_000_000),   # current window
               rec(ts="2024-04-20T00:00:00", inp=1_000_000)]    # previous window
    results = alerts.evaluate_rule(rule, records, default_catalog(), now=now)
    assert results[0]["status"] == "firing"


def test_spend_change_pct_ok_when_flat():
    now = datetime(2024, 5, 15, tzinfo=timezone.utc)
    rule = alerts.Rule(name="r", metric="spend_change_pct", operator="gt", threshold=10)
    records = [rec(ts="2024-05-10T00:00:00", inp=1_000_000),
               rec(ts="2024-04-20T00:00:00", inp=1_000_000)]
    results = alerts.evaluate_rule(rule, records, default_catalog(), now=now)
    assert results[0]["status"] == "ok"


# -- no-data handling -------------------------------------------------------

def test_empty_scope_reports_unknown_not_ok():
    """A data gap must never look like a passing check."""
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=1)
    results = alerts.evaluate_rule(rule, [], default_catalog())
    assert results[0]["status"] == "unknown"


def test_scope_filter_can_empty_the_set():
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=1,
                       scope={"project": "nope"})
    results = alerts.evaluate_rule(rule, [rec(project="demo")], default_catalog())
    assert results[0]["status"] == "unknown"


# -- dimensions -------------------------------------------------------------

def test_dimension_model_checks_each_bucket():
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=0.001,
                       dimension="model")
    records = [rec(model="gpt-4o"), rec(model="gpt-4o-mini")]
    results = alerts.evaluate_rule(rule, records, default_catalog())
    assert len(results) == 2
    assert {r["bucket"] for r in results} == {"gpt-4o", "gpt-4o-mini"}


def test_dimension_day_buckets_by_date():
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=0.001,
                       dimension="day")
    records = [rec(ts="2024-05-01T00:00:00"), rec(ts="2024-05-02T00:00:00")]
    results = alerts.evaluate_rule(rule, records, default_catalog())
    assert {r["bucket"] for r in results} == {"2024-05-01", "2024-05-02"}


def test_disabled_rule_is_skipped():
    rule = alerts.Rule(name="r", metric="cost", operator="gt", threshold=0.001,
                       enabled=False)
    assert alerts.evaluate_rule(rule, [rec()], default_catalog())[0]["status"] == "disabled"


# -- aggregation ------------------------------------------------------------

def test_evaluate_all_orders_by_severity():
    rules = [alerts.Rule(name="info-rule", metric="cost", operator="gt",
                         threshold=0.001, severity="info"),
             alerts.Rule(name="crit-rule", metric="cost", operator="gt",
                         threshold=0.001, severity="critical")]
    results = alerts.evaluate_all(rules, [rec()], default_catalog())
    assert results[0]["severity"] == "critical"


def test_summarize_counts_statuses():
    rules = [alerts.Rule(name="a", metric="cost", operator="gt", threshold=0.001),
             alerts.Rule(name="b", metric="cost", operator="gt", threshold=9999)]
    counts = alerts.summarize(alerts.evaluate_all(rules, [rec()], default_catalog()))
    assert counts == {"firing": 1, "ok": 1}


def test_worst_severity():
    results = [{"status": "firing", "severity": "info"},
               {"status": "firing", "severity": "critical"},
               {"status": "ok", "severity": "critical"}]
    assert alerts.worst_severity(results) == "critical"
    assert alerts.worst_severity([{"status": "ok", "severity": "critical"}]) is None


# -- exit codes -------------------------------------------------------------

def test_exit_code_for_fail_on_critical():
    firing_crit = [{"status": "firing", "severity": "critical"}]
    firing_info = [{"status": "firing", "severity": "info"}]
    assert alerts.exit_code_for(firing_crit, fail_on="critical") == 1
    assert alerts.exit_code_for(firing_info, fail_on="critical") == 0
    assert alerts.exit_code_for(firing_info, fail_on="info") == 1


def test_exit_code_never_fails():
    assert alerts.exit_code_for([{"status": "firing", "severity": "critical"}],
                                fail_on="never") == 0


def test_exit_code_rejects_bad_fail_on():
    with pytest.raises(alerts.AlertError):
        alerts.exit_code_for([], fail_on="sometimes")


# -- rule set helpers -------------------------------------------------------

def test_load_rules_skips_malformed():
    rules = alerts.load_rules([
        {"name": "ok", "metric": "cost", "operator": "gt", "threshold": 1},
        {"name": "bad", "metric": "nonsense", "operator": "gt", "threshold": 1},
        "not-a-dict",
    ])
    assert [r.name for r in rules] == ["ok"]


def test_load_rules_handles_non_list():
    assert alerts.load_rules({"nope": 1}) == []


def test_upsert_replaces_same_name():
    rules = [alerts.Rule(name="a", metric="cost", operator="gt", threshold=1)]
    updated = alerts.upsert_rule(rules, alerts.Rule(name="a", metric="cost",
                                                    operator="gt", threshold=99))
    assert len(updated) == 1 and updated[0].threshold == 99


def test_remove_rule():
    rules = [alerts.Rule(name="a", metric="cost", operator="gt", threshold=1)]
    remaining, removed = alerts.remove_rule(rules, "a")
    assert remaining == [] and removed.name == "a"
    remaining, removed = alerts.remove_rule(rules, "missing")
    assert removed is None and len(remaining) == 1


def test_default_rules_are_valid():
    rules = alerts.default_rules()
    assert len(rules) == 3
    for rule in rules:
        assert rule.metric in alerts.METRICS
