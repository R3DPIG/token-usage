"""Tests for the iteration-9 attribution layer (pure logic)."""

from __future__ import annotations

import pytest

from token_usage import attribution
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog


def rec(model="gpt-4o", inp=1_000_000, out=0, project="default", tags=None,
        note="", ts="2024-05-06T12:00:00"):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp=ts, project=project, tags=list(tags or []),
                       note=note)


def rule(**kwargs):
    return attribution.OwnerRule(**kwargs)


# -- rule matching ----------------------------------------------------------

@pytest.mark.parametrize("predicate,record,expected", [
    ({"project": "api"}, rec(project="api"), True),
    ({"project": "api"}, rec(project="web"), False),
    ({"model": "gpt-4o"}, rec(model="gpt-4o"), True),
    ({"tag": "batch"}, rec(tags=["batch"]), True),
    ({"tag": "batch"}, rec(tags=["live"]), False),
    ({"note_contains": "TICKET-42"}, rec(note="fixes ticket-42"), True),
    ({}, rec(project="anything"), True),
])
def test_rule_predicates(predicate, record, expected):
    assert rule(owner="a", **predicate).matches(record) is expected


def test_rule_requires_all_predicates():
    """Predicates are ANDed, not ORed -- a partial match must not claim."""
    combined = rule(owner="a", project="api", model="gpt-4o")
    assert combined.matches(rec(project="api", model="gpt-4o")) is True
    assert combined.matches(rec(project="api", model="gpt-4o-mini")) is False


def test_rule_rejects_bad_input():
    for kwargs in ({"owner": ""}, {"owner": "a", "split": {"b": 0}},
                   {"owner": "a", "project": 123}):
        with pytest.raises(ValueError):
            rule(**kwargs)


# -- splits -----------------------------------------------------------------

def test_split_normalises_weights():
    shares = dict(rule(owner="", split={"a": 1, "b": 3}).shares())
    assert shares["a"] == pytest.approx(0.25)
    assert shares["b"] == pytest.approx(0.75)


def test_plain_rule_is_a_single_share():
    assert rule(owner="solo").shares() == [("solo", 1.0)]


# -- serialisation ----------------------------------------------------------

def test_rule_round_trips():
    original = rule(owner="a", project="api", tag="batch",
                    split={"a": 1, "b": 1}, label="shared")
    restored = attribution.OwnerRule.from_dict(original.to_dict())
    assert restored.owner == "a" and restored.project == "api"
    assert restored.split == {"a": 1.0, "b": 1.0} and restored.label == "shared"


def test_rule_from_dict_rejects_unknown_field():
    """A typo in owners.json must fail loudly, not silently match nothing."""
    with pytest.raises(ValueError):
        attribution.OwnerRule.from_dict({"owner": "a", "projcet": "api"})


def test_load_rules_skips_malformed_entries():
    rules = attribution.load_rules([
        {"owner": "good", "project": "api"},
        {"owner": "bad", "projcet": "typo"},
        {"owner": ""},
    ])
    assert [r.owner for r in rules] == ["good"]


# -- attribution ------------------------------------------------------------

def test_attribute_splits_cost_across_owners():
    rules = [rule(owner="", project="api", split={"a": 1, "b": 1})]
    data = attribution.attribute([rec(project="api")], rules, default_catalog())
    assert data["coverage"] == 1.0
    assert {row["owner"] for row in data["owners"]} == {"a", "b"}
    assert data["attributed_usd"] == pytest.approx(data["total_usd"])


def test_attribute_reports_unattributed_spend():
    rules = [rule(owner="a", project="api")]
    data = attribution.attribute([rec(project="api"), rec(project="web")],
                                 rules, default_catalog())
    assert data["unattributed_records"] == 1
    assert data["unattributed_usd"] > 0
    assert 0.0 < data["coverage"] < 1.0


def test_attribute_first_match_wins():
    """Order is the contract: the earlier rule must claim the record."""
    rules = [rule(owner="first", project="api"),
             rule(owner="second", project="api")]
    data = attribution.attribute([rec(project="api")], rules, default_catalog())
    assert [row["owner"] for row in data["owners"]] == ["first"]


def test_attribute_without_rules_is_fully_unattributed():
    data = attribution.attribute([rec()], [], default_catalog())
    assert data["coverage"] == 0.0 and data["owners"] == []
    assert data["unattributed_usd"] == data["total_usd"]


def test_attribute_empty_records():
    data = attribution.attribute([], [rule(owner="a")], default_catalog())
    assert data["total_usd"] == 0.0 and data["coverage"] == 0.0


# -- orphans and suggestions ------------------------------------------------

def test_unattributed_records_sorted_by_cost():
    rules = [rule(owner="a", project="api")]
    records = [rec(project="web", inp=100), rec(project="web", inp=1_000_000)]
    data = attribution.unattributed_records(records, rules, default_catalog())
    assert data["count"] == 2
    assert data["records"][0]["cost_usd"] > data["records"][1]["cost_usd"]


def test_suggest_rules_groups_by_project():
    rules = [rule(owner="a", project="api")]
    records = [rec(project="web"), rec(project="web"), rec(project="api")]
    items = attribution.suggest_rules(records, rules, default_catalog())
    assert len(items) == 1
    assert items[0]["project"] == "web" and items[0]["records"] == 2
    assert items[0]["proposed_rule"]["project"] == "web"


def test_suggest_rules_respects_min_cost():
    items = attribution.suggest_rules([rec(inp=10)], [], default_catalog(),
                                      min_cost_usd=100.0)
    assert items == []


# -- invoice ----------------------------------------------------------------

def test_invoice_breaks_down_by_project_and_model():
    rules = [rule(owner="team", project="api")]
    records = [rec(project="api", model="gpt-4o"),
               rec(project="api", model="gpt-4o-mini")]
    data = attribution.invoice(records, rules, default_catalog(), "team")
    assert data["found"] is True and data["records"] == 2
    assert {row["model"] for row in data["by_model"]} == {"gpt-4o", "gpt-4o-mini"}
    assert data["by_project"][0]["project"] == "api"


def test_invoice_unknown_owner():
    data = attribution.invoice([rec()], [rule(owner="a")],
                               default_catalog(), "nobody")
    assert data["found"] is False and data["cost_usd"] == 0.0


def test_invoice_honours_split_weight():
    split = attribution.invoice([rec(project="api")],
                                [rule(owner="", project="api",
                                      split={"a": 1, "b": 3})],
                                default_catalog(), "b")
    full = attribution.invoice([rec(project="api")],
                               [rule(owner="b", project="api")],
                               default_catalog(), "b")
    assert split["cost_usd"] == pytest.approx(full["cost_usd"] * 0.75)


# -- reconciliation ---------------------------------------------------------

@pytest.mark.parametrize("rules,records", [
    ([rule(owner="", project="api", split={"a": 1, "b": 2})], [rec(project="api")]),
    ([rule(owner="a", project="api")], [rec(project="api"), rec(project="web")]),
    ([], []),
])
def test_reconcile_balances(rules, records):
    """Splits, orphans and empty input must all balance to the true total."""
    data = attribution.reconcile(records, rules, default_catalog())
    assert data["balanced"] is True and data["drift_usd"] == 0.0


# -- summary ----------------------------------------------------------------

def test_summarize_shape():
    rules = [rule(owner="a", project="api")]
    data = attribution.summarize([rec(project="api"), rec(project="web")],
                                 rules, default_catalog())
    for key in ("owners", "total_usd", "coverage", "reconcile", "orphans",
                "suggestions"):
        assert key in data
    assert data["reconcile"]["balanced"] is True
