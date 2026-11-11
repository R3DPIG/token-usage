"""Unit tests for the iteration-14 bill reconciliation module."""

from __future__ import annotations

import json

import pytest

from token_usage import reconcile
from token_usage.models import UsageRecord
from token_usage.pricing import default_catalog

CATALOG = default_catalog()
# gpt-4o prices: $2.50/1M input, $10.00/1M output -> 1M input = $2.50.


def record(model="gpt-4o", inp=1_000_000, out=0, cost=None):
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                       timestamp="2026-11-01T00:00:00", cost_usd=cost)


# -- parsing -----------------------------------------------------------------

def test_parse_bill_rows_accepts_generic_rows():
    lines, errors = reconcile.parse_bill_rows([
        {"model": "gpt-4o", "amount_usd": "2.50", "input_tokens": "1,000,000",
         "output_tokens": "0", "period": "2026-10"},
        {"model": "gpt-4o-mini", "cost_usd": 0.15},
    ])
    assert not errors
    assert [line.model for line in lines] == ["gpt-4o", "gpt-4o-mini"]
    assert lines[0].input_tokens == 1_000_000
    assert lines[0].period == "2026-10"
    assert lines[1].amount_usd == pytest.approx(0.15)


def test_parse_bill_rows_reports_bad_rows_without_losing_good_ones():
    lines, errors = reconcile.parse_bill_rows([
        {"model": "gpt-4o", "amount_usd": "2.50"},
        {"amount_usd": "1.00"},                       # no model
        {"model": "gpt-4o", "amount_usd": "n/a"},     # no usable amount
        "not even a dict",
    ])
    assert len(lines) == 1 and lines[0].model == "gpt-4o"
    assert len(errors) == 3


def test_parse_bill_rows_unwraps_a_dict_payload():
    lines, errors = reconcile.parse_bill_rows(
        {"data": [{"model": "gpt-4o", "amount_usd": 2.50}]})
    assert not errors and len(lines) == 1


def test_coerce_amount_strips_currency_noise():
    assert reconcile._coerce_amount("$1,234.50") == pytest.approx(1234.50)
    assert reconcile._coerce_amount(" 12 ") == pytest.approx(12.0)
    assert reconcile._coerce_amount("") is None
    assert reconcile._coerce_amount("free") is None


def test_bill_line_token_counts_are_optional():
    with_tokens = reconcile.BillLine(model="m", amount_usd=1.0,
                                     input_tokens=10, output_tokens=5)
    without = reconcile.BillLine(model="m", amount_usd=1.0)
    assert with_tokens.total_tokens() == 15
    assert without.total_tokens() is None


@pytest.mark.parametrize("suffix,writer", [
    (".csv", lambda p: p.write_text(
        "model,amount_usd,input_tokens,output_tokens\n"
        "gpt-4o,2.50,1000000,0\n", encoding="utf-8")),
    (".json", lambda p: p.write_text(
        json.dumps([{"model": "gpt-4o", "amount_usd": 2.50,
                     "input_tokens": 1_000_000, "output_tokens": 0}]),
        encoding="utf-8")),
])
def test_load_bill_reads_csv_and_json(tmp_path, suffix, writer):
    path = tmp_path / f"bill{suffix}"
    writer(path)
    lines, errors = reconcile.load_bill(path)
    assert not errors and len(lines) == 1
    assert lines[0].amount_usd == pytest.approx(2.50)
    assert lines[0].input_tokens == 1_000_000


# -- aggregation ---------------------------------------------------------------

def test_billed_totals_aggregates_per_model_and_counts_token_lines():
    lines = [
        reconcile.BillLine(model="gpt-4o", amount_usd=2.0,
                           input_tokens=800_000, output_tokens=0),
        reconcile.BillLine(model="gpt-4o", amount_usd=1.0),  # no tokens
        reconcile.BillLine(model="gpt-4o-mini", amount_usd=0.5),
    ]
    totals = reconcile.billed_totals(lines)
    assert totals["gpt-4o"]["amount_usd"] == pytest.approx(3.0)
    assert totals["gpt-4o"]["tokens"] == 800_000
    assert totals["gpt-4o"]["token_lines"] == 1 and totals["gpt-4o"]["lines"] == 2
    assert totals["gpt-4o-mini"]["token_lines"] == 0


def test_local_totals_prices_records_through_the_catalog():
    totals = reconcile.local_totals([record()], CATALOG)
    assert totals["gpt-4o"]["amount_usd"] == pytest.approx(2.50)
    assert totals["gpt-4o"]["tokens"] == 1_000_000
    assert totals["gpt-4o"]["records"] == 1


def test_local_totals_honours_cost_overrides():
    totals = reconcile.local_totals([record(cost=9.99)], CATALOG)
    assert totals["gpt-4o"]["amount_usd"] == pytest.approx(9.99)


# -- comparison ----------------------------------------------------------------

def bill(model="gpt-4o", amount=2.50, **tokens):
    return reconcile.BillLine(model=model, amount_usd=amount, **tokens)


def test_compare_reports_clean_when_rounding_is_within_tolerance():
    result = reconcile.compare([bill(amount=2.504)], [record()], CATALOG)
    assert result["status"] == "clean"
    assert result["counts"][reconcile.STATUS_MATCHED] == 1
    assert result["rows"][0]["status"] == reconcile.STATUS_MATCHED
    assert result["net_gap"] == pytest.approx(0.004, abs=1e-6)


def test_compare_flags_a_pricing_mismatch_beyond_tolerance():
    result = reconcile.compare([bill(amount=4.00)], [record()], CATALOG)
    assert result["status"] == "gap"
    assert result["counts"][reconcile.STATUS_MISMATCH] == 1
    assert result["net_gap"] == pytest.approx(1.50)


def test_compare_finds_a_model_missing_locally():
    result = reconcile.compare([bill(model="claude-3-opus", amount=15.0)],
                               [record()], CATALOG)
    row = result["rows"][0]
    assert row["status"] == reconcile.STATUS_MISSING_LOCALLY
    assert row["local_usd"] == 0.0 and row["local_records"] == 0


def test_compare_finds_a_model_not_on_the_bill():
    result = reconcile.compare([bill(amount=2.50)],
                               [record(), record(model="gpt-4o-mini")], CATALOG)
    row = next(r for r in result["rows"] if r["model"] == "gpt-4o-mini")
    assert row["status"] == reconcile.STATUS_NOT_ON_BILL
    assert row["billed_usd"] == 0.0 and row["local_records"] == 1


def test_compare_rejects_a_negative_tolerance():
    with pytest.raises(ValueError):
        reconcile.compare([bill()], [record()], CATALOG, tolerance=-0.01)


def test_compare_orders_rows_by_billed_amount():
    lines = [bill(model="gpt-4o-mini", amount=0.15), bill(amount=2.50)]
    result = reconcile.compare(lines, [record()], CATALOG)
    assert [row["model"] for row in result["rows"]][0] == "gpt-4o"


def test_compare_coverage_is_the_matched_share_of_the_bill():
    lines = [bill(amount=2.50), bill(model="claude-3-opus", amount=7.50)]
    result = reconcile.compare(lines, [record()], CATALOG)
    # 2.50 of a 10.00 bill is matched.
    assert result["matched_usd"] == pytest.approx(2.50)
    assert result["coverage"] == pytest.approx(0.25)


def test_compare_warns_when_coverage_is_low():
    lines = [bill(amount=2.50), bill(model="claude-3-opus", amount=7.50)]
    result = reconcile.compare(lines, [record()], CATALOG)
    assert any("unreliable" in text for text in result["warnings"])


def test_compare_warns_when_both_sides_are_empty():
    result = reconcile.compare([], [], CATALOG)
    assert result["status"] == "clean"          # nothing to chase, but...
    assert any("both empty" in text for text in result["warnings"])


def test_gap_rows_returns_only_rows_that_need_attention():
    lines = [bill(amount=2.50), bill(model="claude-3-opus", amount=15.0)]
    result = reconcile.compare(lines, [record()], CATALOG)
    assert [row["model"] for row in reconcile.gap_rows(result)] == ["claude-3-opus"]


# -- token verdicts --------------------------------------------------------------

def verdict_rows(bill_amount, bill_tokens, local_tokens):
    """A mismatch row where both sides exist, with controllable token counts."""
    bill_side = {"amount_usd": bill_amount, "tokens": bill_tokens,
                 "token_lines": 1, "lines": 1}
    local_side = {"amount_usd": 2.50, "tokens": local_tokens, "records": 1}
    return bill_side, local_side


def test_token_verdict_price_when_token_counts_agree():
    b, l = verdict_rows(bill_amount=4.00, bill_tokens=1_000_000,
                        local_tokens=1_000_000)
    verdict = reconcile.token_verdict("gpt-4o", b, l, CATALOG)
    assert verdict["verdict"] == "price"
    assert "the rate" in verdict["reason"]
    assert verdict["implied_rate"] == pytest.approx(4.00)
    assert verdict["catalog_rate"] == pytest.approx(2.50)


def test_token_verdict_records_when_token_counts_differ():
    b, l = verdict_rows(bill_amount=4.00, bill_tokens=2_000_000,
                        local_tokens=1_000_000)
    verdict = reconcile.token_verdict("gpt-4o", b, l, CATALOG)
    assert verdict["verdict"] == "records"
    assert "cannot be judged" in verdict["reason"]


def test_token_verdict_not_evaluated_without_bill_token_counts():
    b, l = verdict_rows(bill_amount=4.00, bill_tokens=0, local_tokens=1_000_000)
    b["token_lines"] = 0                    # the bill carries no counts at all
    verdict = reconcile.token_verdict("gpt-4o", b, l, CATALOG)
    assert verdict["verdict"] == "not evaluated"
    assert "no token counts" in verdict["reason"]


def test_compare_attaches_a_verdict_to_mismatched_rows_only():
    lines = [bill(amount=4.00, input_tokens=1_000_000, output_tokens=0)]
    result = reconcile.compare(lines, [record()], CATALOG)
    row = result["rows"][0]
    assert row["status"] == reconcile.STATUS_MISMATCH
    assert row["verdict"]["verdict"] == "price"


# -- explanations -----------------------------------------------------------------

def test_explain_lists_hypotheses_for_each_gap():
    lines = [bill(amount=2.50), bill(model="claude-3-opus", amount=15.0),
             bill(model="gpt-4-turbo", amount=10.0)]
    result = reconcile.compare(
        lines, [record(), record(model="gpt-4o-mini")], CATALOG)
    explained = reconcile.explain(result)
    statuses = {item["model"]: item["status"] for item in explained}
    assert statuses == {"claude-3-opus": reconcile.STATUS_MISSING_LOCALLY,
                        "gpt-4-turbo": reconcile.STATUS_MISSING_LOCALLY,
                        "gpt-4o-mini": reconcile.STATUS_NOT_ON_BILL}
    for item in explained:
        assert item["hypotheses"], "every gap row carries candidate causes"


def test_explain_carries_token_evidence_when_available():
    lines = [bill(amount=4.00, input_tokens=1_000_000, output_tokens=0)]
    result = reconcile.compare(lines, [record()], CATALOG)
    explained = reconcile.explain(result)
    assert explained[0]["token_evidence"]["verdict"] == "price"


def test_explain_is_empty_when_the_bill_reconciles():
    result = reconcile.compare([bill(amount=2.50)], [record()], CATALOG)
    assert reconcile.explain(result) == []
