"""Tests for the iteration-2 ingest layer (provider export parsing)."""

from __future__ import annotations

import json

import pytest

from token_usage.ingest import (
    detect_format,
    ingest_file,
    ingest_payload,
    parse_anthropic_usage,
    parse_generic_rows,
    parse_openai_usage,
)


def test_parse_openai_usage():
    payload = {"data": [
        {"snapshot_id": "gpt-4o", "n_context_tokens_total": 1000,
         "n_generated_tokens_total": 500, "aggregation_timestamp": 1714521600,
         "n_requests": 3},
        {"snapshot_id": "gpt-4o-mini", "n_context_tokens_total": "2,000",
         "n_generated_tokens_total": "1.0", "aggregation_timestamp": 1714521600},
    ]}
    result = parse_openai_usage(payload)
    assert result.imported == 2
    assert result.records[0].model == "gpt-4o"
    assert result.records[0].total_tokens == 1500
    assert result.records[1].input_tokens == 2000  # "2,000" coerced
    assert result.records[0].timestamp.startswith("2024-05-01")


def test_parse_openai_skips_rows_without_model():
    result = parse_openai_usage({"data": [{"n_context_tokens_total": 5}, "junk"]})
    assert result.imported == 0 and result.skipped == 2


def test_parse_anthropic_folds_cache_creation_and_clamps_reads():
    payload = {"data": [{"starting_at": "2024-05-01T00:00:00", "results": [
        {"model": "claude-3-5-sonnet", "input_tokens": 100,
         "cache_creation_input_tokens": 50, "cache_read_input_tokens": 30,
         "output_tokens": 200},
        # cache_read larger than input must be clamped, not rejected.
        {"model": "claude-3-5-haiku", "input_tokens": 10,
         "cache_read_input_tokens": 999, "output_tokens": 5},
    ]}]}
    result = parse_anthropic_usage(payload)
    assert result.imported == 2
    assert result.records[0].input_tokens == 150  # 100 + 50
    assert result.records[0].cached_input_tokens == 30
    assert result.records[1].cached_input_tokens == 10  # clamped to input


def test_parse_generic_rows_and_tags():
    result = parse_generic_rows([
        {"model": "gpt-4o", "input_tokens": 10, "output_tokens": 5,
         "timestamp": "2024-05-01T00:00:00", "tags": "a,b", "project": "p"},
        {"model": "gpt-4o", "input_tokens": 1, "output_tokens": 1},
    ])
    assert result.imported == 2
    assert result.records[0].tags == ["a", "b"]
    assert result.records[0].project == "p"
    assert result.records[1].project == "imported"  # default applied


def test_parse_generic_rejects_non_list():
    result = parse_generic_rows("not a list")
    assert result.imported == 0 and result.errors


def test_detect_format():
    assert detect_format({"data": [{"snapshot_id": "x"}]}) == "openai"
    assert detect_format({"data": [{"starting_at": "t", "results": []}]}) == "anthropic"
    assert detect_format([{"model": "x"}]) == "generic"
    assert detect_format({"nonsense": True}) == "unknown"


def test_ingest_payload_rejects_unknown_format():
    with pytest.raises(ValueError):
        ingest_payload([], "made-up")


def test_ingest_file_json_and_csv(tmp_path):
    json_path = tmp_path / "u.json"
    json_path.write_text(json.dumps([{"model": "gpt-4o", "input_tokens": 10,
                                      "output_tokens": 5}]), encoding="utf-8")
    assert ingest_file(json_path).imported == 1

    csv_path = tmp_path / "u.csv"
    csv_path.write_text("model,input_tokens,output_tokens\ngpt-4o,10,5\n", encoding="utf-8")
    assert ingest_file(csv_path).imported == 1


def test_ingest_file_undetectable_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nonsense": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        ingest_file(path)
