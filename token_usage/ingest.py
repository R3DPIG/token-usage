"""Ingest usage data from provider exports.

The manual write path (``record``) is fine for spot checks, but real usage data
arrives as a provider export. This module turns those exports into
:class:`~token_usage.models.UsageRecord` objects so they flow through the same
storage and analytics as hand-entered records.

Three shapes are recognised:

* ``openai``    -- the legacy OpenAI usage API payload (``data[].snapshot_id``).
* ``anthropic`` -- the Anthropic Admin usage report (``data[].results[]``).
* ``generic``   -- a flat list of rows using our own field names.

Parsing is pure (payload in, records + report out); only :func:`load_payload`
touches the filesystem, so every parser is unit-testable.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from token_usage.models import UsageRecord
from token_usage.utils import ISO_FORMAT, utcnow_iso

SUPPORTED_FORMATS = ("openai", "anthropic", "generic")


@dataclass
class IngestResult:
    """Outcome of parsing one export: what was kept, skipped, and why."""

    records: List[UsageRecord] = field(default_factory=list)
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    source_format: str = "unknown"

    @property
    def imported(self) -> int:
        return len(self.records)

    def summary(self) -> str:
        parts = [f"{self.imported} record(s) parsed from {self.source_format}"]
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return ", ".join(parts)


def _coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort integer coercion; tolerates ``'1,000'`` and ``'1.0'``."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(",", "").replace("_", "")
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _unix_to_iso(value: Any) -> str:
    """Convert unix seconds to our ISO format; fall back to 'now'."""
    seconds = _coerce_int(value, default=0)
    if seconds <= 0:
        return utcnow_iso()
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(ISO_FORMAT)


def _coerce_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _add(result: IngestResult, **kwargs) -> None:
    """Build a record, counting validation failures instead of raising."""
    try:
        result.records.append(UsageRecord(**kwargs))
    except ValueError as exc:
        result.skipped += 1
        result.errors.append(str(exc))


def parse_openai_usage(payload: Any, project: str = "openai") -> IngestResult:
    """Parse the legacy OpenAI usage API payload."""
    result = IngestResult(source_format="openai")
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        result.errors.append("openai payload has no 'data' list")
        return result
    for row in rows:
        if not isinstance(row, dict):
            result.skipped += 1
            continue
        model = row.get("snapshot_id") or row.get("model")
        if not model:
            result.skipped += 1
            continue
        _add(result, model=str(model),
             input_tokens=_coerce_int(row.get("n_context_tokens_total")),
             output_tokens=_coerce_int(row.get("n_generated_tokens_total")),
             timestamp=_unix_to_iso(row.get("aggregation_timestamp")),
             project=project, tags=["import", "openai"],
             note=f"requests={_coerce_int(row.get('n_requests'))}")
    return result


def parse_anthropic_usage(payload: Any, project: str = "anthropic") -> IngestResult:
    """Parse the Anthropic Admin usage report.

    Cache-creation tokens are billed at the standard input rate, so they are
    folded into ``input_tokens``; cache reads are kept separate and clamped so
    they can never exceed the input total.
    """
    result = IngestResult(source_format="anthropic")
    buckets = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(buckets, list):
        result.errors.append("anthropic payload has no 'data' list")
        return result
    for bucket in buckets:
        if not isinstance(bucket, dict):
            result.skipped += 1
            continue
        timestamp = bucket.get("starting_at") or bucket.get("ending_at") or utcnow_iso()
        rows = bucket.get("results") if isinstance(bucket.get("results"), list) else [bucket]
        for row in rows:
            if not isinstance(row, dict) or not row.get("model"):
                result.skipped += 1
                continue
            input_tokens = (_coerce_int(row.get("input_tokens"))
                            + _coerce_int(row.get("cache_creation_input_tokens")))
            cached = min(_coerce_int(row.get("cache_read_input_tokens")), input_tokens)
            _add(result, model=str(row["model"]), input_tokens=input_tokens,
                 output_tokens=_coerce_int(row.get("output_tokens")),
                 timestamp=str(timestamp), cached_input_tokens=cached,
                 project=project, tags=["import", "anthropic"])
    return result


def parse_generic_rows(rows: Any, project: str = "imported") -> IngestResult:
    """Parse a flat list of rows using our own field names."""
    result = IngestResult(source_format="generic")
    if isinstance(rows, dict):
        rows = rows.get("records", rows.get("data", []))
    if not isinstance(rows, list):
        result.errors.append("generic payload is not a list of rows")
        return result
    for row in rows:
        if not isinstance(row, dict) or not row.get("model"):
            result.skipped += 1
            continue
        _add(result, model=str(row["model"]),
             input_tokens=_coerce_int(row.get("input_tokens")),
             output_tokens=_coerce_int(row.get("output_tokens")),
             timestamp=str(row.get("timestamp") or utcnow_iso()),
             cached_input_tokens=_coerce_int(row.get("cached_input_tokens")),
             project=str(row.get("project") or project),
             tags=_coerce_tags(row.get("tags")),
             note=str(row.get("note") or ""))
    return result


_PARSERS = {"openai": parse_openai_usage, "anthropic": parse_anthropic_usage,
            "generic": parse_generic_rows}


def detect_format(payload: Any) -> str:
    """Guess the export shape from its structure; ``unknown`` if unrecognised."""
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        first = payload["data"][0] if payload["data"] else None
        if isinstance(first, dict):
            if "snapshot_id" in first or "n_context_tokens_total" in first:
                return "openai"
            if "results" in first or "starting_at" in first:
                return "anthropic"
    if isinstance(payload, list) or (isinstance(payload, dict) and "records" in payload):
        return "generic"
    return "unknown"


def load_payload(path: Path) -> Any:
    """Read a JSON or CSV export from disk into a Python structure."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".csv":
        return list(csv.DictReader(io.StringIO(text)))
    return json.loads(text)


def ingest_payload(payload: Any, fmt: str, project: str = "imported") -> IngestResult:
    """Dispatch to the named parser."""
    if fmt not in _PARSERS:
        raise ValueError(f"unsupported format {fmt!r}; expected one of {SUPPORTED_FORMATS}")
    return _PARSERS[fmt](payload, project=project)


def ingest_file(path: Path, fmt: str = "", project: str = "imported") -> IngestResult:
    """Load and parse a file in one step, auto-detecting the format if omitted."""
    payload = load_payload(path)
    resolved = fmt or detect_format(payload)
    if resolved == "unknown":
        raise ValueError("could not detect export format; pass --format explicitly")
    return ingest_payload(payload, resolved, project=project)
