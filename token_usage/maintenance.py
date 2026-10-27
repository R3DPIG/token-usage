"""Data lifecycle: dedupe, prune, edit, merge, integrity scan.

Records are append-only by design (see :mod:`token_usage.storage`), which keeps
writes safe but means mistakes accumulate: a re-run import double-counts, a typo
sticks forever, two teammates' workspaces never combine. This module holds the
pure logic for cleaning that history up; the plugins wrap it with safety rails
(dry-run by default, explicit confirmation for destructive actions).

Same contract as the other domain modules: plain data in, plain data out.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import parse_date, parse_iso, round_usd

# Fields a user is allowed to change through ``edit``. Anything else (notably
# ``cost_usd``, which is derived) is rejected rather than silently accepted.
EDITABLE_FIELDS = ("model", "input_tokens", "output_tokens", "cached_input_tokens",
                   "timestamp", "project", "tags", "note")


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def fingerprint(record: UsageRecord, ignore_timestamp: bool = False) -> Tuple:
    """Identity of a record for duplicate detection.

    By default the timestamp is part of the identity, so two genuinely separate
    calls that happen to use the same model and token counts are *not* merged.
    ``ignore_timestamp`` relaxes that, which is what you want when re-importing
    the same export twice with slightly different clock skew.
    """
    key = (record.model, record.input_tokens, record.output_tokens,
           record.cached_input_tokens, record.project)
    return key if ignore_timestamp else key + (record.timestamp,)


def find_duplicates(records: Sequence[UsageRecord],
                    ignore_timestamp: bool = False) -> List[List[int]]:
    """Return groups of indices that share a fingerprint (only groups of 2+)."""
    buckets: Dict[Tuple, List[int]] = {}
    for index, record in enumerate(records):
        buckets.setdefault(fingerprint(record, ignore_timestamp), []).append(index)
    return [indices for indices in buckets.values() if len(indices) > 1]


def dedupe(records: Sequence[UsageRecord], ignore_timestamp: bool = False,
           keep: str = "first") -> Tuple[List[UsageRecord], List[UsageRecord], List[List[int]]]:
    """Drop duplicate records, keeping the first (or last) of each group.

    Returns ``(kept, removed, groups)``. ``removed`` holds the actual dropped
    records so a caller can report their value; ``groups`` holds the colliding
    index groups for diagnostics.
    """
    if keep not in ("first", "last"):
        raise ValueError(f"keep must be 'first' or 'last', got {keep!r}")
    groups = find_duplicates(records, ignore_timestamp)
    drop = set()
    for indices in groups:
        survivor = indices[0] if keep == "first" else indices[-1]
        drop.update(i for i in indices if i != survivor)
    kept = [record for index, record in enumerate(records) if index not in drop]
    removed = [record for index, record in enumerate(records) if index in drop]
    return kept, removed, groups


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def _matches(record: UsageRecord, project: Optional[str], model: Optional[str],
             tag: Optional[str], before: Optional[datetime],
             after: Optional[datetime]) -> bool:
    """True when the record satisfies *every* supplied criterion."""
    if project is not None and record.project != project:
        return False
    if model is not None and record.model != model:
        return False
    if tag is not None and tag not in record.tags:
        return False
    timestamp = parse_iso(record.timestamp)
    if before is not None and timestamp >= before:
        return False
    if after is not None and timestamp < after:
        return False
    return True


def prune(records: Iterable[UsageRecord], project: Optional[str] = None,
          model: Optional[str] = None, tag: Optional[str] = None,
          before: Optional[str] = None, after: Optional[str] = None
          ) -> Tuple[List[UsageRecord], List[UsageRecord]]:
    """Split records into ``(kept, removed)`` by the given criteria.

    ``before`` is exclusive and ``after`` inclusive, matching how people read
    "delete everything before May" (May itself survives). Bare ``YYYY-MM-DD``
    dates are accepted alongside full timestamps.
    """
    before_dt = _bound(before) if before else None
    after_dt = _bound(after) if after else None
    kept: List[UsageRecord] = []
    removed: List[UsageRecord] = []
    for record in records:
        target = removed if _matches(record, project, model, tag, before_dt, after_dt) else kept
        target.append(record)
    return kept, removed


def _bound(value: str) -> datetime:
    return parse_iso(value) if "T" in value else parse_date(value)


def split_by_age(records: Iterable[UsageRecord], cutoff: str
                 ) -> Tuple[List[UsageRecord], List[UsageRecord]]:
    """Split into ``(recent, old)`` at ``cutoff``; old means strictly before it."""
    cutoff_dt = _bound(cutoff)
    recent, old = [], []
    for record in records:
        (old if parse_iso(record.timestamp) < cutoff_dt else recent).append(record)
    return recent, old


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def apply_edits(record: UsageRecord, changes: Dict) -> UsageRecord:
    """Return a new record with ``changes`` applied, re-validated.

    Rebuilding through :meth:`UsageRecord.from_dict` means every invariant
    (non-negative tokens, cached <= input) is re-checked, so an edit can never
    produce a record that the normal write path would have rejected.
    """
    unknown = [key for key in changes if key not in EDITABLE_FIELDS]
    if unknown:
        raise ValueError(f"cannot edit field(s) {unknown}; "
                         f"editable fields are {list(EDITABLE_FIELDS)}")
    data = record.to_dict()
    for key, value in changes.items():
        if key == "tags" and isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        data[key] = value
    return UsageRecord.from_dict(data)


def diff_records(before: UsageRecord, after: UsageRecord) -> List[Tuple[str, object, object]]:
    """Return ``(field, old, new)`` triples for every field that changed."""
    old, new = before.to_dict(), after.to_dict()
    return [(key, old[key], new[key]) for key in EDITABLE_FIELDS if old[key] != new[key]]


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_records(*sources: Iterable[UsageRecord], dedupe_after: bool = True,
                  ignore_timestamp: bool = False) -> Tuple[List[UsageRecord], Dict]:
    """Concatenate several record streams, optionally de-duplicating the result.

    Merging is non-destructive to the sources; the caller decides what to write.
    """
    combined = [record for source in sources for record in source]
    if not dedupe_after:
        return combined, {"total": len(combined), "duplicates_removed": 0}
    kept, removed, _ = dedupe(combined, ignore_timestamp=ignore_timestamp)
    return kept, {"total": len(combined), "duplicates_removed": len(removed)}


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def scan_integrity(records: Iterable[UsageRecord], catalog: PricingCatalog,
                   now: Optional[datetime] = None) -> List[Dict]:
    """Find records that are structurally suspicious.

    Each issue carries the record's index so a caller can act on it. Nothing is
    modified here -- this is a read-only audit.
    """
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(days=1)  # tolerate small clock skew
    issues: List[Dict] = []
    for index, record in enumerate(records):
        if record.total_tokens == 0:
            issues.append({"index": index, "kind": "zero_tokens",
                           "detail": f"{record.model} has no tokens"})
        if record.cached_input_tokens > record.input_tokens:
            issues.append({"index": index, "kind": "cached_exceeds_input",
                           "detail": f"{record.cached_input_tokens} > {record.input_tokens}"})
        if parse_iso(record.timestamp) > horizon:
            issues.append({"index": index, "kind": "future_timestamp",
                           "detail": record.timestamp})
        if catalog.get(record.model) is None:
            issues.append({"index": index, "kind": "unknown_model",
                           "detail": record.model})
    return issues


def summarize_issues(issues: Iterable[Dict]) -> Dict[str, int]:
    """Count issues by kind, for a compact report."""
    counts: Dict[str, int] = {}
    for issue in issues:
        counts[issue["kind"]] = counts.get(issue["kind"], 0) + 1
    return counts


def estimate_value(records: Iterable[UsageRecord], catalog: PricingCatalog) -> Dict:
    """Token and cost totals for a set of records, used to preview a deletion."""
    records = list(records)
    tokens = sum(record.total_tokens for record in records)
    cost = sum(catalog.cost_for_record(record) for record in records)
    return {"records": len(records), "tokens": tokens, "cost_usd": round_usd(cost)}
