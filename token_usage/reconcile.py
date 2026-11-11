"""Bill reconciliation: does the provider's invoice match local records?

:mod:`token_usage.ingest` turns a provider export into local records, but nothing
afterwards checks those records against what the provider actually charged. This
module answers the question finance asks first: *the invoice says $X, the
workspace says $Y -- where does the gap come from?*

A bill is a list of **bill lines**: one row per model (or per model+period) with
the amount the provider charged, and optionally the token counts behind it. The
local side is the workspace's records priced through the catalog. Both sides are
aggregated per model and subtracted.

Every model lands in one of four states:

``matched``
    Billed and local agree within the tolerance. Nothing to do.
``mismatch``
    Both sides have the model but the amounts differ beyond the tolerance.
    When the bill carries token counts, :func:`token_verdict` can narrow the
    cause; without them the cause is ``not evaluated``, never guessed.
``missing_locally``
    The bill charges for a model the workspace has no records for.
``not_on_bill``
    The workspace has records for a model the bill never charges.

Rules that keep the reconciliation honest:

* **Explanations are hypotheses, not diagnoses.** Two aggregate totals can never
  determine a unique cause, so :func:`explain` says what a gap is *consistent
  with* and never asserts why it happened.
* **A model without token counts is "not evaluated", not "fine".** Silence from
  the evidence is reported as silence, not as agreement.
* **Coverage below half the bill makes the whole verdict unreliable** and is
  warned about rather than buried.
* **The tolerance is explicit.** Rounding noise exists; a default of one USD
  keeps it out of the report, and every row states which side of the line it
  fell on.

Pure functions only: bill lines and records in, plain dicts out. Only
:func:`load_bill` touches the filesystem.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import round_usd

# Amounts within this many dollars of each other count as agreeing; the default
# absorbs provider-side rounding without absorbing a real pricing error.
DEFAULT_TOLERANCE = 1.0
# When less than this share of the bill is matched, the verdict is unreliable.
MIN_COVERAGE = 0.5

STATUS_MATCHED = "matched"
STATUS_MISMATCH = "mismatch"
STATUS_MISSING_LOCALLY = "missing_locally"
STATUS_NOT_ON_BILL = "not_on_bill"

# What each kind of gap is consistent with. Wording matters: these are candidate
# explanations for a human to check, never conclusions the data cannot support.
HYPOTHESES = {
    STATUS_MISSING_LOCALLY: [
        "an export for this model was never imported",
        "records were pruned or archived out of the workspace",
        "the bill covers a period the workspace does not",
    ],
    STATUS_NOT_ON_BILL: [
        "records were imported more than once",
        "test or manual records mixed into the workspace",
        "the workspace covers a period the bill does not",
    ],
    STATUS_MISMATCH: [
        "the pricing catalog is stale relative to the provider",
        "a negotiated or discounted rate is not in the catalog",
        "the two sides cover different record sets",
    ],
}


@dataclass
class BillLine:
    """One chargeable row from a provider invoice.

    Token counts are optional because real invoices often carry only amounts;
    when absent, token-level verification for that model is "not evaluated"
    rather than silently skipped.
    """

    model: str
    amount_usd: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    period: str = ""

    def total_tokens(self) -> Optional[int]:
        """Token total when the line carries counts, else ``None``."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def _coerce_amount(value: Any) -> Optional[float]:
    """Parse a money-ish value ('$1,234.50', '12.5'); ``None`` when unusable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    """Token counts are optional: blank means unknown, not zero."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(",", "").replace("_", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_bill_rows(rows: Any) -> Tuple[List[BillLine], List[str]]:
    """Turn raw rows (list of dicts) into bill lines; invalid rows are counted.

    Returns ``(lines, errors)`` so a single malformed row never hides the rows
    that did parse -- the same tolerance :mod:`token_usage.ingest` applies.
    """
    if isinstance(rows, dict):
        rows = rows.get("lines", rows.get("data", rows.get("records", [])))
    if not isinstance(rows, list):
        return [], ["bill is not a list of rows"]
    lines: List[BillLine] = []
    errors: List[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"row {index}: not an object")
            continue
        model = str(row.get("model") or "").strip()
        if not model:
            errors.append(f"row {index}: missing model")
            continue
        amount = _coerce_amount(row.get("amount_usd", row.get("cost_usd")))
        if amount is None:
            errors.append(f"row {index} ({model}): missing or invalid amount_usd")
            continue
        lines.append(BillLine(
            model=model, amount_usd=amount,
            input_tokens=_coerce_optional_int(row.get("input_tokens")),
            output_tokens=_coerce_optional_int(row.get("output_tokens")),
            cached_input_tokens=_coerce_optional_int(row.get("cached_input_tokens")),
            period=str(row.get("period") or "").strip()))
    return lines, errors


def load_bill(path: Path) -> Tuple[List[BillLine], List[str]]:
    """Read a bill from a JSON or CSV file and parse it into lines."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        rows: Any = list(csv.DictReader(io.StringIO(text)))
    else:
        rows = json.loads(text)
    return parse_bill_rows(rows)


def billed_totals(lines: Iterable[BillLine]) -> Dict[str, Dict]:
    """Aggregate bill lines per model: amount, token total, line count."""
    result: Dict[str, Dict] = {}
    for line in lines:
        bucket = result.setdefault(line.model, {
            "amount_usd": 0.0, "tokens": 0, "token_lines": 0, "lines": 0})
        bucket["amount_usd"] += line.amount_usd
        bucket["lines"] += 1
        tokens = line.total_tokens()
        if tokens is not None:
            bucket["tokens"] += tokens
            bucket["token_lines"] += 1
    for bucket in result.values():
        bucket["amount_usd"] = round_usd(bucket["amount_usd"])
    return result


def local_totals(records: Iterable[UsageRecord],
                 catalog: PricingCatalog) -> Dict[str, Dict]:
    """Aggregate workspace records per model: priced amount, tokens, count."""
    result: Dict[str, Dict] = {}
    for record in records:
        bucket = result.setdefault(record.model, {
            "amount_usd": 0.0, "tokens": 0, "records": 0})
        bucket["amount_usd"] += catalog.cost_for_record(record)
        bucket["tokens"] += record.total_tokens
        bucket["records"] += 1
    for bucket in result.values():
        bucket["amount_usd"] = round_usd(bucket["amount_usd"])
    return result


def _implied_per_1m(amount: float, tokens: int) -> Optional[float]:
    """Effective blended rate; ``None`` when there is nothing to divide by."""
    if tokens <= 0:
        return None
    return round_usd(amount / tokens * 1_000_000.0)


def token_verdict(model: str, bill: Dict, local: Dict,
                  catalog: PricingCatalog) -> Optional[Dict]:
    """Narrow a mismatch using token counts, when the bill carries them.

    Returns ``None`` for rows where token evidence cannot exist (missing or
    unmatched models). A verdict of ``not evaluated`` means the bill showed up
    without token counts -- that is reported, never treated as agreement.
    """
    if bill.get("token_lines", 0) == 0:
        return {"verdict": "not evaluated",
                "reason": "the bill carries no token counts for this model"}
    bill_tokens = bill["tokens"]
    local_tokens = local["tokens"]
    implied = _implied_per_1m(bill["amount_usd"], bill_tokens)
    catalog_rate = _implied_per_1m(local["amount_usd"], local_tokens)
    if bill_tokens == local_tokens:
        return {"verdict": "price",
                "reason": "token counts agree, so the gap is the rate",
                "bill_tokens": bill_tokens, "local_tokens": local_tokens,
                "implied_rate": implied, "catalog_rate": catalog_rate}
    return {"verdict": "records",
            "reason": "token counts differ, so the record sets disagree; "
                      "the rate cannot be judged from these totals",
            "bill_tokens": bill_tokens, "local_tokens": local_tokens,
            "implied_rate": implied, "catalog_rate": catalog_rate}


def compare(lines: Iterable[BillLine], records: Iterable[UsageRecord],
            catalog: PricingCatalog,
            tolerance: float = DEFAULT_TOLERANCE) -> Dict:
    """Reconcile a bill against workspace records, one row per model.

    Rows are ordered by billed amount (models missing from the bill sort last,
    by their local amount). The result carries everything the views need:
    per-model rows, overall totals, coverage, and any reliability warnings.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    bill = billed_totals(lines)
    local = local_totals(records, catalog)

    rows: List[Dict] = []
    for model in sorted(set(bill) | set(local)):
        b = bill.get(model)
        l = local.get(model)
        if b is not None and l is not None:
            diff = round_usd(b["amount_usd"] - l["amount_usd"])
            status = (STATUS_MATCHED if abs(diff) <= tolerance
                      else STATUS_MISMATCH)
            row = {"model": model, "status": status,
                   "billed_usd": b["amount_usd"], "local_usd": l["amount_usd"],
                   "diff_usd": diff, "abs_diff": round_usd(abs(diff)),
                   "bill_tokens": b["tokens"], "local_tokens": l["tokens"],
                   "bill_lines": b["lines"], "local_records": l["records"],
                   "verdict": (token_verdict(model, b, l, catalog)
                               if status == STATUS_MISMATCH else None)}
        elif b is not None:
            row = {"model": model, "status": STATUS_MISSING_LOCALLY,
                   "billed_usd": b["amount_usd"], "local_usd": 0.0,
                   "diff_usd": b["amount_usd"], "abs_diff": b["amount_usd"],
                   "bill_tokens": b["tokens"], "local_tokens": 0,
                   "bill_lines": b["lines"], "local_records": 0, "verdict": None}
        else:
            row = {"model": model, "status": STATUS_NOT_ON_BILL,
                   "billed_usd": 0.0, "local_usd": l["amount_usd"],
                   "diff_usd": round_usd(-l["amount_usd"]),
                   "abs_diff": l["amount_usd"],
                   "bill_tokens": 0, "local_tokens": l["tokens"],
                   "bill_lines": 0, "local_records": l["records"], "verdict": None}
        rows.append(row)
    rows.sort(key=lambda r: (r["billed_usd"], r["local_usd"]), reverse=True)

    bill_total = round_usd(sum(r["billed_usd"] for r in rows))
    local_total = round_usd(sum(r["local_usd"] for r in rows))
    matched_usd = round_usd(sum(r["billed_usd"] for r in rows
                                if r["status"] == STATUS_MATCHED))
    counts = {name: sum(1 for r in rows if r["status"] == name)
              for name in (STATUS_MATCHED, STATUS_MISMATCH,
                           STATUS_MISSING_LOCALLY, STATUS_NOT_ON_BILL)}
    coverage = round(matched_usd / bill_total, 4) if bill_total > 0 else 1.0

    warnings: List[str] = []
    if bill_total == 0.0 and local_total == 0.0:
        warnings.append("bill and workspace are both empty; nothing to reconcile")
    if bill_total > 0 and coverage < MIN_COVERAGE:
        warnings.append(
            f"only {coverage * 100:.1f}% of the bill is matched; "
            "treat this verdict as unreliable until the gaps are explained")

    return {"tolerance": tolerance, "rows": rows, "counts": counts,
            "bill_total": bill_total, "local_total": local_total,
            "net_gap": round_usd(bill_total - local_total),
            "matched_usd": matched_usd, "coverage": coverage,
            "status": "clean" if counts[STATUS_MISMATCH] == 0
            and counts[STATUS_MISSING_LOCALLY] == 0
            and counts[STATUS_NOT_ON_BILL] == 0 else "gap",
            "warnings": warnings}


def gap_rows(result: Dict) -> List[Dict]:
    """Only the rows that need attention, most money first."""
    return [row for row in result["rows"] if row["status"] != STATUS_MATCHED]


def explain(result: Dict) -> List[Dict]:
    """Candidate explanations for every gap row, in gap order.

    Each hypothesis is phrased as something the gap is *consistent with*; the
    data behind two aggregate totals cannot prove a cause, and this function
    refuses to pretend otherwise.
    """
    explained: List[Dict] = []
    for row in gap_rows(result):
        entry: Dict[str, Any] = {
            "model": row["model"], "status": row["status"],
            "diff_usd": row["diff_usd"],
            "hypotheses": list(HYPOTHESES[row["status"]])}
        verdict = row.get("verdict")
        if verdict is not None:
            entry["token_evidence"] = verdict
        explained.append(entry)
    return explained
