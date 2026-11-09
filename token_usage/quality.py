"""Data quality audit: how far should you trust these numbers?

:mod:`token_usage.maintenance` cleans data and :mod:`token_usage.plugins.doctor`
runs a handful of pass/fail probes. Neither answers the question someone about to
publish a figure actually has: *is this dataset good enough to reason from, and
if not, which specific defect is costing me?*

This module scores the dataset instead of merely probing it. Every check reports
how much of the data it affects, converts that share into a deduction from a 100
point score, and names the command that fixes it. The score is transparent by
construction -- every point lost is attributable to exactly one named check, so
"82.0 / 100" can always be expanded into a list of reasons.

Design rules, consistent with the rest of the package:

* **A check that cannot be evaluated says so.** With three records there is no
  meaningful outlier test, so the check reports ``ok`` *and* a reason, rather than
  passing silently and implying the data was inspected.
* **Severity reflects consequence, not counts.** One future-dated record corrupts
  every projection downstream; a thousand untagged records only cost a reporting
  dimension. Those must not score the same, so severity sets the ceiling on the
  deduction and the affected share sets the amount.
* **Penalties are capped per check.** Without a cap, one systemic problem -- a
  100k-row import with no tags -- would drive the score to zero and hide every
  other finding behind it.
* **A failure is never free.** A share-based deduction can round to ``-0.0`` when
  a critical defect touches very little of the data. Charging nothing for a
  reported defect invites the reader to skip it, so every failure costs at least
  :data:`MIN_PENALTY`.

Pure functions only: records + catalog in, plain dicts out.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from token_usage import maintenance
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import day_key, parse_iso, round_usd

SEVERITIES = ("critical", "warning", "notice")     # most to least severe

# Ceiling on each check's deduction: a critical defect may cost a fifth of the
# score, a notice at most four points.
MAX_PENALTY = {"critical": 20.0, "warning": 10.0, "notice": 4.0}
MIN_PENALTY = 0.1
# MAD -> standard-deviation scaling factor, used by the outlier check.
MAD_SCALE = 1.4826

GRADES = ((90.0, "excellent"), (75.0, "good"), (60.0, "fair"),
          (40.0, "poor"), (0.0, "unusable"))

CHECK_IDS = ("duplicates", "future_timestamps", "zero_tokens", "unknown_models",
             "cost_override_drift", "untagged", "default_project",
             "extreme_tokens", "day_coverage", "retention")


def grade(score: float) -> str:
    """Map a 0-100 score onto a human label."""
    for floor, label in GRADES:
        if score >= floor:
            return label
    return "unusable"


def severity_rank(severity: str) -> int:
    """Index in :data:`SEVERITIES`; lower is more severe."""
    return SEVERITIES.index(severity)


def _share(part: float, whole: float) -> float:
    return round(part / whole, 4) if whole > 0 else 0.0


def _spend(records: Iterable[UsageRecord], catalog: PricingCatalog) -> float:
    return sum(catalog.cost_for_record(record) for record in records)


def _check(check_id: str, title: str, severity: str, affected: int, total: int,
           summary: str, fix: str, share: Optional[float] = None,
           samples: Optional[List[Dict]] = None) -> Dict:
    """Build one uniform check record.

    ``share`` defaults to the affected-record ratio; money-weighted checks pass it
    explicitly so one expensive bad record outranks a hundred cheap ones.
    """
    if severity not in MAX_PENALTY:
        raise ValueError(f"unknown severity {severity!r}")
    ratio = _share(affected, total) if share is None else round(share, 4)
    ratio = max(0.0, min(1.0, ratio))
    penalty = round(MAX_PENALTY[severity] * ratio, 1)
    if affected > 0:
        penalty = max(penalty, MIN_PENALTY)      # see MIN_PENALTY above
    return {"id": check_id, "title": title, "severity": severity,
            "ok": affected == 0, "affected": affected, "share": ratio,
            "penalty": penalty, "summary": summary, "fix": fix,
            "samples": list(samples or [])[:3]}


def _samples(records: Iterable[UsageRecord]) -> List[Dict]:
    """Sample rows for a report, trimmed to the three most useful by the caller."""
    return [{"timestamp": r.timestamp, "model": r.model} for r in records]


def _money_check(check_id: str, title: str, offenders: List[UsageRecord],
                 records: List[UsageRecord], catalog: PricingCatalog,
                 describe, fix: str) -> Dict:
    """A notice scored by the share of *spend* the offenders represent.

    The coverage checks describe well-formed records that are merely
    under-specified, so counting rows would over-weight a thousand cheap calls
    against one expensive one.
    """
    total_cost = _spend(records, catalog)
    bad_cost = _spend(offenders, catalog)
    share = (_share(bad_cost, total_cost) if total_cost
             else _share(len(offenders), len(records)))
    return _check(check_id, title, "notice", len(offenders), len(records),
                  describe(bad_cost, share), fix, share=share,
                  samples=[{"timestamp": r.timestamp, "model": r.model,
                            "project": r.project} for r in offenders])


# -- structural checks: the record itself is malformed ----------------------

def check_zero_tokens(records: List[UsageRecord]) -> Dict:
    """Records that consumed nothing.

    Almost always a failed call that was still logged, or a parsing bug that
    dropped the counts. They cost nothing, but they inflate call counts and skew
    every per-call average.
    """
    offenders = [r for r in records if r.total_tokens == 0]
    return _check("zero_tokens", "records with no tokens", "warning",
                  len(offenders), len(records),
                  f"{len(offenders)} record(s) consumed zero tokens",
                  "token-usage edit --index <i> --note 'zero-token call' or prune",
                  samples=_samples(offenders))


def check_duplicates(records: List[UsageRecord],
                     catalog: PricingCatalog) -> Dict:
    """Double-counted records, weighted by the money they overstate spend.

    A re-run import silently doubling last month's bill is the most damaging
    defect for cost reporting, so this check is critical and weighted by dollars
    rather than by row count.
    """
    total_cost = _spend(records, catalog)
    duplicated = 0.0
    rows = 0
    samples: List[Dict] = []
    for group in maintenance.find_duplicates(records):
        for index in group[1:]:
            record = records[index]
            cost = catalog.cost_for_record(record)
            duplicated += cost
            rows += 1
            if len(samples) < 3:
                samples.append({"timestamp": record.timestamp,
                                "model": record.model,
                                "cost_usd": round_usd(cost)})
    share = (_share(duplicated, total_cost) if total_cost
             else _share(rows, len(records)))
    return _check("duplicates", "duplicate records inflating spend", "critical",
                  rows, len(records),
                  f"${duplicated:.4f} of spend is double-counted across "
                  f"{rows} record(s)",
                  "token-usage dedupe --apply", share=share, samples=samples)


def check_unknown_models(records: List[UsageRecord],
                         catalog: PricingCatalog) -> Dict:
    """Models with no catalog entry, silently priced at the fallback rate.

    The totals still produce a number, which is exactly the danger: the figure is
    confident and wrong. Reporting the share makes that error visible.
    """
    known = set(catalog.models())
    offenders = [r for r in records if r.model not in known]
    names = sorted({r.model for r in offenders})
    return _check(
        "unknown_models", "models missing from the price catalog", "warning",
        len(offenders), len(records),
        (f"{len(offenders)} record(s) on {len(names)} unpriced model(s): "
         f"{', '.join(names[:4])}" if offenders else "every model is priced"),
        "token-usage pricing --action set --model <m> --input-price <usd/1M> "
        "--output-price <usd/1M>",
        samples=_samples(offenders))


def check_future_timestamps(records: List[UsageRecord],
                            now: Optional[datetime] = None) -> Dict:
    """Records dated past tomorrow (one day of clock skew is tolerated).

    Future data is poison for anything period-scoped: the record sits outside
    every window until its date arrives, then shows up as a spike.
    """
    horizon = (now or datetime.now(timezone.utc)) + timedelta(days=1)
    offenders = [r for r in records if parse_iso(r.timestamp) > horizon]
    return _check("future_timestamps", "records dated in the future", "critical",
                  len(offenders), len(records),
                  f"{len(offenders)} record(s) are dated after {day_key(horizon)}",
                  "token-usage edit --index <i> --timestamp <yyyy-mm-ddThh:mm:ss>",
                  samples=_samples(offenders))


def check_cost_override_drift(records: List[UsageRecord], catalog: PricingCatalog,
                              tolerance_pct: float = 1.0) -> Dict:
    """Stored cost overrides that disagree with the catalog.

    A pinned ``cost_usd`` is respected on purpose (contract prices, credits), but
    a large drift is usually a leftover from an old price list, and it makes the
    bill unreproducible.
    """
    offenders = []
    for record in records:
        if record.cost_usd is None:
            continue
        computed = catalog.cost_for(record.model, record.input_tokens,
                                    record.output_tokens,
                                    record.cached_input_tokens)
        if abs(record.cost_usd - computed) > \
                abs(computed) * (tolerance_pct / 100.0) + 1e-9:
            offenders.append((record, computed))
    return _check(
        "cost_override_drift", "cost overrides out of step with the catalog",
        "warning", len(offenders), len(records),
        f"{len(offenders)} record(s) carry a pinned cost that no longer matches "
        "the catalog",
        "re-record the rows, or update pricing.json to the rate actually paid",
        samples=[{"timestamp": r.timestamp, "model": r.model,
                  "stored_usd": round_usd(r.cost_usd or 0.0),
                  "catalog_usd": round_usd(computed)}
                 for r, computed in offenders])


def check_extreme_tokens(records: List[UsageRecord], sigma: float = 5.0,
                         ratio_multiple: float = 100.0,
                         min_samples: int = 10) -> Dict:
    """Token counts implausibly far above the typical record.

    The centre and spread come from the **median and MAD**, not the mean and
    standard deviation. That matters: a single 900M-token row inflates mean and
    stddev enough to disguise itself as ordinary, so the naive z-score used
    elsewhere in this package reports "clean" on exactly the record it exists to
    catch. The MAD is immune, because half the data sits at the median regardless
    of how extreme the tail is.

    When more than half the records are identical the MAD is zero and no z-score
    exists; the check then falls back to a ratio test against the median and says
    which rule it used, rather than silently passing.
    """
    if len(records) < min_samples:
        return _check(
            "extreme_tokens", "implausibly large token counts", "notice", 0,
            len(records),
            f"not evaluated: needs {min_samples} records, found {len(records)}",
            "record more data")

    values = sorted(float(r.total_tokens) for r in records)
    median = statistics.median(values)
    scale = statistics.median([abs(v - median) for v in values]) * MAD_SCALE
    if scale > 0:
        cutoff = median + sigma * scale
        basis = f"median {median:,.0f}, robust sigma {scale:,.0f}"
    else:
        cutoff = median * ratio_multiple if median > 0 else float("inf")
        basis = f"median {median:,.0f}, spread too flat for a z-score"
    offenders = [r for r in records if r.total_tokens > cutoff]
    return _check(
        "extreme_tokens", "implausibly large token counts", "notice",
        len(offenders), len(records),
        f"{len(offenders)} record(s) exceed {cutoff:,.0f} tokens ({basis})",
        "verify these rows against the provider's dashboard before reporting",
        samples=[{"timestamp": r.timestamp, "model": r.model,
                  "total_tokens": r.total_tokens} for r in offenders])


# -- coverage and hygiene: well-formed but under-specified ------------------

def check_day_coverage(records: List[UsageRecord]) -> Dict:
    """Calendar days with no records inside the observed span.

    Gaps are not automatically wrong -- a quiet weekend is normal -- so this is a
    notice weighted by the missing share. A large gap still matters, because every
    average computed across it assumes zero-spend days.
    """
    days = sorted({day_key(parse_iso(r.timestamp)) for r in records})
    if len(days) < 2:
        return _check("day_coverage", "gaps in the daily series", "notice", 0,
                      max(len(records), 1),
                      "not evaluated: fewer than two distinct days",
                      "record more data")
    first = parse_iso(days[0] + "T00:00:00")
    last = parse_iso(days[-1] + "T00:00:00")
    span = (last - first).days + 1
    missing = span - len(days)
    largest = max(((parse_iso(b + "T00:00:00") - parse_iso(a + "T00:00:00")).days
                   - 1 for a, b in zip(days, days[1:])), default=0)
    return _check(
        "day_coverage", "gaps in the daily series", "notice", missing, span,
        (f"{missing} of {span} day(s) between {days[0]} and {days[-1]} have no "
         f"records (largest gap {largest} day(s))" if missing
         else f"every day from {days[0]} to {days[-1]} has records"),
        "if usage was exported, re-import the missing days; otherwise ignore")


def check_untagged(records: List[UsageRecord], catalog: PricingCatalog) -> Dict:
    """Spend that carries no tag, weighted by dollars.

    Tags drive every per-feature or per-customer roll-up. Untagged spend is not
    wrong, but it is unallocatable, so it is measured in money rather than rows.
    """
    offenders = [r for r in records if not r.tags]
    return _money_check(
        "untagged", "spend with no tag", offenders, records, catalog,
        lambda cost, share: (f"${cost:.4f} ({share * 100:.1f}% of spend) has no "
                             "tag, so it cannot be split by feature"),
        "token-usage edit --index <i> --tags <tag1,tag2> --apply")


def check_default_project(records: List[UsageRecord],
                          catalog: PricingCatalog) -> Dict:
    """Spend still sitting in the implicit ``default`` project.

    A record with no project assigned lands there. It costs nothing in accuracy,
    but it means the per-project report has a hole in it, so it is scored.
    """
    offenders = [r for r in records if r.project == "default"]
    return _money_check(
        "default_project", "spend left in the default project", offenders,
        records, catalog,
        lambda cost, share: (f"${cost:.4f} ({share * 100:.1f}% of spend) is not "
                             "assigned to a project"),
        "token-usage edit --index <i> --project <name>")


def check_retention(records: List[UsageRecord], now: Optional[datetime] = None,
                    max_age_days: int = 365) -> Dict:
    """Records old enough that they should have been archived.

    Not a correctness problem, but an ever-growing active file slows every command
    and makes period comparisons read against ancient history.
    """
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=max_age_days)
    old = [r for r in records if parse_iso(r.timestamp) < cutoff]
    oldest = min((r.timestamp for r in records), default="")
    return _check(
        "retention", "records past the retention horizon", "notice",
        len(old), len(records),
        (f"{len(old)} record(s) are older than {max_age_days} day(s) "
         f"(oldest {oldest[:10]})" if old
         else f"all records are within {max_age_days} day(s) of today"),
        f"token-usage archive --label old --before {day_key(cutoff)} --apply",
        samples=_samples(old))


# -- aggregation ------------------------------------------------------------

def run_checks(records: Iterable[UsageRecord], catalog: PricingCatalog,
               now: Optional[datetime] = None) -> List[Dict]:
    """Run every check; order is fixed so reports are diffable run to run."""
    records = list(records)
    now = now or datetime.now(timezone.utc)
    return [
        check_duplicates(records, catalog),
        check_future_timestamps(records, now=now),
        check_zero_tokens(records),
        check_unknown_models(records, catalog),
        check_cost_override_drift(records, catalog),
        check_untagged(records, catalog),
        check_default_project(records, catalog),
        check_extreme_tokens(records),
        check_day_coverage(records),
        check_retention(records, now=now),
    ]


def audit(records: Iterable[UsageRecord], catalog: PricingCatalog,
          now: Optional[datetime] = None) -> Dict:
    """Score the dataset and report every finding behind the score.

    An empty dataset scores 100 with ``empty`` set: nothing is wrong with it, and
    there is nothing in it either. The flag lets a caller tell "clean" from "no
    data" instead of reading a perfect score as good news.
    """
    records = list(records)
    if not records:
        return {"empty": True, "checks": [], "failing": [], "score": 100.0,
                "grade": "excellent", "penalty_total": 0.0, "critical": 0,
                "warnings": 0, "notices": 0, "total_records": 0, "total_usd": 0.0}

    checks = run_checks(records, catalog, now=now)
    penalty = round(sum(check["penalty"] for check in checks), 1)
    score = round(max(0.0, 100.0 - penalty), 1)
    failing = [check for check in checks if not check["ok"]]
    counts = {level: sum(1 for c in failing if c["severity"] == level)
              for level in SEVERITIES}
    return {"empty": False, "checks": checks, "failing": failing, "score": score,
            "grade": grade(score), "penalty_total": penalty,
            "critical": counts["critical"], "warnings": counts["warning"],
            "notices": counts["notice"],
            "total_records": len(records),
            "total_usd": round_usd(_spend(records, catalog))}


def next_steps(result: Dict, limit: int = 5) -> List[Dict]:
    """Failing checks ranked the way a person should work through them.

    Severity first, then points lost: finishing a critical finding is worth more
    than shaving a notice, and among equals the bigger deduction wins.
    """
    ranked = sorted(result["failing"],
                    key=lambda c: (severity_rank(c["severity"]), -c["penalty"]))
    return [{"id": c["id"], "severity": c["severity"], "penalty": c["penalty"],
             "summary": c["summary"], "fix": c["fix"]} for c in ranked[:limit]]


def filter_checks(checks: Iterable[Dict], check_id: Optional[str] = None,
                  min_severity: Optional[str] = None) -> List[Dict]:
    """Narrow the report for display, without changing the score.

    Filtering is presentation only: hiding a check must never improve the score,
    or the score stops meaning anything.
    """
    if min_severity is not None and min_severity not in SEVERITIES:
        raise ValueError(f"min_severity must be one of {SEVERITIES}")
    ceiling = severity_rank(min_severity) if min_severity else len(SEVERITIES) - 1
    return [check for check in checks
            if (check_id is None or check["id"] == check_id)
            and severity_rank(check["severity"]) <= ceiling]
