"""Backtesting: how wrong has the forecast actually been?

``forecast`` fits a line and prints a number. ``plan`` divides the remaining
budget by the remaining days. Both are used to make commitments, and neither
says whether the number has ever been right. This module answers that by
running the forecast over history: for each past cutoff it predicts what the
next window will cost, then compares the prediction with what was actually
spent.

The method is **walk-forward, expanding window**:

1. Pick every eligible cutoff date in history.
2. For each, fit on data strictly *before* the cutoff.
3. Predict the window after it.
4. Score the prediction against the realised cost.

Three honesty rules shape the output:

* **No look-ahead.** The fit at every cutoff sees only earlier records. A
  backtest that lets the model peek at the answers measures nothing, so the
  window boundaries are enforced by construction rather than by convention.
* **A small sample is reported as small.** Error statistics over three
  predictions are noise; the result carries the sample size and a
  ``reliable`` flag, and the CLI refuses to call a thin result a verdict.
* **Systematic bias and random error are different faults.** A forecast that
  is always 40% low is fixable by a multiplier; one that is sometimes high and
  sometimes low needs better inputs. Mean error (bias) and mean absolute error
  are computed separately for exactly that reason.

Pure functions only: records + catalog + clock in, plain dicts out. Every
metric that cannot be computed is ``None`` with a reason, never a zero that
would read as a perfect score.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from token_usage import analytics
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import parse_iso, round_usd

# Fewer predictions than this and the error statistics are anecdote, not evidence.
MIN_SAMPLES = 4
# A backtest needs at least this many days of history before the first cutoff.
MIN_HISTORY_DAYS = 3


def _daily_series(records: Iterable[UsageRecord],
                  catalog: PricingCatalog) -> List[Dict]:
    """Per-day cost, chronological, with no zero-filled gaps.

    Gaps stay out on purpose: this module scores predictions against days that
    actually exist, and inventing calendar days the workspace never observed
    would silently penalise the forecast for not predicting silence.
    """
    return [{"day": day, "cost_usd": payload["cost_usd"]}
            for day, payload in analytics.by_day(records, catalog).items()]


def _fit(days: List[Dict]) -> Optional[Dict]:
    """Least-squares fit over one window, reusing the shipped forecast.

    Delegating to :func:`token_usage.analytics.linear_forecast` keeps the
    backtest measuring the product's real predictor instead of a private copy
    that could drift away from it.
    """
    if len(days) < 2:
        return None
    daily = {row["day"]: {"cost_usd": row["cost_usd"]} for row in days}
    return analytics.linear_forecast(daily, 1)


def backtest(records: Iterable[UsageRecord], catalog: PricingCatalog,
             window_days: int = 7, min_history_days: int = MIN_HISTORY_DAYS,
             now: Optional[datetime] = None) -> Dict:
    """Walk the forecast forward through history and score every prediction.

    At each cutoff the model is fitted on the ``min_history_days``+ days before
    it and asked to predict the total for the following ``window_days``. The
    realised total for that window is then measured from the records, so the
    prediction is always graded on data it never saw.
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    if min_history_days < 2:
        raise ValueError("min_history_days must be >= 2")

    now = now or datetime.now(timezone.utc)
    series = _daily_series(records, catalog)
    if len(series) < min_history_days + 1:
        return {"samples": [], "metrics": None, "window_days": window_days,
                "reason": (f"need at least {min_history_days + 1} days of history "
                           f"for one prediction; have {len(series)}"),
                "reliable": False}

    predictions: List[Dict] = []
    for index in range(min_history_days, len(series)):
        history = series[:index]                      # strictly earlier days
        fit = _fit(history)
        if fit is None:
            continue
        cutoff = parse_iso(f"{history[-1]['day']}T00:00:00")
        horizon_end = cutoff + timedelta(days=window_days)
        forward = [row for row in series[index:]
                   if parse_iso(f"{row['day']}T00:00:00") <= horizon_end]
        # A partial window would compare a full-horizon prediction against a
        # truncated actual, which always looks like over-prediction.
        if len(forward) < window_days:
            continue
        predicted = round_usd(fit["projected_daily"] * window_days)
        actual = round_usd(sum(row["cost_usd"] for row in forward[:window_days]))
        predictions.append({
            "cutoff": history[-1]["day"],
            "predicted_usd": predicted,
            "actual_usd": actual,
            "error_usd": round_usd(predicted - actual),
            "abs_error_usd": round_usd(abs(predicted - actual)),
            "error_pct": (round((predicted - actual) / actual * 100.0, 2)
                          if actual else None),
        })

    return {"samples": predictions, "window_days": window_days,
            "metrics": metrics(predictions), "reliable": len(predictions) >= MIN_SAMPLES,
            "reason": "" if predictions else "no prediction could be graded"}


def metrics(predictions: List[Dict]) -> Optional[Dict]:
    """Error statistics over a prediction list; ``None`` when there are none.

    ``bias_pct`` is signed (positive means the forecast ran high) and is a
    different diagnosis from ``mape``: a consistently wrong forecast can be
    corrected, an erratically wrong one cannot be corrected by a constant.
    """
    if not predictions:
        return None
    signed = [row["error_usd"] for row in predictions]
    absolute = [row["abs_error_usd"] for row in predictions]
    pct_rows = [row for row in predictions if row["error_pct"] is not None]
    n = len(predictions)
    bias = sum(signed) / n
    mae = sum(absolute) / n
    actual_total = sum(row["actual_usd"] for row in predictions)
    return {
        "count": n,
        "bias_usd": round_usd(bias),
        "bias_pct": (round(sum(row["error_pct"] for row in pct_rows)
                           / len(pct_rows), 2) if pct_rows else None),
        "mae_usd": round_usd(mae),
        "mape": (round(sum(abs(row["error_pct"]) for row in pct_rows)
                       / len(pct_rows), 2) if pct_rows else None),
        "worst_abs_usd": round_usd(max(absolute)),
        "total_error_usd": round_usd(sum(signed)),
        "actual_total_usd": round_usd(actual_total),
        "coverage": (round(sum(1 for row in predictions if row["error_pct"] is not None)
                           / n, 4)),
    }


def accuracy_grade(metrics_result: Optional[Dict]) -> str:
    """Human verdict on MAPE, refusing to grade a result with no percentage.

    Thresholds are deliberately coarse: an LLM spend forecast is driven by
    workload, and pretending to distinguish 12% from 15% error would imply a
    precision the method does not have.
    """
    if not metrics_result or metrics_result.get("mape") is None:
        return "not evaluated"
    mape = metrics_result["mape"]
    if mape <= 10.0:
        return "close"
    if mape <= 25.0:
        return "usable"
    if mape <= 50.0:
        return "loose"
    return "unreliable"


def bias_note(metrics_result: Optional[Dict]) -> Dict:
    """Characterise the error as bias, noise, or neither.

    Returns ``kind`` -- ``bias``, ``noise``, ``both`` or ``clean`` -- plus a
    one-line explanation. Distinguishing these two matters because only one of
    them can be fixed by rescaling the forecast.
    """
    if not metrics_result or metrics_result.get("mape") is None:
        return {"kind": "unknown",
                "note": "no percentage errors, so bias and noise cannot be told apart"}
    bias = abs(metrics_result["bias_pct"])
    mape = metrics_result["mape"]
    # When the average miss has a clear direction it is bias; when the signed
    # average is much smaller than the absolute average the errors cancel out.
    directional = bias >= 10.0 and bias >= mape * 0.5
    noisy = metrics_result["mape"] >= 25.0
    if directional and noisy:
        return {"kind": "both",
                "note": "the forecast is both biased and erratic; "
                        "fixing the multiplier alone will not be enough"}
    if directional:
        direction = "high" if metrics_result["bias_pct"] > 0 else "low"
        return {"kind": "bias",
                "note": (f"the forecast runs consistently {direction} by about "
                         f"{bias:.1f}%; a correction factor would help")}
    if noisy:
        return {"kind": "noise",
                "note": "errors cancel out on average but individual calls miss "
                        "widely; the inputs carry more noise than the model can absorb"}
    return {"kind": "clean", "note": "errors are small and do not lean one way"}


def worst_cases(result: Dict, limit: int = 5) -> List[Dict]:
    """The largest absolute misses, worst first."""
    samples = sorted(result["samples"],
                     key=lambda row: row["abs_error_usd"], reverse=True)
    return samples[:limit]


def summarize(records: Iterable[UsageRecord], catalog: PricingCatalog,
              window_days: int = 7, now: Optional[datetime] = None) -> Dict:
    """One-shot backtest verdict: metrics, grade, bias character and caveats."""
    result = backtest(records, catalog, window_days=window_days, now=now)
    metrics_result = result["metrics"]
    grade = accuracy_grade(metrics_result)
    warnings: List[str] = []
    if not result["samples"]:
        warnings.append(result["reason"] or "no predictions to grade")
    elif not result["reliable"]:
        warnings.append(
            f"only {len(result['samples'])} prediction(s); "
            f"treat this as a sanity check, not a verdict")
    if metrics_result and metrics_result["coverage"] < 1.0:
        warnings.append(
            f"{metrics_result['coverage'] * 100:.0f}% of predictions had a "
            "non-zero actual; the rest had nothing to grade against")
    return {**result, "grade": grade, "bias": bias_note(metrics_result),
            "warnings": warnings}
