"""Peer comparison: how does this workspace look next to others?

:mod:`token_usage.analytics` totals one dataset. :mod:`token_usage.insights`
compares one dataset against its own past. :mod:`token_usage.merge` combines
datasets but never contrasts them. None of them answers the question a team lead
actually asks when several teams share a bill: *are we expensive, or just busy?*

Raw totals cannot answer that -- whoever uses the most tokens always spends the
most, which says nothing about efficiency. So every peer is normalised before
anything is compared:

``cost_per_1m``
    blended USD per million tokens. The headline unit cost.
``cost_per_call``
    average cost of one request.
``output_ratio``
    share of tokens that were generated rather than read.
``cache_ratio``
    share of input tokens served from cache.

Rules that keep the comparison honest:

* **A tiny peer is excluded from the benchmark.** One call at a lucky cache hit
  would otherwise define "best in class" for everyone else. Peers below the token
  floor are still listed with their metrics, flagged ``eligible: False``.
* **Unlike workloads are flagged, not silently compared.** A peer doing mostly
  embeddings has an output ratio near zero and a naturally lower unit cost. When
  output ratios diverge, the module says so instead of declaring that peer the
  winner -- a unit-cost ranking across different workloads is a category error.
* **Savings are hypothetical and labelled as such.** ``savings_vs_best`` reprices
  each peer's *own* tokens at the best observed unit cost. The arithmetic is
  exact; the premise (that the workload would be served equally well) is stated
  as an assumption rather than presented as a fact.
* **Outliers use median and MAD**, matching :mod:`token_usage.quality`, because a
  single extreme peer inflates mean and stdev enough to hide itself.

Pure functions only: named record sets in, plain dicts out.
"""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Optional

from token_usage import analytics
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import round_usd

# Metrics a peer can be ranked by. ``cost_usd`` and ``tokens`` are raw scale and
# are offered only because people ask for them, not because they measure anything.
METRICS = ("cost_per_1m", "cost_per_call", "output_ratio", "cache_ratio",
           "cost_usd", "tokens")
# Lower is better for spending metrics; higher is better for ratio metrics.
LOWER_IS_BETTER = {"cost_per_1m": True, "cost_per_call": True, "cost_usd": True,
                   "tokens": True, "output_ratio": False, "cache_ratio": False}

# Below this token volume a peer's unit cost is noise, not a benchmark.
MIN_BENCHMARK_TOKENS = 10_000
# Output ratios further apart than this mean the peers are doing different work.
WORKLOAD_GAP = 0.15
MAD_SCALE = 1.4826


def profile(name: str, records: Iterable[UsageRecord], catalog: PricingCatalog,
            min_tokens: int = MIN_BENCHMARK_TOKENS) -> Dict:
    """One peer's normalised metrics.

    ``eligible`` is False when the peer has too few tokens to benchmark against,
    or no tokens at all. It is kept in the list rather than dropped so the reader
    can see that a peer exists and was deliberately not used as a yardstick.
    """
    totals = analytics.totals(records, catalog)
    tokens = totals["total_tokens"]
    cost = totals["cost_usd"]
    calls = totals["records"]
    return {
        "peer": name,
        "cost_usd": cost,
        "tokens": tokens,
        "calls": calls,
        "cost_per_1m": round_usd(cost / tokens * 1_000_000.0) if tokens else 0.0,
        "cost_per_call": round_usd(cost / calls) if calls else 0.0,
        "output_ratio": (round(totals["output_tokens"] / tokens, 4)
                         if tokens else 0.0),
        "cache_ratio": (round(totals["cached_input_tokens"] / totals["input_tokens"], 4)
                        if totals["input_tokens"] else 0.0),
        "eligible": tokens >= min_tokens,
    }


def build_profiles(sources: Dict[str, Iterable[UsageRecord]],
                   catalog: PricingCatalog,
                   min_tokens: int = MIN_BENCHMARK_TOKENS) -> List[Dict]:
    """Profile every named source, largest spend first."""
    rows = [profile(name, records, catalog, min_tokens=min_tokens)
            for name, records in sources.items()]
    return sorted(rows, key=lambda row: row["cost_usd"], reverse=True)


def eligible(profiles: Iterable[Dict]) -> List[Dict]:
    """Only the peers usable as a benchmark."""
    return [row for row in profiles if row["eligible"]]


def rank(profiles: Iterable[Dict], metric: str) -> List[Dict]:
    """Order peers by ``metric`` and attach ``rank`` and ``percentile``.

    Ties share a rank (competition ranking). The percentile is the share of peers
    this one beats, so ``1.0`` is always best and ``0.0`` always worst, whichever
    direction the metric runs -- that keeps the number readable without a lookup.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    rows = list(profiles)
    if not rows:
        return []
    # Sign-flipping lets a single comparison direction cover both cases.
    sign = 1.0 if LOWER_IS_BETTER[metric] else -1.0
    scored = [(row, sign * row[metric]) for row in rows]
    span = len(rows) - 1
    result = [{**row, "rank": 1 + sum(1 for _, other in scored if other < score),
               "percentile": (round(sum(1 for _, other in scored if other > score)
                                    / span, 4) if span else 1.0)}
              for row, score in scored]
    return sorted(result, key=lambda item: item["rank"])


def spread(profiles: Iterable[Dict], metric: str) -> Dict:
    """min / median / max and the ratio between them for one metric.

    ``multiple`` is the headline number: "the most expensive peer pays 4.2x what
    the cheapest does" is actionable in a way that two separate numbers are not.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    rows = list(profiles)
    values = [row[metric] for row in rows]
    if not values:
        return {"metric": metric, "count": 0, "min": 0.0, "median": 0.0,
                "max": 0.0, "multiple": 0.0, "cv": 0.0, "best": None, "worst": None}
    ranked = rank(rows, metric)
    low, high = min(values), max(values)
    mean = statistics.mean(values)
    return {
        "metric": metric,
        "count": len(values),
        "min": low,
        "median": statistics.median(values),
        "max": high,
        "multiple": round(high / low, 2) if low > 0 else 0.0,
        # Coefficient of variation: 0 means every peer is identical, so a wide
        # CV is the signal that something is worth investigating.
        "cv": round(statistics.pstdev(values) / mean, 4) if mean else 0.0,
        "best": ranked[0]["peer"],
        "worst": ranked[-1]["peer"],
    }


def savings_vs_best(profiles: Iterable[Dict], metric: str = "cost_per_1m") -> List[Dict]:
    """Reprice each eligible peer's tokens at the best observed unit cost.

    The arithmetic is exact -- tokens are the peer's own -- but the premise is an
    assumption: that the cheaper peer's rate is reachable for the same workload.
    ``note`` carries that caveat into the output rather than leaving the reader to
    infer it.
    """
    rows = list(profiles)
    pool = eligible(rows) or rows
    if not pool:
        return []
    if metric not in ("cost_per_1m", "cost_per_call"):
        raise ValueError("savings_vs_best supports cost_per_1m and cost_per_call")
    best_row = rank(pool, metric)[0]
    best_value = best_row[metric]
    results = []
    for row in rows:
        repriced = (row["tokens"] / 1_000_000.0 * best_value
                    if metric == "cost_per_1m" else row["calls"] * best_value)
        # Clamped once and reused: a peer already at the benchmark has a raw
        # saving of -0.0, which would otherwise print as "-0.0%".
        saving = max(0.0, row["cost_usd"] - repriced) + 0.0
        results.append({
            "peer": row["peer"],
            "eligible": row["eligible"],
            "current_usd": row["cost_usd"],
            "repriced_usd": round_usd(repriced),
            "savings_usd": round_usd(saving),
            "savings_pct": (round(saving / row["cost_usd"] * 100.0, 2)
                            if row["cost_usd"] > 0 else 0.0),
            "benchmark": best_row["peer"],
            "benchmark_value": best_value,
            "note": ("assumes this peer's workload can be served at the "
                     f"benchmark rate set by {best_row['peer']}"),
        })
    results.sort(key=lambda item: item["savings_usd"], reverse=True)
    return results


def outliers(profiles: Iterable[Dict], metric: str, sigma: float = 3.0,
             ratio_multiple: float = 5.0) -> List[Dict]:
    """Peers whose ``metric`` sits far from the group's typical value.

    Median and MAD rather than mean and stdev: with a handful of peers a single
    extreme one moves the mean and the stdev by about the same amount, so the
    naive z-score lands above the outlier and reports a clean group.

    When more than half the peers share one value the MAD is zero and no z-score
    exists -- which is common here, since several teams usually pay the same
    model price. Returning ``[]`` in that case would report "no outlier" for a
    group where one peer is 18x the rest, so the check falls back to a ratio test
    against the median and labels the basis it used.
    """
    rows = list(profiles)
    if len(rows) < 3:
        return []
    values = [float(row[metric]) for row in rows]
    median = statistics.median(values)
    scale = statistics.median([abs(v - median) for v in values]) * MAD_SCALE

    flagged = []
    for row in rows:
        if scale > 0:
            score = (row[metric] - median) / scale
            hit = abs(score) >= sigma
            basis, high = "mad", score > 0
        else:
            if median <= 0:
                continue
            score = row[metric] / median
            hit = score >= ratio_multiple or score <= 1.0 / ratio_multiple
            basis, high = "ratio", score > 1
        if hit:
            flagged.append({"peer": row["peer"], "metric": metric,
                            "value": row[metric], "median": median,
                            "score": round(score, 2), "basis": basis,
                            "direction": "high" if high else "low"})
    return sorted(flagged, key=lambda item: abs(item["score"]), reverse=True)


def workload_warning(profiles: Iterable[Dict]) -> Optional[str]:
    """Warn when peers are not doing comparable work.

    A peer that mostly produces embeddings has an output ratio near zero and a
    much lower unit cost. Ranking it above a chat-heavy peer is a category error,
    so the caller is told this before reading any ranking.
    """
    pool = eligible(profiles)
    if len(pool) < 2:
        return None
    ratios = [row["output_ratio"] for row in pool]
    if max(ratios) - min(ratios) <= WORKLOAD_GAP:
        return None
    return (f"output ratios span {min(ratios) * 100:.0f}%-{max(ratios) * 100:.0f}%, "
            "so these peers are not doing the same kind of work; unit-cost "
            "rankings across them are not like-for-like")


def efficiency_note(profiles: Iterable[Dict]) -> Optional[str]:
    """Explain what the unit-cost spread does and does not tell you.

    A wide spread has two possible causes -- different prices, or different work
    -- and the reader should be pointed at both rather than left to assume the
    first.
    """
    pool = eligible(profiles)
    if len(pool) < 2:
        return None
    data = spread(pool, "cost_per_1m")
    if data["multiple"] <= 1.2:
        return "unit costs are within 20% of each other; little is explained by price alone"
    return (f"the priciest peer pays {data['multiple']}x the cheapest per token; "
            "check model choice and cache hit rate before assuming it is waste")


def summarize(sources: Dict[str, Iterable[UsageRecord]],
              catalog: PricingCatalog,
              min_tokens: int = MIN_BENCHMARK_TOKENS) -> Dict:
    """One-shot comparison used by the plugin's default view.

    Only the unit-cost spread and the warnings are precomputed here; ranking,
    per-metric spreads and savings are computed by whichever view asks for them,
    because each depends on a metric the caller chooses at display time.
    """
    profiles = build_profiles(sources, catalog, min_tokens=min_tokens)
    pool = eligible(profiles)
    warnings = [text for text in (workload_warning(profiles),
                                  efficiency_note(profiles)) if text]
    return {
        "profiles": profiles,
        "eligible_count": len(pool),
        "total_peers": len(profiles),
        "total_usd": round_usd(sum(row["cost_usd"] for row in profiles)),
        "total_tokens": sum(row["tokens"] for row in profiles),
        "unit_cost": spread(pool or profiles, "cost_per_1m"),
        "warnings": warnings,
        "comparable": bool(pool) and workload_warning(profiles) is None,
        "min_tokens": min_tokens,
    }
