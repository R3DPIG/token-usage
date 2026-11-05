"""Spend drivers: not just *how much* changed, but *why*.

:mod:`token_usage.insights` compares two periods and reports a delta. That tells
you the bill moved; it does not tell you whether you are being charged more, or
simply doing more work. Those two cases call for opposite responses, so this
module decomposes the delta instead of restating it.

The decomposition is exact. With ``q`` tokens, ``p`` blended unit price and
``Q = sum(q)``::

    cost = Q * mean_price          where mean_price = sum(share * p)

so a delta splits into four terms that always re-add to the observed change:

===============  =========================================================================
term             meaning
===============  =========================================================================
volume           ``(Q1 - Q0) * mean_price0`` -- you used more or fewer tokens
mix              ``Q0 * (mean_price_at_new_mix - mean_price0)`` -- you leaned on pricier models
unit price       ``Q0 * (mean_price1 - mean_price_at_new_mix)`` -- each token got cheaper/dearer
interaction      ``(Q1 - Q0) * (mean_price1 - mean_price0)`` -- the two moved together
===============  =========================================================================

The interaction term is kept rather than folded into the others, because
arbitrarily assigning it would make the report look tidier than the data allows.

``unit_price`` is a **blended** price (cost per token), not the published list
price. It moves when the cached share or the input/output split changes -- and
that is worth knowing, so it is reported as its own driver rather than hidden.

A model that appears in only one period has no observable price history; it
inherits its other-period price. That substitution assumption is stated in the
output (``assumed_price``) instead of being silently applied.

Pure functions only: records in, plain dicts out.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from token_usage import analytics
from token_usage.models import UsageRecord
from token_usage.pricing import PricingCatalog
from token_usage.utils import parse_iso, period_start, round_usd

DRIVER_ORDER = ("volume", "mix", "unit_price", "interaction")
COMPARABLE = ("daily", "weekly", "monthly")
PER_MILLION = 1_000_000.0


def split_periods(records: Iterable[UsageRecord], period: str = "monthly",
                  now: Optional[datetime] = None
                  ) -> Tuple[List[UsageRecord], List[UsageRecord]]:
    """Split records into ``(current, previous)`` windows of equal length.

    The previous window mirrors the *elapsed* part of the current period, so a
    comparison made mid-month is not half a month against a full one. This is
    the same rule ``insights.compare_periods`` uses; it lives here too because
    this module needs the records, not just their totals.
    """
    if period not in COMPARABLE:
        raise ValueError(f"period must be one of {COMPARABLE}, got {period!r}")
    now = now or datetime.now(timezone.utc)
    current_start = period_start(period, now=now)
    previous_start = current_start - (now - current_start)

    records = list(records)
    current = [r for r in records if parse_iso(r.timestamp) >= current_start]
    previous = [r for r in records
                if previous_start <= parse_iso(r.timestamp) < current_start]
    return current, previous


def _profile(records: Iterable[UsageRecord],
             catalog: PricingCatalog) -> Dict[str, Dict]:
    """Per-model tokens and cost, with no pricing assumptions yet."""
    return {name: {"tokens": payload["total_tokens"],
                   "cost_usd": payload["cost_usd"],
                   "records": payload["records"]}
            for name, payload in analytics.by_model(records, catalog).items()}


def unit_prices(current: Dict[str, Dict], previous: Dict[str, Dict]
                ) -> Dict[str, Dict]:
    """Blended price per 1M tokens for every model in either period.

    A model missing from one side inherits the other side's price and is flagged
    ``assumed`` -- without that flag a model that simply appeared would look
    like a price change, which would be a lie told with real numbers.
    """
    result: Dict[str, Dict] = {}
    for name in set(current) | set(previous):
        cur, prev = current.get(name), previous.get(name)
        cur_price = (cur["cost_usd"] / cur["tokens"] * PER_MILLION
                     if cur and cur["tokens"] else None)
        prev_price = (prev["cost_usd"] / prev["tokens"] * PER_MILLION
                      if prev and prev["tokens"] else None)
        assumed = cur_price is None or prev_price is None
        price = cur_price if cur_price is not None else prev_price
        result[name] = {"price_per_1m": round(price or 0.0, 6),
                        "assumed_price": assumed}
    return result


def _both_sides(records: Iterable[UsageRecord], catalog: PricingCatalog,
                period: str, now: Optional[datetime]
                ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict]:
    """Profiles for both windows, the price table, and record counts."""
    current_records, previous_records = split_periods(records, period, now=now)
    cur = _profile(current_records, catalog)
    prev = _profile(previous_records, catalog)
    counts = {"current": len(current_records), "previous": len(previous_records)}
    return cur, prev, unit_prices(cur, prev), counts


def decompose(records: Iterable[UsageRecord], catalog: PricingCatalog,
              period: str = "monthly", now: Optional[datetime] = None) -> Dict:
    """Split the period-over-period cost change into volume/mix/price.

    Returns the four terms plus the observed delta, so a caller can verify the
    identity rather than trust it. ``balanced`` is that check.
    """
    cur, prev, prices, counts = _both_sides(records, catalog, period, now)

    q0 = sum(row["tokens"] for row in prev.values()) / PER_MILLION
    q1 = sum(row["tokens"] for row in cur.values()) / PER_MILLION
    c0 = sum(row["cost_usd"] for row in prev.values())
    c1 = sum(row["cost_usd"] for row in cur.values())
    delta = c1 - c0

    mean0 = c0 / q0 if q0 else 0.0            # mean blended price, previous
    mean1 = c1 / q1 if q1 else 0.0            # mean blended price, current

    # Mean price the current mix would have cost at previous unit prices.
    mean_at_new_mix = 0.0
    if q1:
        for name, row in cur.items():
            share1 = (row["tokens"] / PER_MILLION) / q1
            mean_at_new_mix += share1 * prices[name]["price_per_1m"]

    volume = (q1 - q0) * mean0
    mix = q0 * (mean_at_new_mix - mean0)
    price = q0 * (mean1 - mean_at_new_mix)
    interaction = (q1 - q0) * (mean1 - mean0)
    accounted = volume + mix + price + interaction

    # With no history there is nothing to decompose *against*: q0 and mean0 are
    # both zero, so the identity pushes the whole delta into ``interaction``.
    # That is arithmetically correct but reads as "volume and price moved
    # together", which is false -- there was simply no baseline. Flag it so the
    # caller can say so instead of narrating a comparison that never happened.
    no_baseline = q0 == 0 or q1 == 0

    return {
        "period": period,
        "delta_usd": round_usd(delta),
        "current_usd": round_usd(c1),
        "previous_usd": round_usd(c0),
        "current_tokens": int(q1 * PER_MILLION),
        "previous_tokens": int(q0 * PER_MILLION),
        "mean_price_current": round(mean1, 6),
        "mean_price_previous": round(mean0, 6),
        "drivers": {
            "volume": round_usd(volume),
            "mix": round_usd(mix),
            "unit_price": round_usd(price),
            "interaction": round_usd(interaction),
        },
        "accounted_usd": round_usd(accounted),
        "drift_usd": round_usd(delta - accounted),
        "balanced": abs(delta - accounted) < 1e-6,
        "no_baseline": no_baseline,
        "assumed_prices": sorted(name for name, row in prices.items()
                                 if row["assumed_price"]),
        "records": counts,
    }


def dominant_driver(drivers: Dict[str, float]) -> str:
    """The driver with the largest absolute contribution.

    Returns ``"flat"`` when every term rounds to zero -- there is no story to
    tell about a change that did not happen.
    """
    if all(abs(value) < 1e-9 for value in drivers.values()):
        return "flat"
    return max(DRIVER_ORDER, key=lambda name: abs(drivers.get(name, 0.0)))


def model_contributions(records: Iterable[UsageRecord],
                        catalog: PricingCatalog, period: str = "monthly",
                        now: Optional[datetime] = None,
                        limit: int = 10) -> List[Dict]:
    """Which models moved the bill, largest absolute change first.

    Each row splits its own delta into a volume part and a price part using the
    same identity as :func:`decompose`, so the rows are additive and explain the
    total instead of merely correlating with it.
    """
    cur, prev, prices, _ = _both_sides(records, catalog, period, now)

    rows = []
    for name in set(cur) | set(prev):
        row_cur, row_prev = cur.get(name), prev.get(name)
        q1 = (row_cur["tokens"] if row_cur else 0) / PER_MILLION
        q0 = (row_prev["tokens"] if row_prev else 0) / PER_MILLION
        c1 = row_cur["cost_usd"] if row_cur else 0.0
        c0 = row_prev["cost_usd"] if row_prev else 0.0
        price = prices[name]["price_per_1m"]
        volume = (q1 - q0) * price
        price_effect = (c1 - c0) - volume
        rows.append({
            "model": name,
            "current_usd": round_usd(c1),
            "previous_usd": round_usd(c0),
            "delta_usd": round_usd(c1 - c0),
            "volume_effect_usd": round_usd(volume),
            "price_effect_usd": round_usd(price_effect),
            "current_tokens": int(q1 * PER_MILLION),
            "previous_tokens": int(q0 * PER_MILLION),
            "status": ("new" if not row_prev else
                       "dropped" if not row_cur else "ongoing"),
            "assumed_price": prices[name]["assumed_price"],
        })
    rows.sort(key=lambda item: abs(item["delta_usd"]), reverse=True)
    return rows[:limit]


def mix_shift(records: Iterable[UsageRecord], catalog: PricingCatalog,
              period: str = "monthly", now: Optional[datetime] = None) -> List[Dict]:
    """How each model's share of total tokens changed.

    Share is measured in **tokens**, not dollars, on purpose: dollars already
    appear in :func:`model_contributions`, and using tokens here keeps the mix
    metric from double-counting the price change it is meant to isolate.
    """
    cur, prev, _, _ = _both_sides(records, catalog, period, now)
    q1_total = sum(row["tokens"] for row in cur.values())
    q0_total = sum(row["tokens"] for row in prev.values())

    rows = []
    for name in set(cur) | set(prev):
        q1 = cur.get(name, {}).get("tokens", 0)
        q0 = prev.get(name, {}).get("tokens", 0)
        share1 = q1 / q1_total if q1_total else 0.0
        share0 = q0 / q0_total if q0_total else 0.0
        rows.append({"model": name,
                     "share_current": round(share1, 4),
                     "share_previous": round(share0, 4),
                     "share_delta": round(share1 - share0, 4)})
    rows.sort(key=lambda item: abs(item["share_delta"]), reverse=True)
    return rows


def explain(data: Dict) -> List[str]:
    """Turn a decomposition into ordered, readable sentences.

    Ordered by absolute contribution so the first line is always the one worth
    reading. Zero-value drivers are omitted rather than printed as "$0.0000",
    which would pad the output without informing anyone.
    """
    drivers = data["drivers"]
    delta = data["delta_usd"]
    if abs(delta) < 1e-9:
        return ["spend is unchanged period over period"]

    lines = []
    has_current = data["current_usd"] > 0
    if data["no_baseline"]:
        # Say what actually happened instead of narrating a four-way split that
        # only looks meaningful because one side of the arithmetic is zero.
        state = "new spend" if has_current else "spend disappeared"
        lines.append(f"{state}: ${data['previous_usd']:.4f} -> "
                     f"${data['current_usd']:.4f} -- no comparable baseline, "
                     "so this cannot be decomposed")
        return lines

    direction = "up" if delta > 0 else "down"
    lines.append(f"spend is {direction} ${abs(delta):.4f} "
                 f"(${data['previous_usd']:.4f} -> ${data['current_usd']:.4f})")

    ordered = sorted(DRIVER_ORDER, key=lambda n: abs(drivers.get(n, 0.0)),
                     reverse=True)
    for name in ordered:
        value = drivers.get(name, 0.0)
        if abs(value) < 1e-9:
            continue
        verb = "added" if value > 0 else "removed"
        if name == "volume":
            lines.append(f"volume {verb} ${abs(value):.4f}: "
                         f"{data['previous_tokens']:,} -> "
                         f"{data['current_tokens']:,} tokens")
        elif name == "mix":
            lines.append(f"model mix {verb} ${abs(value):.4f}: usage shifted "
                         "toward different models")
        elif name == "unit_price":
            lines.append(f"blended unit price {verb} ${abs(value):.4f}: "
                         f"${data['mean_price_previous']:.4f} -> "
                         f"${data['mean_price_current']:.4f} per 1M tokens")
        else:
            lines.append(f"interaction {verb} ${abs(value):.4f}: volume and "
                         "price moved together")

    if data["assumed_prices"]:
        lines.append(f"{len(data['assumed_prices'])} model(s) appeared or "
                     "vanished, so their unit price was assumed: "
                     + ", ".join(data["assumed_prices"][:3]))
    return lines


def summarize(records: Iterable[UsageRecord], catalog: PricingCatalog,
              period: str = "monthly", now: Optional[datetime] = None) -> Dict:
    """One-shot view used by the plugin's default output."""
    data = decompose(records, catalog, period=period, now=now)
    data["dominant"] = dominant_driver(data["drivers"])
    data["lines"] = explain(data)
    data["models"] = model_contributions(records, catalog, period=period, now=now)
    return data
