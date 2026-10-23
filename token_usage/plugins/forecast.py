"""``forecast`` -- project future spend from recent history.

    token-usage forecast --days 30 --period monthly

Least-squares extrapolation of daily spend; for planning, not accounting.
"""

from __future__ import annotations

from token_usage import analytics
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd


class ForecastPlugin(Plugin):
    name = "forecast"
    help = "Extrapolate future spend from daily history."
    usage = "forecast [--days <n>] [--period daily|weekly|monthly|total] [--project <p>]"

    def run(self, ctx: PluginContext) -> int:
        days = ctx.args.get_int("days", 30)
        if days <= 0:
            ctx.warn("--days must be > 0")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=ctx.args.get("project"), model=ctx.args.get("model"))
        period = ctx.args.get("period", "monthly")
        if period != "total":
            records = analytics.filter_by_period(records, period)
        if not records:
            ctx.out("no records in scope; nothing to forecast")
            return 0

        daily = analytics.by_day(records, ctx.catalog)
        projection = analytics.linear_forecast(daily, days)
        trend = analytics.trend(daily)

        ctx.out(f"forecast based on {projection['samples']} day(s) of history")
        ctx.out(f"  trend slope/day    {human_usd(projection['slope'])}")
        ctx.out(f"  last-day trend     {trend['direction']} ({trend['delta_pct']:+.1f}%)")
        ctx.out(f"  projected daily    {human_usd(projection['projected_daily'])}")
        ctx.out(f"  projected {days:>3} days {human_usd(projection['projected_total'])}")
        if days >= 30:
            ctx.out(f"  => ~{human_usd(projection['projected_daily'] * 30)} / 30 days at this pace")
        if trend["direction"] == "up" and trend["delta_pct"] > 25:
            ctx.warn(f"spend jumped {trend['delta_pct']:.0f}% day-over-day; "
                     "check for a runaway loop or a pricier model switch")
        return 0


register(ForecastPlugin())
