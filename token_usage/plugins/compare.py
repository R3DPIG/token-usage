"""``compare`` -- current period versus the previous one.

    token-usage compare --period monthly
    token-usage compare --period weekly --project demo
"""

from __future__ import annotations

from token_usage import analytics, insights
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd


class ComparePlugin(Plugin):
    name = "compare"
    help = "Compare this period's spend against the previous one."
    aliases = ["diff"]
    usage = "compare [--period daily|weekly|monthly] [--project <p>] [--model <m>]"

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        period = args.get("period", "monthly")
        if period not in insights.COMPARABLE_PERIODS:
            ctx.warn(f"--period must be one of {insights.COMPARABLE_PERIODS}")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"))
        if not records:
            ctx.out("no records in scope")
            return 0

        result = insights.compare_periods(records, ctx.catalog, period=period)
        cur, prev = result["current"], result["previous"]

        ctx.out(f"{period} comparison")
        ctx.out(f"  {'':<12}{'current':>14}{'previous':>14}{'change':>12}")
        ctx.out(f"  {'records':<12}{cur['records']:>14,}{prev['records']:>14,}"
                f"{cur['records'] - prev['records']:>+12,}")
        ctx.out(f"  {'tokens':<12}{human_int(cur['total_tokens']):>14}"
                f"{human_int(prev['total_tokens']):>14}"
                f"{cur['total_tokens'] - prev['total_tokens']:>+12,}")
        ctx.out(f"  {'cost':<12}{human_usd(cur['cost_usd']):>14}"
                f"{human_usd(prev['cost_usd']):>14}"
                f"{human_usd(result['delta_usd']):>12}")

        arrow = {"up": "^", "down": "v", "flat": "="}[result["direction"]]
        ctx.out("")
        ctx.out(f"  {arrow} spend {result['direction']} {result['delta_pct']:+.1f}% "
                f"({human_usd(result['delta_usd'])})")

        if result["direction"] == "up" and result["delta_pct"] > 20:
            ctx.warn(f"spend is up {result['delta_pct']:.0f}% versus the previous {period}; "
                     "run 'anomaly' to see which days drove it")
        return 0


register(ComparePlugin())
