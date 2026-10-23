"""``stats`` -- descriptive statistics over the stored history.

    token-usage stats --period monthly --project demo
"""

from __future__ import annotations

from statistics import mean, median

from token_usage import analytics
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd


class StatsPlugin(Plugin):
    name = "stats"
    help = "Descriptive statistics (min/median/max, busiest day)."
    usage = "stats [--period daily|weekly|monthly|total] [--project <p>]"

    def run(self, ctx: PluginContext) -> int:
        records = analytics.filter_records(
            ctx.records(), project=ctx.args.get("project"), model=ctx.args.get("model"))
        period = ctx.args.get("period", "total")
        if period != "total":
            records = analytics.filter_by_period(records, period)
        if not records:
            ctx.out("no records in scope")
            return 0

        costs = [ctx.catalog.cost_for_record(r) for r in records]
        tokens = [r.total_tokens for r in records]
        ctx.out(f"records        {len(records)}")
        ctx.out(f"cost  min      {human_usd(min(costs))}")
        ctx.out(f"cost  median   {human_usd(median(costs))}")
        ctx.out(f"cost  mean     {human_usd(mean(costs))}")
        ctx.out(f"cost  max      {human_usd(max(costs))}")
        ctx.out(f"tokens median  {human_int(int(median(tokens)))}")
        ctx.out(f"tokens max     {human_int(max(tokens))}")

        daily = analytics.by_day(records, ctx.catalog)
        if daily:
            busiest = max(daily.items(), key=lambda kv: kv[1]["cost_usd"])
            cheapest = min(daily.items(), key=lambda kv: kv[1]["cost_usd"])
            ctx.out("")
            ctx.out(f"busiest day    {busiest[0]}  {human_usd(busiest[1]['cost_usd'])}")
            ctx.out(f"cheapest day   {cheapest[0]}  {human_usd(cheapest[1]['cost_usd'])}")
            ctx.out(f"active days    {len(daily)}")

        projects = analytics.by_project(records, ctx.catalog)
        if projects:
            ctx.out("")
            ctx.out("per project:")
            for name, payload in projects.items():
                ctx.out(f"  {name:<20} {human_int(payload['total_tokens']):>12} tok  "
                        f"{human_usd(payload['cost_usd'])}")
        return 0


register(StatsPlugin())
