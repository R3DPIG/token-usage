"""``report`` -- aggregated spending reports with ASCII bars.

    token-usage report --group day --period monthly --top 5
"""

from __future__ import annotations

from token_usage import analytics
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import bar, human_int, human_usd


class ReportPlugin(Plugin):
    name = "report"
    help = "Aggregate usage by model/project/tag/day/week/month."
    usage = ("report [--group model|project|tag|day|week|month] "
             "[--period daily|weekly|monthly|total] [--since <date>] [--until <date>] "
             "[--project <p>] [--top <n>] [--no-bars]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        dimension = args.get("group", "model")
        if dimension not in analytics.GROUPINGS:
            ctx.warn(f"unknown --group {dimension!r}; expected one of {sorted(analytics.GROUPINGS)}")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            tag=args.get("tag"), since=args.get("since"), until=args.get("until"))
        period = args.get("period")
        if period and period != "total":
            records = analytics.filter_by_period(records, period)
        if not records:
            ctx.out("no records in scope")
            return 0

        overall = analytics.totals(records, ctx.catalog)
        ctx.out("usage summary")
        ctx.out(f"  records          {human_int(overall['records'])}")
        ctx.out(f"  input tokens     {human_int(overall['input_tokens'])}")
        ctx.out(f"  output tokens    {human_int(overall['output_tokens'])}")
        ctx.out(f"  total tokens     {human_int(overall['total_tokens'])}")
        ctx.out(f"  models used      {overall['models_used']}")
        ctx.out(f"  average / call   {human_usd(analytics.average_cost_per_call(records, ctx.catalog))}")
        ctx.out(f"  total cost       {human_usd(overall['cost_usd'])}")

        grouped = analytics.group(records, dimension, ctx.catalog)
        top = args.get_int("top", 0)
        items = analytics.top_n(grouped, top) if top > 0 else list(grouped.items())
        maximum = max((p["cost_usd"] for _, p in items), default=0.0)
        show_bars = not args.get_bool("no_bars")

        ctx.out("")
        ctx.out(f"grouped by {dimension}:")
        for name, payload in items:
            marker = " " + bar(payload["cost_usd"], maximum) if show_bars else ""
            ctx.out(f"  {name:<22} {human_int(payload['total_tokens']):>12} tok  "
                    f"{human_usd(payload['cost_usd']):>11}  "
                    f"({payload['records']} rec){marker}")

        savings = analytics.cache_savings(records, ctx.catalog)
        if savings["cached_input_tokens"]:
            ctx.out("")
            ctx.out(f"cache: {human_int(savings['cached_input_tokens'])} tokens served from cache, "
                    f"estimated savings {human_usd(savings['saved_usd'])}")
        return 0


register(ReportPlugin())
