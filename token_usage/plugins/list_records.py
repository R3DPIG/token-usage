"""``list`` -- print stored records in a compact table.

    token-usage list --project demo --limit 20 --since 2024-05-01
"""

from __future__ import annotations

from token_usage.analytics import filter_records
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd


class ListPlugin(Plugin):
    name = "list"
    help = "List stored usage records, newest first."
    usage = ("list [--project <p>] [--model <m>] [--tag <t>] "
             "[--since <date>] [--until <date>] [--limit <n>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        records = filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            tag=args.get("tag"), since=args.get("since"), until=args.get("until"))
        records.sort(key=lambda r: r.timestamp, reverse=not args.get_bool("oldest"))

        limit = args.get_int("limit", 50)
        shown = records[:limit] if limit > 0 else records
        if not shown:
            ctx.out("no records matched the given filters")
            return 0

        header = (f"{'timestamp':<20} {'model':<24} {'in':>10} {'out':>10} "
                  f"{'cached':>10} {'cost':>10}  project")
        ctx.out(header)
        ctx.out("-" * len(header))
        total = 0.0
        for record in shown:
            cost = ctx.catalog.cost_for_record(record)
            total += cost
            ctx.out(f"{record.timestamp:<20} {record.model:<24} "
                    f"{record.input_tokens:>10,} {record.output_tokens:>10,} "
                    f"{record.cached_input_tokens:>10,} {human_usd(cost):>10}  {record.project}")
        ctx.out("-" * len(header))
        ctx.out(f"showing {len(shown)} of {len(records)} record(s); "
                f"listed cost {human_usd(total)}")
        return 0


register(ListPlugin())
