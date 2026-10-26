"""``tags`` -- tag inventory and per-tag spend.

    token-usage tags
    token-usage tags --top 5

Complements ``report --group tag`` by also listing tags that exist in the data
but carry no cost, which usually signals a tagging mistake.
"""

from __future__ import annotations

from collections import Counter

from token_usage import analytics
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import bar, human_int, human_usd


class TagsPlugin(Plugin):
    name = "tags"
    help = "List tags with their spend and usage counts."
    aliases = ["tag"]
    usage = "tags [--top <n>] [--project <p>] [--no-bars]"

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"))
        if not records:
            ctx.out("no records in scope")
            return 0

        counts = Counter(tag for record in records for tag in record.tags)
        untagged = sum(1 for record in records if not record.tags)
        grouped = analytics.by_tag(records, ctx.catalog)

        top = args.get_int("top", 0)
        items = list(grouped.items())
        if top > 0:
            items = items[:top]
        maximum = max((p["cost_usd"] for _, p in items), default=0.0)
        show_bars = not args.get_bool("no_bars")

        ctx.out(f"{len(counts)} distinct tag(s) across {len(records)} record(s)")
        if untagged:
            ctx.out(f"{untagged} record(s) carry no tag")
        ctx.out("")
        ctx.out(f"  {'tag':<18}{'records':>9}{'tokens':>13}{'cost':>12}")
        for name, payload in items:
            marker = " " + bar(payload["cost_usd"], maximum) if show_bars else ""
            ctx.out(f"  {name:<18}{counts.get(name, 0):>9}"
                    f"{human_int(payload['total_tokens']):>13}"
                    f"{human_usd(payload['cost_usd']):>12}{marker}")

        if untagged:
            ctx.warn("untagged records are invisible to per-tag reports; "
                     "consider re-recording them with --tags")
        return 0


register(TagsPlugin())
