"""``prune`` -- delete records matching a filter.

    token-usage prune --before 2024-01-01              # preview
    token-usage prune --project scratch --apply
    token-usage prune --model gpt-3.5-turbo --apply

Refuses to run with no criteria at all, so a typo cannot wipe the history.
"""

from __future__ import annotations

from token_usage import maintenance
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd

_CRITERIA = ("project", "model", "tag", "before", "after")


class PrunePlugin(Plugin):
    name = "prune"
    help = "Delete records matching a filter (dry-run by default)."
    aliases = ["delete", "rm"]
    usage = ("prune [--project <p>] [--model <m>] [--tag <t>] "
             "[--before <date>] [--after <date>] [--apply]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        if not any(args.get(key) for key in _CRITERIA):
            ctx.warn("refusing to prune without any criteria; "
                     f"supply at least one of {list(_CRITERIA)}")
            ctx.out(self.usage)
            return 2

        records = ctx.records()
        if not records:
            ctx.out("no records found")
            return 0

        kept, removed = maintenance.prune(
            records, project=args.get("project"), model=args.get("model"),
            tag=args.get("tag"), before=args.get("before"), after=args.get("after"))

        ctx.out(f"scanned {human_int(len(records))} record(s)")
        ctx.out(f"matched {human_int(len(removed))}, keeping {human_int(len(kept))}")
        if not removed:
            ctx.out("nothing to do")
            return 0

        value = maintenance.estimate_value(removed, ctx.catalog)
        ctx.out(f"removing {human_int(value['tokens'])} tokens "
                f"worth {human_usd(value['cost_usd'])}")

        if not args.get_bool("apply"):
            ctx.out("")
            ctx.out("dry run: nothing written. re-run with --apply to commit.")
            return 0

        ctx.store.rewrite_records(kept)
        ctx.out(f"pruned {human_int(len(removed))} record(s); "
                f"{human_int(len(kept))} remain")
        return 0


register(PrunePlugin())
