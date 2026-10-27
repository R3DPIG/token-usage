"""``dedupe`` -- remove duplicate records.

    token-usage dedupe                      # preview only
    token-usage dedupe --apply              # actually rewrite the history
    token-usage dedupe --ignore-timestamp --keep last --apply

Dry-run by default: destructive commands must be asked twice.
"""

from __future__ import annotations

from token_usage import maintenance
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd


class DedupePlugin(Plugin):
    name = "dedupe"
    help = "Find and remove duplicate usage records."
    aliases = ["dedup"]
    usage = "dedupe [--ignore-timestamp] [--keep first|last] [--apply]"

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        keep = args.get("keep", "first")
        if keep not in ("first", "last"):
            ctx.warn("--keep must be 'first' or 'last'")
            return 2

        records = ctx.records()
        if not records:
            ctx.out("no records found")
            return 0

        ignore_ts = args.get_bool("ignore_timestamp")
        kept, removed, groups = maintenance.dedupe(
            records, ignore_timestamp=ignore_ts, keep=keep)

        ctx.out(f"scanned {human_int(len(records))} record(s)")
        ctx.out(f"duplicate groups  {len(groups)}")
        ctx.out(f"removable records {human_int(len(removed))}")
        if ignore_ts:
            ctx.out("(timestamp ignored when matching)")

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
        ctx.out(f"removed {human_int(len(removed))} duplicate record(s); "
                f"{human_int(len(kept))} remain")
        return 0


register(DedupePlugin())
