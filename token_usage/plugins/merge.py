"""``merge`` -- combine another workspace or archive into this one.

    token-usage merge --from ../other-workspace
    token-usage merge --archive 2024-05 --apply
    token-usage merge --from ../team --no-dedupe --apply

Sources are read-only; only the current workspace is written. Duplicates are
removed by default, which is what you want when two people imported the same
export.
"""

from __future__ import annotations

from pathlib import Path

from token_usage import maintenance
from token_usage.registry import Plugin, PluginContext, register
from token_usage.storage import JsonStore
from token_usage.utils import human_int, human_usd


class MergePlugin(Plugin):
    name = "merge"
    help = "Merge another workspace or archive into this one."
    aliases = ["combine"]
    usage = ("merge [--from <home>] [--archive <label>] "
             "[--no-dedupe] [--ignore-timestamp] [--apply]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        sources = []

        other_home = args.get("from")
        if other_home:
            other = JsonStore(Path(other_home).expanduser())
            if not other.usage_path.exists():
                ctx.warn(f"no usage.jsonl under {other.home}")
                return 2
            sources.append((f"workspace {other.home}", other.load_records()))

        label = args.get("archive")
        if label:
            archived = ctx.store.read_archive(label)
            if not archived:
                ctx.warn(f"archive {label!r} not found or empty; "
                         f"available: {ctx.store.list_archives() or 'none'}")
                return 2
            sources.append((f"archive {label}", archived))

        if not sources:
            ctx.warn("nothing to merge; supply --from <home> or --archive <label>")
            ctx.out(self.usage)
            return 2

        current = ctx.records()
        incoming = [record for _, records in sources for record in records]
        for name, records in sources:
            ctx.out(f"source {name}: {human_int(len(records))} record(s)")

        merged, stats = maintenance.merge_records(
            current, incoming,
            dedupe_after=not args.get_bool("no_dedupe"),
            ignore_timestamp=args.get_bool("ignore_timestamp"))

        added = len(merged) - len(current)
        ctx.out("")
        ctx.out(f"current          {human_int(len(current))}")
        ctx.out(f"incoming         {human_int(len(incoming))}")
        ctx.out(f"duplicates       {human_int(stats['duplicates_removed'])}")
        ctx.out(f"net new records  {human_int(added)}")

        if added <= 0:
            ctx.out("nothing new to add")
            return 0

        value = maintenance.estimate_value(merged[len(current):], ctx.catalog)
        ctx.out(f"added value      {human_int(value['tokens'])} tokens, "
                f"{human_usd(value['cost_usd'])}")

        if not args.get_bool("apply"):
            ctx.out("")
            ctx.out("dry run: nothing written. re-run with --apply to commit.")
            return 0

        ctx.store.rewrite_records(merged)
        ctx.out(f"merged; {human_int(len(merged))} record(s) now stored")
        return 0


register(MergePlugin())
