"""``archive`` -- move old records out of the active history.

    token-usage archive --action list
    token-usage archive --action create --label 2024 --before 2025-01-01
    token-usage archive --action restore --label 2024 --apply
    token-usage archive --action drop --label 2024 --apply

Archiving keeps the active file small and fast without losing data: records move
to ``<home>/archive/<label>.jsonl`` and can be restored later. ``create`` removes
the archived records from the active history, so it is dry-run by default.
"""

from __future__ import annotations

from token_usage import maintenance
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd


class ArchivePlugin(Plugin):
    name = "archive"
    help = "Move old records to an archive, or restore/drop archives."
    aliases = ["archives"]
    usage = ("archive [--action list|create|restore|drop] [--label <name>] "
             "[--before <date>] [--apply]")

    def run(self, ctx: PluginContext) -> int:
        action = ctx.args.get("action", "list")
        handler = {"list": self._list, "ls": self._list, "create": self._create,
                   "add": self._create, "restore": self._restore,
                   "drop": self._drop, "delete": self._drop}.get(action)
        if handler is None:
            ctx.warn(f"unknown --action {action!r}")
            ctx.out(self.usage)
            return 2
        return handler(ctx)

    def _list(self, ctx: PluginContext) -> int:
        labels = ctx.store.list_archives()
        if not labels:
            ctx.out("no archives")
            return 0
        ctx.out(f"{len(labels)} archive(s) under {ctx.store.archive_dir}")
        for label in labels:
            records = ctx.store.read_archive(label)
            value = maintenance.estimate_value(records, ctx.catalog)
            ctx.out(f"  {label:<20} {human_int(value['records']):>8} rec  "
                    f"{human_int(value['tokens']):>14} tok  {human_usd(value['cost_usd'])}")
        return 0

    def _create(self, ctx: PluginContext) -> int:
        args = ctx.args
        label = args.get("label")
        before = args.get("before")
        if not label or not before:
            ctx.warn("--action create requires both --label and --before")
            return 2

        records = ctx.records()
        recent, old = maintenance.split_by_age(records, before)
        if not old:
            ctx.out(f"no records older than {before}")
            return 0

        value = maintenance.estimate_value(old, ctx.catalog)
        ctx.out(f"archiving {human_int(len(old))} record(s) older than {before} "
                f"into {label!r}")
        ctx.out(f"  {human_int(value['tokens'])} tokens, {human_usd(value['cost_usd'])}")
        ctx.out(f"  active history would keep {human_int(len(recent))} record(s)")

        if not args.get_bool("apply"):
            ctx.out("")
            ctx.out("dry run: nothing written. re-run with --apply to commit.")
            return 0

        path = ctx.store.write_archive(label, old)
        ctx.store.rewrite_records(recent)
        ctx.out(f"wrote {path}")
        return 0

    def _restore(self, ctx: PluginContext) -> int:
        args = ctx.args
        label = args.get("label")
        if not label:
            ctx.warn("--action restore requires --label")
            return 2
        archived = ctx.store.read_archive(label)
        if not archived:
            ctx.warn(f"archive {label!r} not found or empty")
            return 2

        current = ctx.records()
        merged, stats = maintenance.merge_records(current, archived)
        added = len(merged) - len(current)
        ctx.out(f"restoring {human_int(len(archived))} record(s) from {label!r}")
        ctx.out(f"  duplicates skipped {human_int(stats['duplicates_removed'])}, "
                f"net new {human_int(added)}")

        if not args.get_bool("apply"):
            ctx.out("")
            ctx.out("dry run: nothing written. re-run with --apply to commit.")
            return 0

        ctx.store.rewrite_records(merged)
        ctx.out(f"restored; {human_int(len(merged))} record(s) now active")
        return 0

    def _drop(self, ctx: PluginContext) -> int:
        args = ctx.args
        label = args.get("label")
        if not label:
            ctx.warn("--action drop requires --label")
            return 2
        if not ctx.store.archive_path(label).exists():
            ctx.warn(f"archive {label!r} not found")
            return 2

        if not args.get_bool("apply"):
            ctx.out(f"would delete archive {label!r} "
                    f"({human_int(len(ctx.store.read_archive(label)))} record(s))")
            ctx.out("dry run: nothing written. re-run with --apply to commit.")
            return 0

        ctx.store.delete_archive(label)
        ctx.out(f"deleted archive {label!r}")
        return 0


register(ArchivePlugin())
