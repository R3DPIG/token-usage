"""``edit`` -- correct a single record in place.

    token-usage edit --index 3 --project prod --tags api,prod
    token-usage edit --index 3 --input 5000 --apply

Records are append-only, so "editing" means rewriting the file with one row
changed. ``--index`` refers to the position in ``list`` output (0-based, newest
first is *not* used here -- the index is the storage order, matching ``--oldest``).
"""

from __future__ import annotations

from token_usage import maintenance
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd


class EditPlugin(Plugin):
    name = "edit"
    help = "Edit one record's fields by index."
    aliases = ["update"]
    usage = ("edit --index <n> [--model <m>] [--input <n>] [--output <n>] "
             "[--cached <n>] [--timestamp <iso>] [--project <p>] "
             "[--tags a,b] [--note <text>] [--apply]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        records = ctx.records()
        if not records:
            ctx.out("no records found")
            return 0

        index = args.get_int("index", -1)
        if index < 0 or index >= len(records):
            ctx.warn(f"--index must be between 0 and {len(records) - 1}")
            return 2

        changes = self._collect_changes(args)
        if not changes:
            ctx.warn("no fields to change; supply at least one editable field")
            ctx.out(self.usage)
            return 2

        before = records[index]
        try:
            after = maintenance.apply_edits(before, changes)
        except ValueError as exc:
            ctx.warn(str(exc))
            return 2

        diff = maintenance.diff_records(before, after)
        if not diff:
            ctx.out("no effective change")
            return 0

        ctx.out(f"record #{index} ({before.model})")
        for field, old, new in diff:
            ctx.out(f"  {field:<22} {old!r} -> {new!r}")
        old_cost = ctx.catalog.cost_for_record(before)
        new_cost = ctx.catalog.cost_for_record(after)
        ctx.out(f"  {'cost_usd':<22} {human_usd(old_cost)} -> {human_usd(new_cost)}")

        if not args.get_bool("apply"):
            ctx.out("")
            ctx.out("dry run: nothing written. re-run with --apply to commit.")
            return 0

        records[index] = after
        ctx.store.rewrite_records(records)
        ctx.out(f"updated record #{index}")
        return 0

    def _collect_changes(self, args) -> dict:
        """Map CLI flags to record fields, skipping anything not supplied."""
        mapping = {"model": "model", "input": "input_tokens",
                   "output": "output_tokens", "cached": "cached_input_tokens",
                   "timestamp": "timestamp", "project": "project",
                   "tags": "tags", "note": "note"}
        changes = {}
        for flag, field in mapping.items():
            if args.get(flag) is None:
                continue
            if field in ("input_tokens", "output_tokens", "cached_input_tokens"):
                changes[field] = args.get_int(flag)
            else:
                changes[field] = args.get(flag)
        return changes


register(EditPlugin())
