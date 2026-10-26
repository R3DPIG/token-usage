"""``import`` -- load a provider usage export into the local history.

    token-usage import --file export.json --format openai --project prod
    token-usage import --file usage.csv --dry-run

Format is auto-detected when ``--format`` is omitted. ``--dry-run`` parses and
reports without writing, which is the safe way to check a new export shape.
"""

from __future__ import annotations

from pathlib import Path

from token_usage.ingest import SUPPORTED_FORMATS, ingest_file
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd


class ImportPlugin(Plugin):
    name = "import"
    help = "Import a provider usage export (JSON/CSV)."
    aliases = ["ingest"]
    usage = ("import --file <path> [--format openai|anthropic|generic] "
             "[--project <p>] [--dry-run]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        raw_path = args.get("file") or (args.positionals[0] if args.positionals else None)
        if not raw_path:
            ctx.warn("--file is required")
            ctx.out(self.usage)
            return 2

        path = Path(raw_path)
        if not path.exists():
            ctx.warn(f"file not found: {path}")
            return 2

        fmt = args.get("format", "")
        if fmt and fmt not in SUPPORTED_FORMATS:
            ctx.warn(f"unsupported --format {fmt!r}; expected one of {SUPPORTED_FORMATS}")
            return 2

        try:
            result = ingest_file(path, fmt=fmt, project=args.get("project", "imported"))
        except (ValueError, OSError) as exc:
            ctx.warn(f"could not parse {path}: {exc}")
            return 1

        ctx.out(result.summary())
        for error in result.errors[:5]:
            ctx.out(f"  skipped: {error}")
        if result.errors and len(result.errors) > 5:
            ctx.out(f"  ... and {len(result.errors) - 5} more")

        if not result.records:
            ctx.warn("nothing to import")
            return 1

        total = sum(ctx.catalog.cost_for_record(r) for r in result.records)
        tokens = sum(r.total_tokens for r in result.records)
        ctx.out(f"  tokens {human_int(tokens)}, estimated cost {human_usd(total)}")

        if args.get_bool("dry_run"):
            ctx.out("dry run: nothing written")
            return 0

        inserted = ctx.store.append_records(result.records)
        ctx.out(f"imported {inserted} record(s) into {ctx.store.home}")
        return 0


register(ImportPlugin())
