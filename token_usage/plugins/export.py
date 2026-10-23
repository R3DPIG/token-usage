"""``export`` -- write usage data out as CSV or JSON (pure read + render).

    token-usage export --format csv --output output/usage.csv
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from token_usage import analytics
from token_usage.registry import Plugin, PluginContext, register

RECORD_COLUMNS = ["timestamp", "model", "input_tokens", "output_tokens",
                  "cached_input_tokens", "project", "tags", "note", "cost_usd"]


class ExportPlugin(Plugin):
    name = "export"
    help = "Export records or an aggregate report as CSV/JSON."
    aliases = ["dump"]
    usage = ("export [--format csv|json] [--output <path>] "
             "[--group model|project|tag|day|week|month] [--project <p>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        fmt = (args.get("format", "csv") or "csv").lower()
        if fmt not in {"csv", "json"}:
            ctx.warn(f"unsupported --format {fmt!r}; use csv or json")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            tag=args.get("tag"), since=args.get("since"), until=args.get("until"))

        dimension = args.get("group")
        if dimension:
            if dimension not in analytics.GROUPINGS:
                ctx.warn(f"unknown --group {dimension!r}")
                return 2
            grouped = analytics.group(records, dimension, ctx.catalog)
            rows = [{"bucket": name, **payload} for name, payload in grouped.items()]
        else:
            rows = []
            for record in records:
                data = record.to_dict()
                data["cost_usd"] = ctx.catalog.cost_for_record(record)
                data["tags"] = ",".join(record.tags)
                rows.append(data)

        text = self._render(rows, fmt, bool(dimension))
        output = args.get("output")
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            ctx.out(f"wrote {len(rows)} row(s) to {path}")
        else:
            ctx.out(text)
        return 0

    def _render(self, rows, fmt: str, aggregated: bool) -> str:
        if fmt == "json":
            return json.dumps(rows, indent=2, ensure_ascii=False)
        if not rows:
            return ""
        buffer = io.StringIO()
        if aggregated:
            columns = ["bucket"] + sorted({k for row in rows for k in row})
        else:
            columns = RECORD_COLUMNS
        writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(columns)),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()


register(ExportPlugin())
