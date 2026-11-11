"""``reconcile`` -- check the provider's invoice against local records.

    token-usage reconcile --bill invoice.csv
    token-usage reconcile --bill invoice.json --view models
    token-usage reconcile --bill invoice.csv --view gaps --tolerance 0.25
    token-usage reconcile --bill invoice.csv --view explain --strict

The bill is a JSON or CSV file with one row per model: ``model`` and
``amount_usd`` are required, token counts are optional. Token counts unlock the
narrower verdicts (``price`` vs ``records``); without them a mismatch is
reported as ``not evaluated`` rather than guessed at.

``quality`` asks whether the dataset is internally consistent; this command
asks whether it agrees with the outside world that sent the invoice. The bill
is read-only, and so is the workspace: reconciliation never edits anything.
"""

from __future__ import annotations

from pathlib import Path

from token_usage import analytics, reconcile
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd

_VIEWS = ("summary", "models", "gaps", "explain")
_STATUS_LABEL = {reconcile.STATUS_MATCHED: "ok",
                 reconcile.STATUS_MISMATCH: "MISMATCH",
                 reconcile.STATUS_MISSING_LOCALLY: "not local",
                 reconcile.STATUS_NOT_ON_BILL: "not on bill"}


class ReconcilePlugin(Plugin):
    name = "reconcile"
    help = "Reconcile a provider bill against the records in this workspace."
    aliases = ["bill", "audit-bill"]
    usage = ("reconcile --bill <file.csv|file.json> "
             "[--view summary|models|gaps|explain] [--tolerance <usd>] "
             "[--model <name>] [--strict]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2
        bill_path = args.get("bill")
        if not bill_path:
            ctx.warn("--bill is required: pass the invoice file to reconcile")
            return 2

        try:
            lines, errors = reconcile.load_bill(Path(str(bill_path)))
        except FileNotFoundError:
            ctx.warn(f"bill file not found: {bill_path}")
            return 2
        except (ValueError, KeyError) as exc:
            ctx.warn(f"could not parse bill {bill_path}: {exc}")
            return 2
        for text in errors:
            ctx.warn(f"bill: {text}")
        if not lines:
            ctx.out("no usable bill lines; nothing to reconcile")
            return 0

        tolerance = args.get_float("tolerance", reconcile.DEFAULT_TOLERANCE)
        if tolerance < 0:
            ctx.warn("--tolerance must be >= 0")
            return 2
        records = analytics.filter_records(
            ctx.records(), model=args.get("model"))
        result = reconcile.compare(lines, records, ctx.catalog,
                                   tolerance=tolerance)

        handler = {"summary": self._summary, "models": self._models,
                   "gaps": self._gaps, "explain": self._explain}[view]
        exit_code = handler(ctx, result)
        if args.get_bool("strict") and result["status"] != "clean":
            ctx.warn("--strict: unexplained gaps remain")
            return 1
        return exit_code

    # -- views -------------------------------------------------------------
    def _header(self, ctx: PluginContext, result) -> None:
        counts = result["counts"]
        ctx.out(f"bill reconciliation ({result['status']})")
        ctx.out("=" * 70)
        ctx.out(f"  bill       {human_usd(result['bill_total'])} "
                f"({len(result['rows'])} model(s))")
        ctx.out(f"  workspace  {human_usd(result['local_total'])}")
        ctx.out(f"  net gap    {human_usd(result['net_gap'])} "
                f"(bill minus workspace)")
        ctx.out(f"  tolerance  {human_usd(result['tolerance'])}")
        ctx.out(f"  rows       {counts['matched']} matched, "
                f"{counts['mismatch']} mismatched, "
                f"{counts['missing_locally']} missing locally, "
                f"{counts['not_on_bill']} not on bill")

    def _summary(self, ctx: PluginContext, result) -> int:
        self._header(ctx, result)
        ctx.out("")
        ctx.out(f"  coverage   {result['coverage'] * 100:.1f}% of the bill "
                "matched within tolerance")
        for text in result["warnings"]:
            ctx.out("")
            ctx.warn(text)
        if result["status"] == "clean":
            ctx.out("")
            ctx.out("  the bill and the workspace agree; nothing to chase")
        else:
            ctx.out("")
            ctx.out("  run --view gaps for the models that differ, or "
                    "--view explain for candidate causes")
        return 0

    def _models(self, ctx: PluginContext, result) -> int:
        self._header(ctx, result)
        ctx.out("")
        ctx.out(f"  {'model':<24}{'billed':>12}{'workspace':>12}"
                f"{'diff':>12}  status")
        for row in result["rows"]:
            ctx.out(f"  {row['model']:<24}{human_usd(row['billed_usd']):>12}"
                    f"{human_usd(row['local_usd']):>12}"
                    f"{human_usd(row['diff_usd']):>12}"
                    f"  {_STATUS_LABEL[row['status']]}")
        return 0

    def _gaps(self, ctx: PluginContext, result) -> int:
        self._header(ctx, result)
        ctx.out("")
        gaps = reconcile.gap_rows(result)
        if not gaps:
            ctx.out("  every model agrees within tolerance")
            return 0
        for row in gaps:
            ctx.out(f"  {row['model']}: {_STATUS_LABEL[row['status']]}, "
                    f"diff {human_usd(row['diff_usd'])}")
            verdict = row.get("verdict")
            if verdict is None:
                continue
            ctx.out(f"    tokens: {verdict['verdict']} -- {verdict['reason']}")
            if verdict["verdict"] != "not evaluated":
                ctx.out(f"    bill {human_int(verdict['bill_tokens'])} tok vs "
                        f"workspace {human_int(verdict['local_tokens'])} tok")
                if verdict.get("implied_rate") and verdict.get("catalog_rate"):
                    ctx.out(f"    implied {human_usd(verdict['implied_rate'])}/1M "
                            f"vs catalog {human_usd(verdict['catalog_rate'])}/1M")
        return 0

    def _explain(self, ctx: PluginContext, result) -> int:
        self._header(ctx, result)
        ctx.out("")
        explained = reconcile.explain(result)
        if not explained:
            ctx.out("  every model agrees within tolerance; nothing to explain")
            return 0
        for item in explained:
            ctx.out(f"  {item['model']} ({_STATUS_LABEL[item['status']]}, "
                    f"diff {human_usd(item['diff_usd'])}) is consistent with:")
            for hypothesis in item["hypotheses"]:
                ctx.out(f"    - {hypothesis}")
            evidence = item.get("token_evidence")
            if evidence is not None:
                ctx.out(f"    token evidence: {evidence['verdict']} -- "
                        f"{evidence['reason']}")
        ctx.out("")
        ctx.out("  hypotheses are things to check, not conclusions; "
                "two totals cannot prove a cause")
        for text in result["warnings"]:
            ctx.warn(text)
        return 0


register(ReconcilePlugin())
