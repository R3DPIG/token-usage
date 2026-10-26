"""``doctor`` -- one-shot health check over the whole workspace.

    token-usage doctor

Runs every diagnostic in one pass: data integrity, cost concentration, model
efficiency, waste, anomalies, and budget status. Intended as the first command
to run when something looks off, and as a CI gate (non-zero exit on problems).
"""

from __future__ import annotations

from token_usage import analytics, insights
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd


class DoctorPlugin(Plugin):
    name = "doctor"
    help = "Run a full health check over the stored usage data."
    aliases = ["check", "health"]
    usage = "doctor [--project <p>] [--strict]"

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        records = analytics.filter_records(ctx.records(), project=args.get("project"))
        problems = []

        ctx.out("token-usage doctor")
        ctx.out("=" * 46)

        if not records:
            ctx.out("no records found")
            ctx.out(f"hint: run 'sample' for demo data, or 'import --file <path>'")
            return 1

        overall = analytics.totals(records, ctx.catalog)
        ctx.out(f"records        {human_int(overall['records'])}")
        ctx.out(f"total tokens   {human_int(overall['total_tokens'])}")
        ctx.out(f"total cost     {human_usd(overall['cost_usd'])}")
        ctx.out(f"models used    {overall['models_used']}")

        # 1. Data integrity: records with no tokens are almost always mistakes.
        empty = [r for r in records if r.total_tokens == 0]
        if empty:
            problems.append(f"{len(empty)} record(s) have zero tokens")

        # 2. Cost concentration.
        conc = insights.concentration(records, ctx.catalog, dimension="model", top=3)
        ctx.out("")
        ctx.out(f"top-3 model share  {conc['top_share'] * 100:.1f}%")
        if conc["top_share"] > 0.9 and len(conc["top"]) > 1:
            problems.append("over 90% of spend sits in the top 3 models")

        # 3. Model efficiency.
        efficiency = insights.model_efficiency(records, ctx.catalog)
        if efficiency:
            ctx.out("")
            ctx.out("cheapest models per 1K tokens:")
            for row in efficiency[:3]:
                ctx.out(f"  {row['model']:<24}{human_usd(row['cost_per_1k_tokens']):>12}"
                        f"  out-ratio {row['output_ratio']:.2f}")

        # 4. Waste: paid-for calls that produced almost nothing.
        waste = insights.waste_estimate(records, ctx.catalog)
        ctx.out("")
        ctx.out(f"low-output spend   {human_usd(waste['wasted_usd'])} "
                f"across {waste['count']} call(s)")
        if waste["wasted_usd"] > 0 and waste["wasted_usd"] > overall["cost_usd"] * 0.1:
            problems.append("over 10% of spend went to near-empty responses")

        # 5. Anomalies.
        anomalies = insights.detect_anomalies(records, ctx.catalog)
        ctx.out(f"anomalous days     {len(anomalies)}")
        if anomalies:
            problems.append(f"{len(anomalies)} day(s) deviate sharply from the mean")

        # 6. Budgets.
        budgets = ctx.store.load_budgets()
        if budgets:
            results = analytics.evaluate_budgets(records, ctx.catalog, budgets)
            exceeded = [r for r in results if r["status"] == "exceeded"]
            warned = [r for r in results if r["status"] == "warning"]
            ctx.out(f"budgets            {len(results)} "
                    f"({len(exceeded)} exceeded, {len(warned)} warning)")
            for item in exceeded:
                problems.append(f"budget {item['label']!r} exceeded "
                                f"({human_usd(item['spent_usd'])} / {human_usd(item['limit_usd'])})")
        else:
            ctx.out("budgets            none configured")

        ctx.out("")
        ctx.out("=" * 46)
        if not problems:
            ctx.out("all checks passed")
            return 0

        ctx.out(f"{len(problems)} issue(s) found:")
        for problem in problems:
            ctx.out(f"  - {problem}")
        # --strict turns warnings into a failing exit code for CI use.
        return 1 if args.get_bool("strict") or problems else 0


register(DoctorPlugin())
