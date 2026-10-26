"""``anomaly`` -- flag days whose spend deviates sharply from the norm.

    token-usage anomaly
    token-usage anomaly --z 1.5 --min-samples 3
"""

from __future__ import annotations

from token_usage import analytics, insights
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd


class AnomalyPlugin(Plugin):
    name = "anomaly"
    help = "Flag days with unusually high or low spend."
    aliases = ["outliers"]
    usage = "anomaly [--z <sigma>] [--min-samples <n>] [--project <p>]"

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        threshold = args.get_float("z", 2.0)
        if threshold <= 0:
            ctx.warn("--z must be > 0")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"))
        if not records:
            ctx.out("no records in scope")
            return 0

        daily = analytics.by_day(records, ctx.catalog)
        min_samples = args.get_int("min_samples", 5)
        flagged = insights.detect_anomalies(
            records, ctx.catalog, z_threshold=threshold, min_samples=min_samples)

        ctx.out(f"scanned {len(daily)} day(s) at z >= {threshold}")
        if not flagged:
            ctx.out("no anomalies detected")
            return 0

        ctx.out("")
        ctx.out(f"  {'day':<12}{'cost':>12}{'mean':>12}{'z':>8}  direction")
        for item in flagged:
            ctx.out(f"  {item['day']:<12}{human_usd(item['cost_usd']):>12}"
                    f"{human_usd(item['mean_usd']):>12}{item['z_score']:>8.2f}  "
                    f"{item['direction']}")
        spikes = [i for i in flagged if i["direction"] == "spike"]
        if spikes:
            ctx.warn(f"{len(spikes)} spike(s) detected; "
                     "check for retries, a model switch, or a runaway loop")
        return 0


register(AnomalyPlugin())
