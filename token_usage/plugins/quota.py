"""``quota`` -- split a budget across models/projects by historical share.

    token-usage quota --total 100
    token-usage quota --total 100 --dimension model --floor 5
    token-usage quota --total 100 --dimension project --period monthly

Answers "if I have $100 to spend, how much should each project get?" using each
bucket's share of past spend. ``--floor`` guarantees a minimum per bucket, which
matters for a bucket with little history that is expected to grow.
"""

from __future__ import annotations

from token_usage import analytics, planning
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import bar, human_int, human_usd


class QuotaPlugin(Plugin):
    name = "quota"
    help = "Allocate a budget across models/projects by historical share."
    aliases = ["allocate"]
    usage = ("quota --total <usd> [--dimension model|project|tag] "
             "[--floor <usd>] [--period daily|weekly|monthly|total]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        total = args.get_float("total", 0.0)
        if total <= 0:
            ctx.warn("--total must be a positive number of USD")
            ctx.out(self.usage)
            return 2

        dimension = args.get("dimension", "project")
        if dimension not in ("model", "project", "tag"):
            ctx.warn("--dimension must be one of model, project, tag")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"))
        period = args.get("period", "total")
        if period != "total":
            records = analytics.filter_by_period(records, period)
        if not records:
            ctx.out("no records in scope; cannot derive a quota split")
            return 0

        floor = args.get_float("floor", 0.0)
        quotas = planning.allocate_quota(total, records, ctx.catalog,
                                         dimension=dimension, floor_usd=floor)
        if not quotas:
            ctx.out("no buckets to allocate")
            return 0

        spent_total = sum(item["cost_usd"] for item in quotas)
        ctx.out(f"allocating {human_usd(total)} across {len(quotas)} {dimension}(s)")
        ctx.out(f"based on {human_usd(spent_total)} of historical spend"
                + (f" (period: {period})" if period != "total" else ""))
        if floor > 0:
            ctx.out(f"floor of {human_usd(floor)} per bucket")
        ctx.out("")

        maximum = max(item["quota_usd"] for item in quotas)
        header = (f"  {'bucket':<20}{'history':>12}{'share':>9}{'quota':>12}")
        ctx.out(header)
        ctx.out("  " + "-" * (len(header) - 2))
        for item in quotas:
            ctx.out(f"  {item['bucket']:<20}{human_usd(item['cost_usd']):>12}"
                    f"{item['share'] * 100:>8.1f}%{human_usd(item['quota_usd']):>12}"
                    f"  {bar(item['quota_usd'], maximum, width=16)}")

        allocated = sum(item["quota_usd"] for item in quotas)
        ctx.out("")
        ctx.out(f"  allocated {human_usd(allocated)} of {human_usd(total)}")
        # Rounding to micro-cents can leave a tiny residue; report it honestly
        # rather than pretending the parts sum exactly.
        residue = total - allocated
        if abs(residue) > 1e-6:
            ctx.out(f"  rounding residue {human_usd(residue)}")

        top = quotas[0]
        if top["share"] > 0.5:
            ctx.warn(f"{top['bucket']!r} takes {top['share'] * 100:.0f}% of the "
                     "budget; consider whether that concentration is intended")
        return 0


register(QuotaPlugin())
