"""``optimize`` -- what should I change, and how much would it save?

    token-usage optimize
    token-usage optimize --view substitute
    token-usage optimize --view cache
    token-usage optimize --tier-drop 1
    token-usage optimize --min-savings 0.01

Every recommendation is backed by repricing the tokens actually observed, so the
dollar figures are what the bill would have been -- not an estimate. Changes
whose saving cannot be computed honestly (batching) report no figure at all.
"""

from __future__ import annotations

from token_usage import analytics, optimize
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd

_VIEWS = ("summary", "substitute", "cache", "repeat", "waste", "batch")
_CONFIDENCE_MARK = {"high": "high", "medium": "med ", "low": "low "}


class OptimizePlugin(Plugin):
    name = "optimize"
    help = "Quantified cost-saving recommendations."
    aliases = ["optimise", "savings"]
    usage = ("optimize [--view summary|substitute|cache|repeat|waste|batch] "
             "[--tier-drop 0|1|2] [--min-savings <usd>] "
             "[--project <p>] [--model <m>] [--since <date>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2

        tier_drop = args.get_int("tier_drop", 0)
        if tier_drop < 0 or tier_drop > len(optimize.TIER_ORDER) - 1:
            ctx.warn(f"--tier-drop must be between 0 and "
                     f"{len(optimize.TIER_ORDER) - 1}")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            since=args.get("since"), until=args.get("until"))
        if not records:
            ctx.out("no records in scope; nothing to optimise")
            return 0

        min_savings = args.get_float("min_savings", 0.0)
        handler = {"summary": self._summary, "substitute": self._substitute,
                   "cache": self._cache, "repeat": self._repeat,
                   "waste": self._waste, "batch": self._batch}[view]
        return handler(ctx, records, tier_drop, min_savings)

    # -- views ------------------------------------------------------------
    def _summary(self, ctx: PluginContext, records, tier_drop: int,
                 min_savings: float) -> int:
        data = optimize.summarize(records, ctx.catalog, allow_tier_drop=tier_drop)
        recs = data["recommendations"]

        ctx.out(f"optimisation review over {len(records)} record(s)")
        ctx.out("=" * 62)
        ctx.out(f"  current spend     {human_usd(data['spent_usd'])}")
        ctx.out(f"  identified saving {human_usd(data['total_savings_usd'])} "
                f"({data['savings_ratio'] * 100:.1f}% of spend)")
        ctx.out(f"  recommendations   {data['quantified']} quantified, "
                f"{data['unquantified']} unquantified")

        if not recs:
            ctx.out("")
            ctx.out("  no savings found -- usage already looks efficient")
            return 0

        ctx.out("")
        for index, rec in enumerate(recs, start=1):
            saving = ("n/a" if rec.savings_usd is None
                      else human_usd(rec.savings_usd))
            ctx.out(f"  {index}. [{rec.kind}] {rec.title}")
            ctx.out(f"     saving {saving:>12}  confidence "
                    f"{_CONFIDENCE_MARK[rec.confidence]}  effort {rec.effort}")
            ctx.out(f"     {rec.detail}")
        return 0

    def _substitute(self, ctx: PluginContext, records, tier_drop: int,
                    min_savings: float) -> int:
        items = optimize.model_substitutions(records, ctx.catalog,
                                             allow_tier_drop=tier_drop,
                                             min_savings_usd=min_savings)
        ctx.out("model substitution candidates")
        ctx.out("=" * 62)
        if not items:
            ctx.out("  no cheaper model found within the allowed tier drop")
            if tier_drop == 0:
                ctx.out("  hint: --tier-drop 1 also considers one band cheaper")
            return 0
        for item in items:
            ctx.out(f"  {item['model']} -> {item['alternative']}")
            ctx.out(f"     now {human_usd(item['current_cost_usd']):>12}  "
                    f"would be {human_usd(item['alternative_cost_usd']):>12}  "
                    f"save {human_usd(item['savings_usd'])}")
            ctx.out(f"     {human_int(item['records'])} call(s), "
                    f"tier drop {item['tier_drop']}")
        return 0

    def _cache(self, ctx: PluginContext, records, tier_drop: int,
               min_savings: float) -> int:
        items = optimize.cache_opportunities(records, ctx.catalog)
        ctx.out("prompt caching opportunities")
        ctx.out("=" * 62)
        if not items:
            ctx.out("  every model with a cache discount is already caching, "
                    "or has too little traffic to matter")
            return 0
        for item in items:
            ctx.out(f"  {item['model']}")
            ctx.out(f"     {human_int(item['input_tokens'])} input tokens, "
                    f"0 cached, {human_int(item['records'])} call(s)")
            ctx.out(f"     ${item['input_per_1m']:g}/1M -> "
                    f"${item['cached_per_1m']:g}/1M cached, "
                    f"save {human_usd(item['savings_usd'])}")
        return 0

    def _repeat(self, ctx: PluginContext, records, tier_drop: int,
                min_savings: float) -> int:
        items = optimize.repeated_shapes(records, ctx.catalog)
        ctx.out("repeated prompt shapes")
        ctx.out("=" * 62)
        if not items:
            ctx.out("  no prompt shape repeated often enough to be worth caching")
            return 0
        for item in items:
            ctx.out(f"  {item['model']}  {human_int(item['input_tokens'])} in / "
                    f"{human_int(item['output_tokens'])} out")
            ctx.out(f"     sent {item['repeats'] + 1} times, "
                    f"save {human_usd(item['savings_usd'])} by caching repeats")
        return 0

    def _waste(self, ctx: PluginContext, records, tier_drop: int,
               min_savings: float) -> int:
        result = optimize.output_waste(records, ctx.catalog)
        ctx.out("calls with almost no output")
        ctx.out("=" * 62)
        ctx.out(f"  {result['count']} call(s) costing "
                f"{human_usd(result['wasted_usd'])}")
        if not result["offenders"]:
            ctx.out("  nothing to report")
            return 0
        ctx.out("")
        for item in result["offenders"]:
            ctx.out(f"  {item['timestamp']}  {item['model']:<22} "
                    f"out-ratio {item['output_ratio']:.3f}  "
                    f"{human_usd(item['cost_usd'])}")
        return 0

    def _batch(self, ctx: PluginContext, records, tier_drop: int,
               min_savings: float) -> int:
        result = optimize.small_calls(records)
        ctx.out("small-call batching")
        ctx.out("=" * 62)
        ctx.out(f"  threshold   {human_int(result['threshold_tokens'])} tokens")
        ctx.out(f"  small calls {human_int(result['count'])} "
                f"({result['share'] * 100:.1f}% of all calls)")
        ctx.out(f"  their tokens {human_int(result['total_tokens'])}")
        ctx.out("")
        ctx.out("  no dollar figure is given: the saving depends on how the "
                "caller restructures its requests")
        return 0


register(OptimizePlugin())
