"""``plan`` -- where will this period land, and what should I do about it?

    token-usage plan
    token-usage plan --period weekly
    token-usage plan --limit 50
    token-usage plan --what-if gpt-4o --change -30

Combines the pace so far with any configured budgets to answer the mid-period
question: am I on track, and how much can I still spend per day?
"""

from __future__ import annotations

from token_usage import analytics, planning
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd

_VERDICT_MARK = {"on_track": "[ ok ]", "at_risk": "[WARN]", "exceeded": "[OVER]",
                 "no_budget": "[ -- ]"}


class PlanPlugin(Plugin):
    name = "plan"
    help = "Project period-end spend and check it against budgets."
    aliases = ["planning"]
    usage = ("plan [--period daily|weekly|monthly] [--limit <usd>] [--project <p>] "
             "[--what-if <model>] [--change <pct>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        period = args.get("period", "monthly")
        if period not in planning.PLANNABLE_PERIODS:
            ctx.warn(f"--period must be one of {planning.PLANNABLE_PERIODS}")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"))
        if not records:
            ctx.out("no records in scope; nothing to plan")
            return 0

        # An explicit --limit is a one-off budget; otherwise use stored budgets.
        limit = args.get_float("limit", 0.0)
        if limit > 0:
            outcome = planning.budget_outcome(records, ctx.catalog, limit, period=period)
            outcome["label"] = "cli --limit"
            outcomes = [outcome]
        else:
            outcomes = planning.plan_summary(records, ctx.catalog,
                                             ctx.store.load_budgets(),
                                             period=period)["budgets"]

        overall = planning.project_period_end(records, ctx.catalog, period=period)
        self._print_projection(ctx, overall, outcomes)

        what_if = args.get("what_if")
        if what_if:
            return self._print_scenario(ctx, records, what_if,
                                        args.get_float("change", -30.0), period)

        return 1 if any(o["verdict"] == "exceeded" for o in outcomes) else 0

    def _print_projection(self, ctx: PluginContext, overall: dict,
                          outcomes: list) -> None:
        ctx.out(f"{overall['period']} plan")
        ctx.out("=" * 52)
        ctx.out(f"  spent so far      {human_usd(overall['spent_usd'])} "
                f"over {overall['days_elapsed']} day(s)")
        ctx.out(f"  daily pace        {human_usd(overall['daily_rate_usd'])}/day")
        ctx.out(f"  days remaining    {overall['days_remaining']}")
        ctx.out(f"  projected total   {human_usd(overall['projected_total_usd'])}")
        ctx.out(f"  records           {human_int(overall['records'])}")

        if not outcomes:
            ctx.out("")
            ctx.out("no budgets configured; pass --limit <usd> for a one-off check")
            return

        ctx.out("")
        ctx.out(f"  {'budget':<16}{'limit':>12}{'projected':>12}{'verdict':>10}"
                f"{'per-day':>12}")
        for outcome in outcomes:
            mark = _VERDICT_MARK.get(outcome["verdict"], "[????]")
            ctx.out(f"  {outcome['label']:<16}{human_usd(outcome['limit_usd']):>12}"
                    f"{human_usd(outcome['projected_total_usd']):>12}"
                    f"{mark:>10}{human_usd(outcome['daily_allowance_usd']):>12}")

        for outcome in outcomes:
            if outcome["verdict"] == "at_risk":
                ctx.warn(f"{outcome['label']}: projected "
                         f"{human_usd(outcome['overage_usd'])} over limit; "
                         f"cut {outcome['required_reduction_pct']:.1f}% or stay "
                         f"under {human_usd(outcome['daily_allowance_usd'])}/day")
            elif outcome["verdict"] == "exceeded":
                ctx.warn(f"{outcome['label']}: limit already exceeded")

    def _print_scenario(self, ctx: PluginContext, records, model: str,
                        change_pct: float, period: str) -> int:
        result = planning.scenario_shift(records, ctx.catalog, model,
                                         change_pct, period=period)
        ctx.out("")
        ctx.out(f"what-if: {model} {change_pct:+.0f}%")
        if result["note"]:
            ctx.out(f"  {result['note']}")
            return 0
        ctx.out(f"  model spend so far   {human_usd(result['model_cost_usd'])}")
        ctx.out(f"  projected model cost {human_usd(result['projected_model_usd'])}")
        ctx.out(f"  delta                {human_usd(result['delta_usd'])}")
        ctx.out(f"  adjusted total       {human_usd(result['adjusted_total_usd'])} "
                f"(was {human_usd(result['projected_total_usd'])})")
        return 0


register(PlanPlugin())
