"""``simulate`` -- how bad could this period get?

    token-usage simulate
    token-usage simulate --limit 50
    token-usage simulate --view distribution
    token-usage simulate --view methods
    token-usage simulate --method normal --iterations 5000

``plan`` gives one projected number. This command gives a distribution, because
"projected $47" and "90% likely under $63" lead to different decisions.

Results are seeded, so the same data and options always produce the same P90.
"""

from __future__ import annotations

from token_usage import analytics, simulation
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import bar, human_int, human_usd

_VIEWS = ("summary", "distribution", "methods", "budget")
_PERIODS = ("daily", "weekly", "monthly")


class SimulatePlugin(Plugin):
    name = "simulate"
    help = "Simulate period-end spend as a distribution, not a point."
    aliases = ["sim", "risk"]
    usage = ("simulate [--view summary|distribution|methods|budget] "
             "[--period daily|weekly|monthly] [--method bootstrap|normal] "
             "[--iterations <n>] [--limit <usd>] [--seed <n>] "
             "[--project <p>] [--model <m>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2

        period = args.get("period", "monthly")
        if period not in _PERIODS:
            ctx.warn(f"--period must be one of {_PERIODS}")
            return 2

        method = args.get("method", "bootstrap")
        if method not in simulation.METHODS:
            ctx.warn(f"--method must be one of {simulation.METHODS}")
            return 2

        iterations = args.get_int("iterations", 2000)
        if iterations < 1:
            ctx.warn("--iterations must be >= 1")
            return 2

        limit = args.get_float("limit", 0.0)
        seed = args.get_int("seed", 7)

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            since=args.get("since"), until=args.get("until"))
        if not records:
            ctx.out("no records in scope; nothing to simulate")
            return 0

        handler = {"summary": self._summary, "distribution": self._distribution,
                   "methods": self._methods, "budget": self._budget}[view]
        return handler(ctx, records, period, method, iterations, limit, seed)

    # -- views ------------------------------------------------------------
    def _header(self, ctx: PluginContext, title: str) -> None:
        ctx.out(title)
        ctx.out("=" * 64)

    def _risk_lines(self, ctx: PluginContext, data) -> None:
        """The three numbers every risk view needs: median, P90, probability."""
        ctx.out(f"  spent so far   {human_usd(data['spent_usd'])}")
        ctx.out(f"  median end     {human_usd(data['percentiles'][50])}")
        ctx.out(f"  P90 end        {human_usd(data['p90_total_usd'])}")
        if data["exceed_probability"] is not None:
            ctx.out(f"  overspend risk {data['exceed_probability'] * 100:.1f}%")

    def _cut_lines(self, ctx: PluginContext, cut) -> None:
        if cut["cut_usd"] <= 0:
            return
        ctx.out("")
        ctx.out(f"  to bring P90 inside the limit, cut "
                f"{human_usd(cut['cut_usd'])} ({cut['cut_pct']:.1f}%)")
        if not cut["feasible"]:
            ctx.warn(cut["note"])

    def _summary(self, ctx: PluginContext, records, period, method, iterations,
                 limit, seed) -> int:
        data = simulation.summarize(records, ctx.catalog, period=period,
                                    limit_usd=limit, iterations=iterations,
                                    seed=seed)
        self._header(ctx, f"period-end risk ({period}, {method}, "
                          f"{human_int(iterations)} runs)")

        if data["insufficient_data"]:
            ctx.out("  no spend observed yet; every simulated day stays at zero")
            return 0

        ctx.out(f"  spent so far   {human_usd(data['spent_usd'])} over "
                f"{human_int(len(data['values']))} day(s) "
                f"({data['zero_days']} with no usage)")
        ctx.out(f"  daily mean     {human_usd(data['mean_usd'])} "
                f"(stdev {human_usd(data['stdev_usd'])})")
        ctx.out(f"  days left      {data['horizon_days']}")
        ctx.out("")
        ctx.out(f"  median end     {human_usd(data['percentiles'][50])}")
        ctx.out(f"  P90 end        {human_usd(data['p90_total_usd'])}")
        ctx.out(f"  P95 end        {human_usd(data['percentiles'][95])}")

        if data["exceed_probability"] is None:
            ctx.out("")
            ctx.out("  pass --limit to get an overspend probability")
        else:
            ctx.out("")
            ctx.out(f"  limit          {human_usd(limit)}")
            ctx.out(f"  overspend risk {data['exceed_probability'] * 100:.1f}% "
                    f"of runs")
            self._cut_lines(ctx, data["cut"])

        if data["clamped_days"]:
            ctx.out("")
            ctx.warn(f"{human_int(data['clamped_days'])} simulated day(s) drew "
                     "a negative cost and were clamped to zero; the normal "
                     "method is a poor fit for this series")
        return 0

    def _distribution(self, ctx: PluginContext, records, period, method,
                      iterations, limit, seed) -> int:
        data = simulation.simulate(records, ctx.catalog, period=period,
                                   method=method, iterations=iterations,
                                   limit_usd=limit, seed=seed)
        self._header(ctx, f"simulated period-end distribution ({method})")
        if data["insufficient_data"]:
            ctx.out("  no spend observed yet; the distribution is a single point")
            return 0

        # Bars are scaled from zero, so a narrow distribution genuinely renders
        # as a set of near-equal bars -- that is the data, not a chart defect.
        top = max(data["percentiles"].values(), default=0.0)
        for pct in simulation.REPORTED_PERCENTILES:
            value = data["percentiles"][pct]
            marker = "  <- limit" if limit > 0 and value <= limit else ""
            ctx.out(f"  P{pct:<3} {human_usd(value):>12}  "
                    f"{bar(value, top, width=22)}{marker}")
        ctx.out("")
        ctx.out(f"  mean {human_usd(data['mean_total_usd'])}  "
                f"seed {data['seed']}  runs {human_int(data['iterations'])}")
        return 0

    def _methods(self, ctx: PluginContext, records, period, method, iterations,
                 limit, seed) -> int:
        data = simulation.compare_methods(records, ctx.catalog, period=period,
                                          limit_usd=limit,
                                          iterations=iterations, seed=seed)
        self._header(ctx, "bootstrap vs normal")
        rows = []
        for name in simulation.METHODS:
            run = data["runs"][name]
            if run["insufficient_data"]:
                ctx.out(f"  {name:<10} no data")
                continue
            rows.append((name, run))
            ctx.out(f"  {name:<10} median {human_usd(run['percentiles'][50]):>12}  "
                    f"P90 {human_usd(run['p90_total_usd']):>12}")
        if len(rows) < 2:
            return 0
        ctx.out("")
        ctx.out(f"  P90 spread     {human_usd(data['p90_spread_usd'])} "
                f"({data['p90_spread_pct']:.1f}%) -- {data['agreement']}")
        if data["agreement"] == "divergent":
            ctx.warn("the two methods disagree; the answer depends on "
                     "assumptions more than on data")
        else:
            ctx.out("  the methods agree, so the uncertainty estimate is stable")
        return 0

    def _budget(self, ctx: PluginContext, records, period, method, iterations,
                limit, seed) -> int:
        if limit <= 0:
            ctx.warn("--view budget requires --limit <usd>")
            return 2
        data = simulation.simulate(records, ctx.catalog, period=period,
                                   method=method, iterations=iterations,
                                   limit_usd=limit, seed=seed)
        self._header(ctx, f"budget risk against {human_usd(limit)}")
        if data["insufficient_data"]:
            ctx.out("  no spend observed yet; no risk can be assessed")
            return 0

        self._risk_lines(ctx, data)
        ctx.out("")
        risk = data["exceed_probability"]
        if risk >= 0.5:
            ctx.out("  more than half of simulated futures breach the limit; "
                    "treat this as the expected outcome, not a tail risk")
        elif risk >= 0.2:
            ctx.out("  a meaningful tail risk: the limit probably holds, but a "
                    "bad week would break it")
        else:
            ctx.out("  the limit holds in most simulated futures")
        self._cut_lines(ctx, simulation.required_cut(data["p90_total_usd"], limit))
        return 0


register(SimulatePlugin())
