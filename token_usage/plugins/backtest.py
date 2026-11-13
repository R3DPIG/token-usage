"""``backtest`` -- has the forecast ever actually been right?

    token-usage backtest
    token-usage backtest --window 7 --view samples
    token-usage backtest --project api --window 14 --fail-on unreliable
    token-usage backtest --view worst

``forecast`` prints one number and no error bar. This command walks the same
predictor through history: fit on everything before a cutoff, predict the next
window, then grade it against what was really spent. The output is the answer
to "can I trust this number", not another number.

Read-only: records are read, nothing is written.
"""

from __future__ import annotations

from token_usage import analytics, backtest
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd

_VIEWS = ("summary", "samples", "worst")
# Grades a --fail-on gate treats as disqualifying.
_FAIL_ON = {"unreliable": {"unreliable"},
            "loose": {"loose", "unreliable"},
            "never": set()}


class BacktestPlugin(Plugin):
    name = "backtest"
    help = "Score the forecast against history so its error is known."
    aliases = ["bt", "accuracy"]
    usage = ("backtest [--window <days>] [--view summary|samples|worst] "
             "[--project <p>] [--model <m>] [--fail-on unreliable|loose|never]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2
        fail_on = str(args.get("fail_on", "never"))
        if fail_on not in _FAIL_ON:
            ctx.warn(f"--fail-on must be one of {tuple(_FAIL_ON)}")
            return 2
        window = args.get_int("window", 7)
        if window < 1:
            ctx.warn("--window must be >= 1 day")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"))
        result = backtest.summarize(records, ctx.catalog, window_days=window)
        {"summary": self._summary, "samples": self._samples,
         "worst": self._worst}[view](ctx, result)

        if fail_on != "never" and result["grade"] in _FAIL_ON[fail_on]:
            ctx.warn(f"--fail-on {fail_on}: forecast grade is {result['grade']!r}")
            return 1
        return 0

    # -- views -------------------------------------------------------------
    def _header(self, ctx: PluginContext, result) -> None:
        metrics = result["metrics"]
        ctx.out(f"forecast backtest (grade: {result['grade']})")
        ctx.out("=" * 70)
        ctx.out(f"  window     {result['window_days']} day(s) per prediction")
        ctx.out(f"  samples    {len(result['samples'])} prediction(s) graded"
                + ("" if result["reliable"] else "  (below the reliability floor)"))
        if metrics:
            ctx.out(f"  bias       {human_usd(metrics['bias_usd'])}  "
                    f"({_pct(metrics['bias_pct'])})  positive means the "
                    "forecast ran high")
            ctx.out(f"  mae        {human_usd(metrics['mae_usd'])}   "
                    f"mape {_pct(metrics['mape'])}")
            ctx.out(f"  worst      {human_usd(metrics['worst_abs_usd'])} in a "
                    f"single window")
            ctx.out(f"  net        {human_usd(metrics['total_error_usd'])} across "
                    f"{human_usd(metrics['actual_total_usd'])} actually spent")

    def _caveats(self, ctx: PluginContext, result) -> None:
        if result["bias"]["kind"] != "clean":
            ctx.out("")
            ctx.out(f"  {result['bias']['kind']}: {result['bias']['note']}")
        for text in result["warnings"]:
            ctx.out("")
            ctx.warn(text)

    def _summary(self, ctx: PluginContext, result) -> None:
        self._header(ctx, result)
        if not result["samples"]:
            self._caveats(ctx, result)
            return
        ctx.out("")
        ctx.out("  walk-forward: each prediction is fitted only on earlier days "
                "and graded on the days after")
        self._caveats(ctx, result)

    def _samples(self, ctx: PluginContext, result) -> None:
        self._header(ctx, result)
        ctx.out("")
        if not result["samples"]:
            self._caveats(ctx, result)
            return
        ctx.out(f"  {'cutoff':<12}{'predicted':>12}{'actual':>12}"
                f"{'error':>12}{'error %':>10}")
        for row in result["samples"]:
            ctx.out(f"  {row['cutoff']:<12}{human_usd(row['predicted_usd']):>12}"
                    f"{human_usd(row['actual_usd']):>12}"
                    f"{human_usd(row['error_usd']):>12}"
                    f"{_pct(row['error_pct']):>10}")
        self._caveats(ctx, result)

    def _worst(self, ctx: PluginContext, result) -> None:
        self._header(ctx, result)
        ctx.out("")
        cases = backtest.worst_cases(result)
        if not cases:
            self._caveats(ctx, result)
            return
        ctx.out("  largest misses, worst first:")
        for row in cases:
            ctx.out(f"    {row['cutoff']}  predicted {human_usd(row['predicted_usd'])}"
                    f"  actual {human_usd(row['actual_usd'])}"
                    f"  off by {human_usd(row['abs_error_usd'])}")
        ctx.out("")
        ctx.out("  these are the windows to look at if the inputs need fixing")
        self._caveats(ctx, result)


def _pct(value) -> str:
    """Render an optional percentage; absence is stated, not shown as zero."""
    return "n/a" if value is None else f"{value:+.1f}%"


register(BacktestPlugin())
