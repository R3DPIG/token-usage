"""``peers`` -- how does this workspace compare with others?

    token-usage peers --from ../team-a,../team-b
    token-usage peers --from ../team-a --label alpha --metric output_ratio
    token-usage peers --from ../team-a,../team-b --view savings
    token-usage peers --from ../team-a,../team-b --view spread
    token-usage peers --metric cost_per_1m --min-tokens 50000 --include-self

``--from`` takes a comma-separated list; ``--label`` names them in the same order
and falls back to each directory's basename. ``--include-self`` adds the current
workspace as a peer called ``self``.

``report`` totals one dataset; ``compare`` contrasts one dataset with its own
past; ``merge`` combines datasets without contrasting them. This command puts
several workspaces side by side and normalises before it compares, because raw
totals only ever say who used the most tokens. Peers are read-only.
"""

from __future__ import annotations

from pathlib import Path

from token_usage import analytics, peers
from token_usage.registry import Plugin, PluginContext, register
from token_usage.storage import JsonStore
from token_usage.utils import bar, human_int, human_usd

_VIEWS = ("summary", "table", "rank", "spread", "savings", "outliers")
_METRIC_LABEL = {"cost_per_1m": "$/1M tok", "cost_per_call": "$/call",
                 "output_ratio": "out ratio", "cache_ratio": "cache",
                 "cost_usd": "spend", "tokens": "tokens"}
_METRIC_STYLE = {"cost_per_1m": "usd", "cost_per_call": "usd",
                 "cost_usd": "usd", "output_ratio": "ratio",
                 "cache_ratio": "ratio", "tokens": "int"}


def _fmt(metric: str, value: float) -> str:
    """Render a metric value the way that metric is normally read."""
    style = _METRIC_STYLE.get(metric, "ratio")
    if style == "usd":
        return human_usd(value)
    if style == "int":
        return human_int(int(value))
    return f"{value * 100:.1f}%"


class PeersPlugin(Plugin):
    name = "peers"
    help = "Compare several workspaces side by side, normalised for efficiency."
    aliases = ["bench", "benchmark"]
    usage = ("peers --from <home>[,<home>...] [--label <n>[,<n>...]] "
             "[--view summary|table|rank|spread|savings|outliers] "
             "[--metric cost_per_1m|cost_per_call|output_ratio|cache_ratio] "
             "[--min-tokens <n>] [--include-self]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2
        metric = args.get("metric", "cost_per_1m")
        if metric not in peers.METRICS:
            ctx.warn(f"--metric must be one of {peers.METRICS}")
            return 2

        sources = self._collect(ctx)
        if not sources:
            ctx.out("no peers in scope")
            ctx.out("hint: pass --from <home> for each workspace to compare")
            return 0

        min_tokens = args.get_int("min_tokens", peers.MIN_BENCHMARK_TOKENS)
        result = peers.summarize(sources, ctx.catalog, min_tokens=min_tokens)

        handler = {"summary": self._summary, "table": self._table,
                   "rank": self._rank, "spread": self._spread,
                   "savings": self._savings, "outliers": self._outliers}[view]
        return handler(ctx, result, metric)

    def _collect(self, ctx: PluginContext):
        """Read every ``--from`` workspace into a named record list.

        ``--from`` may repeat; ``--label`` names them in order. A missing
        workspace is reported and skipped rather than aborting the comparison,
        because one bad path should not hide the peers that did load.
        """
        args = ctx.args
        homes = args.get_list("from")
        labels = args.get_list("label")
        sources = {}
        for index, home in enumerate(homes):
            store = JsonStore(Path(home).expanduser())
            if not store.usage_path.exists():
                ctx.warn(f"no usage.jsonl under {store.home}; skipping")
                continue
            name = labels[index] if index < len(labels) else store.home.name
            if name in sources:
                name = f"{name}-{index}"
            sources[name] = store.load_records()

        if args.get_bool("include_self"):
            sources = {"self": analytics.filter_records(ctx.records()), **sources}
        return sources

    # -- views -------------------------------------------------------------
    @staticmethod
    def _pool(result) -> list:
        """Peers usable as a benchmark.

        Every ranking view works from this pool so the token floor is applied in
        exactly one place; the table view deliberately shows all peers instead.
        """
        return [row for row in result["profiles"] if row["eligible"]]

    def _header(self, ctx: PluginContext, result, metric: str) -> None:
        ctx.out(f"peer comparison ({metric})")
        ctx.out("=" * 70)
        ctx.out(f"  peers     {result['total_peers']} "
                f"({result['eligible_count']} eligible as benchmark)")
        ctx.out(f"  combined  {human_usd(result['total_usd'])} over "
                f"{human_int(result['total_tokens'])} tokens")
        ctx.out(f"  floor     {human_int(result['min_tokens'])} tokens "
                f"for a peer to be benchmarked")

    def _summary(self, ctx: PluginContext, result, metric: str) -> int:
        self._header(ctx, result, metric)
        unit = result["unit_cost"]
        ctx.out("")
        ctx.out(f"  unit cost  min {human_usd(unit['min'])}  "
                f"median {human_usd(unit['median'])}  "
                f"max {human_usd(unit['max'])}")
        if unit["count"] > 1:
            ctx.out(f"  the priciest peer pays {unit['multiple']}x the cheapest "
                    f"per token (best: {unit['best']}, worst: {unit['worst']})")
        for text in result["warnings"]:
            ctx.out("")
            ctx.warn(text)
        if not result["comparable"]:
            ctx.out("")
            ctx.out("  rankings below are shown for completeness only")
        return 0

    def _table(self, ctx: PluginContext, result, metric: str) -> int:
        self._header(ctx, result, metric)
        ctx.out("")
        ctx.out(f"  {'peer':<14}{'spend':>12}{'tokens':>13}{'$/1M':>11}"
                f"{'$/call':>10}{'out%':>7}{'cache':>7}")
        for row in result["profiles"]:
            flag = " " if row["eligible"] else "*"
            ctx.out(f" {flag}{row['peer']:<14}{human_usd(row['cost_usd']):>12}"
                    f"{human_int(row['tokens']):>13}"
                    f"{human_usd(row['cost_per_1m']):>11}"
                    f"{human_usd(row['cost_per_call']):>10}"
                    f"{row['output_ratio'] * 100:>6.1f}%"
                    f"{row['cache_ratio'] * 100:>6.1f}%")
        if any(not row["eligible"] for row in result["profiles"]):
            ctx.out("")
            ctx.out("  * below the token floor: listed, not used as a benchmark")
        return 0

    def _rank(self, ctx: PluginContext, result, metric: str) -> int:
        self._header(ctx, result, metric)
        ctx.out("")
        rows = peers.rank(self._pool(result) or result["profiles"], metric)
        top = max((row[metric] for row in rows), default=0.0)
        for row in rows:
            ctx.out(f"  #{row['rank']:<3} {row['peer']:<14}"
                    f"{_fmt(metric, row[metric]):>12}  "
                    f"{bar(row[metric], top, width=18)}")
        ctx.out("")
        ctx.out(f"  ordered by {_METRIC_LABEL.get(metric, metric)}; "
                "percentile 100% is best")
        return 0

    def _spread(self, ctx: PluginContext, result, metric: str) -> int:
        self._header(ctx, result, metric)
        pool = self._pool(result)
        ctx.out("")
        if len(pool) < 2:
            ctx.out("  fewer than two eligible peers; no spread to measure")
            return 0
        ctx.out(f"  {'metric':<16}{'min':>12}{'median':>12}{'max':>12}"
                f"{'max/min':>9}{'cv':>8}")
        for name in peers.METRICS:
            data = peers.spread(pool, name)
            ctx.out(f"  {name:<16}{_fmt(name, data['min']):>12}"
                    f"{_fmt(name, data['median']):>12}"
                    f"{_fmt(name, data['max']):>12}"
                    f"{data['multiple']:>8.2f}x{data['cv']:>8.2f}")
        ctx.out("")
        ctx.out("  cv is the coefficient of variation: near zero means the "
                "peers look alike")
        return 0

    def _savings(self, ctx: PluginContext, result, metric: str) -> int:
        if metric not in ("cost_per_1m", "cost_per_call"):
            ctx.warn("--view savings supports --metric cost_per_1m or cost_per_call")
            return 2
        items = peers.savings_vs_best(result["profiles"], metric=metric)
        self._header(ctx, result, metric)
        ctx.out("")
        if not items:
            ctx.out("  no peers to reprice")
            return 0
        total = sum(item["savings_usd"] for item in items)
        ctx.out(f"  repricing every peer at the best observed rate would save "
                f"{human_usd(total)} across {len(items)} peer(s)")
        ctx.out("")
        for item in items:
            flag = "" if item["eligible"] else "  (below floor)"
            ctx.out(f"  {item['peer']:<14}{human_usd(item['current_usd']):>12} -> "
                    f"{human_usd(item['repriced_usd']):>12}  "
                    f"save {human_usd(item['savings_usd'])} "
                    f"({item['savings_pct']:.1f}%){flag}")
        ctx.out("")
        ctx.out(f"  benchmark: {items[0]['benchmark']} at "
                f"{_fmt(metric, items[0]['benchmark_value'])}")
        # The repricing is arithmetic, not a plan. Under incomparable workloads it
        # becomes absurd (a chat peer "saving" 99.7% by adopting an embedding
        # model's rate), so the caveat has to appear here and not only in summary.
        if not result["comparable"]:
            ctx.warn("these peers are not doing the same kind of work, so this "
                     "total is an arithmetic ceiling, not an achievable target")
        ctx.warn(items[0]["note"])
        return 0

    def _outliers(self, ctx: PluginContext, result, metric: str) -> int:
        pool = self._pool(result)
        self._header(ctx, result, metric)
        ctx.out("")
        if len(pool) < 3:
            ctx.out("  needs at least three eligible peers to judge outliers")
            return 0
        flagged = peers.outliers(pool, metric)
        if not flagged:
            ctx.out(f"  no peer is an outlier on {metric}")
            return 0
        for item in flagged:
            score = (f"z {item['score']:+.2f}" if item["basis"] == "mad"
                     else f"{item['score']:.1f}x median")
            ctx.out(f"  {item['peer']:<14}{_fmt(metric, item['value']):>12}  "
                    f"{score}  {item['direction']} vs median "
                    f"{_fmt(metric, item['median'])}")
        ctx.out("")
        ctx.out("  scored against the median and MAD, so one extreme peer "
                "cannot hide itself")
        return 0


register(PeersPlugin())
