"""``drivers`` -- why did spend change?

    token-usage drivers
    token-usage drivers --view models
    token-usage drivers --view mix
    token-usage drivers --view price
    token-usage drivers --period weekly

``compare`` tells you the bill moved. This command tells you whether you are
being charged more or simply doing more work -- two cases that need opposite
responses, and which a single delta cannot distinguish.

The four drivers always re-add to the observed change; ``--view summary`` prints
that check so the arithmetic is verifiable rather than trusted.
"""

from __future__ import annotations

from token_usage import analytics, drivers
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import bar, human_int, human_usd

_VIEWS = ("summary", "models", "mix", "price")


class DriversPlugin(Plugin):
    name = "drivers"
    help = "Decompose a spend change into volume, mix and price."
    aliases = ["driver", "why"]
    usage = ("drivers [--view summary|models|mix|price] "
             "[--period daily|weekly|monthly] [--project <p>] [--model <m>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2

        period = args.get("period", "monthly")
        if period not in drivers.COMPARABLE:
            ctx.warn(f"--period must be one of {drivers.COMPARABLE}")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            since=args.get("since"), until=args.get("until"))
        if not records:
            ctx.out("no records in scope; nothing to explain")
            return 0

        handler = {"summary": self._summary, "models": self._models,
                   "mix": self._mix, "price": self._price}[view]
        return handler(ctx, records, period)

    # -- views ------------------------------------------------------------
    def _header(self, ctx: PluginContext, title: str) -> None:
        ctx.out(title)
        ctx.out("=" * 64)

    def _summary(self, ctx: PluginContext, records, period: str) -> int:
        data = drivers.summarize(records, ctx.catalog, period=period)
        self._header(ctx, f"spend drivers ({period}, current vs previous)")

        if abs(data["delta_usd"]) < 1e-9:
            ctx.out("  spend is unchanged period over period")
            return 0

        for line in data["lines"]:
            ctx.out(f"  {line}")

        if data["no_baseline"]:
            # The four-way split would be arithmetically valid but meaningless:
            # with one side empty the whole delta lands in `interaction`.
            ctx.out("")
            ctx.out("  hint: --period daily or weekly usually gives a narrower "
                    "window with comparable history")
            return 0

        ctx.out("")
        ctx.out(f"  dominant driver  {data['dominant']}")
        ctx.out(f"  reconciliation   observed {human_usd(data['delta_usd'])}, "
                f"accounted {human_usd(data['accounted_usd'])}, "
                f"drift {human_usd(data['drift_usd'])}")
        if not data["balanced"]:
            ctx.warn("drivers do not re-add to the observed change")

        self._render_drivers(ctx, data["drivers"])
        if data["models"]:
            ctx.out("")
            ctx.out("  top movers:")
            for row in data["models"][:3]:
                ctx.out(f"    {row['model']:<24} "
                        f"{human_usd(row['delta_usd']):>12} "
                        f"({row['status']})")
        return 0

    def _render_drivers(self, ctx: PluginContext, values) -> None:
        """Draw a signed bar chart: right of centre is spend added."""
        ctx.out("")
        span = max((abs(value) for value in values.values()), default=0.0)
        for name in drivers.DRIVER_ORDER:
            value = values.get(name, 0.0)
            sign = "+" if value >= 0 else "-"
            ctx.out(f"  {name:<12} {human_usd(value):>12}  "
                    f"{sign} {bar(abs(value), span, width=18)}")

    def _models(self, ctx: PluginContext, records, period: str) -> int:
        rows = drivers.model_contributions(records, ctx.catalog, period=period)
        self._header(ctx, f"per-model contribution ({period})")
        if not rows:
            ctx.out("  no models in either period")
            return 0
        ctx.out(f"  {'model':<24}{'delta':>12}{'volume':>12}{'price':>12}  status")
        for row in rows:
            ctx.out(f"  {row['model']:<24}"
                    f"{human_usd(row['delta_usd']):>12}"
                    f"{human_usd(row['volume_effect_usd']):>12}"
                    f"{human_usd(row['price_effect_usd']):>12}  {row['status']}")
        assumed = [row["model"] for row in rows if row["assumed_price"]]
        if assumed:
            ctx.out("")
            ctx.out(f"  {len(assumed)} model(s) exist in only one period; their "
                    "price effect is not observable and is reported as 0")
        return 0

    def _mix(self, ctx: PluginContext, records, period: str) -> int:
        rows = drivers.mix_shift(records, ctx.catalog, period=period)
        self._header(ctx, f"token share by model ({period})")
        if not rows:
            ctx.out("  no models in either period")
            return 0
        for row in rows:
            arrow = "+" if row["share_delta"] >= 0 else "-"
            ctx.out(f"  {row['model']:<24} "
                    f"{row['share_previous'] * 100:>6.2f}% -> "
                    f"{row['share_current'] * 100:>6.2f}%  "
                    f"{arrow}{abs(row['share_delta']) * 100:.2f}pp")
        ctx.out("")
        ctx.out("  measured in tokens, not dollars, so price changes do not "
                "leak into the mix metric")
        return 0

    def _price(self, ctx: PluginContext, records, period: str) -> int:
        data = drivers.decompose(records, ctx.catalog, period=period)
        self._header(ctx, f"blended unit price ({period})")
        ctx.out(f"  previous  ${data['mean_price_previous']:.6f} per 1M tokens")
        ctx.out(f"  current   ${data['mean_price_current']:.6f} per 1M tokens")
        if data["no_baseline"]:
            ctx.out("")
            ctx.out("  only one period has data, so there is no price movement "
                    "to measure")
            return 0
        previous = data["mean_price_previous"]
        if previous == 0:
            ctx.out("")
            ctx.out("  the previous period recorded no spend, so no percentage "
                    "change is meaningful")
            return 0
        pct = (data["mean_price_current"] / previous - 1) * 100
        if abs(pct) < 0.01:
            ctx.out("")
            ctx.out("  blended price did not move; the change came from volume "
                    "or mix")
            return 0
        ctx.out(f"  movement  {pct:+.2f}%")
        ctx.out("")
        ctx.out("  a blended price moves when usage shifts toward dearer models, "
                "or when the cached share")
        ctx.out("  or input/output split changes -- it is not the same as a list "
                "price change")
        ctx.out(f"  the isolated unit-price driver is "
                f"{human_usd(data['drivers']['unit_price'])}; the rest of the "
                "movement is mix")
        return 0


register(DriversPlugin())
