"""``pricing`` -- inspect and edit the model price catalog.

    token-usage pricing --action list
    token-usage pricing --action set --model m --input-price 1.0 --output-price 2.0
"""

from __future__ import annotations

from token_usage.models import ModelPricing
from token_usage.pricing import default_catalog
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd


class PricingPlugin(Plugin):
    name = "pricing"
    help = "List or edit per-model prices (USD per 1M tokens)."
    aliases = ["prices"]
    usage = ("pricing [--action list|set|show|reset] [--model <m>] "
             "[--input-price <usd>] [--output-price <usd>] [--cached-price <usd>]")

    def run(self, ctx: PluginContext) -> int:
        handler = {"list": self._list, "ls": self._list, "show": self._show,
                   "display": self._show, "set": self._set, "add": self._set,
                   "reset": self._reset}.get(ctx.args.get("action", "list"))
        if handler is None:
            ctx.warn(f"unknown --action {ctx.args.get('action')!r}")
            ctx.out(self.usage)
            return 2
        return handler(ctx)

    def _list(self, ctx: PluginContext) -> int:
        header = f"{'model':<28} {'input/1M':>10} {'output/1M':>11} {'cached/1M':>11}"
        ctx.out(header)
        ctx.out("-" * len(header))
        for p in ctx.catalog.all():
            ctx.out(f"{p.model:<28} {p.input_per_1m:>10.4f} "
                    f"{p.output_per_1m:>11.4f} {p.cached_input_per_1m:>11.4f}")
        ctx.out("")
        ctx.out(f"default fallback model: {ctx.catalog.default_model or '(none)'}")
        return 0

    def _show(self, ctx: PluginContext) -> int:
        model = ctx.args.get("model")
        if not model:
            ctx.warn("--model is required for --action show")
            return 2
        p = ctx.catalog.get(model)
        if p is None:
            ctx.warn(f"no pricing for model {model!r}")
            return 1
        ctx.out(f"model              {p.model}")
        ctx.out(f"input / 1M         {human_usd(p.input_per_1m)}")
        ctx.out(f"output / 1M        {human_usd(p.output_per_1m)}")
        ctx.out(f"cached input / 1M  {human_usd(p.cached_input_per_1m)}")
        ctx.out(f"currency           {p.currency}")
        return 0

    def _set(self, ctx: PluginContext) -> int:
        model = ctx.args.get("model")
        if not model:
            ctx.warn("--model is required for --action set")
            return 2
        try:
            ctx.catalog.add(ModelPricing(
                model=model,
                input_per_1m=ctx.args.get_float("input_price", 0.0),
                output_per_1m=ctx.args.get_float("output_price", 0.0),
                cached_input_per_1m=ctx.args.get_float("cached_price", 0.0)))
        except ValueError as exc:
            ctx.warn(str(exc))
            return 2
        ctx.store.save_pricing(ctx.catalog)
        ctx.out(f"saved pricing for {model}: "
                f"in {human_usd(ctx.args.get_float('input_price', 0.0))}/1M, "
                f"out {human_usd(ctx.args.get_float('output_price', 0.0))}/1M")
        return 0

    def _reset(self, ctx: PluginContext) -> int:
        if ctx.store.pricing_path.exists():
            ctx.store.pricing_path.unlink()
        fresh = default_catalog()
        ctx.catalog._entries = fresh._entries  # noqa: SLF001 - in-place refresh
        ctx.catalog.default_model = fresh.default_model
        ctx.out("pricing overrides cleared; shipped defaults restored")
        return 0


register(PricingPlugin())
