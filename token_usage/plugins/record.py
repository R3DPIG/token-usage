"""``record`` -- append a usage record (the write path).

    token-usage record --model gpt-4o --input 12000 --output 3400 --cached 2000
"""

from __future__ import annotations

from token_usage.models import ModelPricing, UsageRecord
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd, utcnow_iso

_PRICE_KEYS = ("input_price", "output_price", "cached_price")


class RecordPlugin(Plugin):
    name = "record"
    help = "Append a token-usage record to the local history."
    usage = ("record --model <name> --input <n> --output <n> [--cached <n>] "
             "[--project <p>] [--tags a,b] [--note text] "
             "[--input-price <usd_per_1m>] [--output-price <usd_per_1m>] "
             "[--cached-price <usd_per_1m>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        model = args.get("model")
        if not model:
            ctx.warn("--model is required")
            ctx.out(self.usage)
            return 2

        input_tokens = args.get_int("input", 0)
        output_tokens = args.get_int("output", 0)
        if input_tokens == 0 and output_tokens == 0:
            ctx.warn("at least one of --input / --output must be non-zero")
            return 2

        # Persist inline price overrides before pricing, so this record's cost
        # matches what future reports will compute.
        if any(args.get(k) is not None for k in _PRICE_KEYS):
            self._apply_price_overrides(ctx, model)

        try:
            record = UsageRecord(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                timestamp=args.get("time") or utcnow_iso(),
                cached_input_tokens=args.get_int("cached", 0),
                project=args.get("project", "default"),
                tags=args.get_list("tags"),
                note=args.get("note", ""),
            )
        except ValueError as exc:
            ctx.warn(str(exc))
            return 2

        ctx.store.append_record(record)
        cost = ctx.catalog.cost_for_record(record)
        ctx.out(f"recorded {record.total_tokens:,} tokens for {model} "
                f"(~{human_usd(cost)}) in project {record.project!r}")
        return 0

    def _apply_price_overrides(self, ctx: PluginContext, model: str) -> None:
        current = ctx.catalog.get(model)
        args = ctx.args
        ctx.catalog.add(ModelPricing(
            model=model,
            input_per_1m=args.get_float("input_price", current.input_per_1m if current else 0.0),
            output_per_1m=args.get_float("output_price", current.output_per_1m if current else 0.0),
            cached_input_per_1m=args.get_float("cached_price", current.cached_input_per_1m if current else 0.0),
        ))
        ctx.store.save_pricing(ctx.catalog)


register(RecordPlugin())
