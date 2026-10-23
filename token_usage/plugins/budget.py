"""``budget`` -- define spending ceilings and check them.

    token-usage budget --action set --limit 50 --period monthly --project demo
"""

from __future__ import annotations

from token_usage import analytics
from token_usage.models import Budget
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd


class BudgetPlugin(Plugin):
    name = "budget"
    help = "Set and check spending budgets."
    aliases = ["budgets"]
    usage = ("budget [--action set|check|list|remove] [--limit <usd>] "
             "[--period daily|weekly|monthly|total] [--project <p>] "
             "[--label <name>] [--index <n>]")

    def run(self, ctx: PluginContext) -> int:
        action = ctx.args.get("action", "check")
        if action in ("set", "add"):
            return self._set(ctx) or self._check(ctx)
        if action in ("remove", "rm"):
            return self._remove(ctx) or self._check(ctx)
        handler = {"check": self._check, "list": self._list}.get(action)
        if handler is None:
            ctx.warn(f"unknown --action {action!r}")
            ctx.out(self.usage)
            return 2
        return handler(ctx)

    def _set(self, ctx: PluginContext) -> int:
        limit = ctx.args.get_float("limit", 0.0)
        if limit <= 0:
            ctx.warn("--limit must be a positive number of USD")
            return 2
        try:
            budget = Budget(limit_usd=limit, period=ctx.args.get("period", "monthly"),
                            project=ctx.args.get("project"), label=ctx.args.get("label", ""))
        except ValueError as exc:
            ctx.warn(str(exc))
            return 2
        budgets = ctx.store.load_budgets()
        budgets.append(budget)
        ctx.store.save_budgets(budgets)
        ctx.out(f"budget #{len(budgets) - 1} set: {human_usd(limit)} / {budget.period}")
        return 0

    def _remove(self, ctx: PluginContext) -> int:
        budgets = ctx.store.load_budgets()
        index = ctx.args.get_int("index", -1)
        if not budgets:
            ctx.warn("no budgets to remove")
            return 1
        if index < 0 or index >= len(budgets):
            ctx.warn(f"--index must be between 0 and {len(budgets) - 1}")
            return 2
        removed = budgets.pop(index)
        ctx.store.save_budgets(budgets)
        ctx.out(f"removed budget #{index} ({human_usd(removed.limit_usd)})")
        return 0

    def _list(self, ctx: PluginContext) -> int:
        budgets = ctx.store.load_budgets()
        if not budgets:
            ctx.out("no budgets configured")
            return 0
        for index, budget in enumerate(budgets):
            ctx.out(f"  #{index}  {human_usd(budget.limit_usd):>12} / {budget.period:<8} "
                    f"{budget.project or 'all projects'}")
        return 0

    def _check(self, ctx: PluginContext) -> int:
        budgets = ctx.store.load_budgets()
        if not budgets:
            ctx.out("no budgets configured; use --action set to add one")
            return 0
        results = analytics.evaluate_budgets(ctx.records(), ctx.catalog, budgets)
        symbols = {"ok": "[ ok ]", "warning": "[warn]", "exceeded": "[OVER]"}
        worst = "ok"
        rank = {"ok": 0, "warning": 1, "exceeded": 2}
        for item in results:
            ctx.out(f"{symbols[item['status']]} {item['label']:<16} {item['period']:<8} "
                    f"spent {human_usd(item['spent_usd'])} / {human_usd(item['limit_usd'])} "
                    f"({item['ratio'] * 100:.1f}%), remaining {human_usd(item['remaining_usd'])}")
            if rank[item["status"]] > rank[worst]:
                worst = item["status"]
        # Non-zero when any budget is exceeded, so scripts can react.
        return 1 if worst == "exceeded" else 0


register(BudgetPlugin())
