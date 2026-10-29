"""``alert`` -- declarative alert rules over usage data.

    token-usage alert --action init
    token-usage alert --action add --name big-day --metric cost --operator gt \\
        --threshold 5 --dimension day --severity warning
    token-usage alert --action check
    token-usage alert --action check --fail-on critical   # CI gate

Rules live in ``<home>/alerts.json`` and are plain data, so they can also be
edited by hand. ``check`` exits non-zero when a rule at or above ``--fail-on``
severity is firing.
"""

from __future__ import annotations

from token_usage import alerts
from token_usage.registry import Plugin, PluginContext, register

_STATUS_MARK = {"firing": "[FIRE]", "ok": "[ ok ]", "unknown": "[ ?? ]",
                "disabled": "[ -- ]"}


class AlertPlugin(Plugin):
    name = "alert"
    help = "Define and check declarative alert rules."
    aliases = ["alerts"]
    usage = ("alert [--action list|add|remove|enable|disable|init|check|metrics] "
             "[--name <n>] [--metric <m>] [--operator gt|gte|lt|lte] "
             "[--threshold <v>] [--dimension total|model|project|tag|day] "
             "[--period daily|weekly|monthly|total] [--severity info|warning|critical] "
             "[--project <p>] [--model <m>] [--tag <t>] [--fail-on <severity>|never]")

    def run(self, ctx: PluginContext) -> int:
        action = ctx.args.get("action", "check")
        handler = {"list": self._list, "ls": self._list, "add": self._add,
                   "set": self._add, "remove": self._remove, "rm": self._remove,
                   "enable": self._enable, "disable": self._disable,
                   "init": self._init, "check": self._check,
                   "metrics": self._metrics}.get(action)
        if handler is None:
            ctx.warn(f"unknown --action {action!r}")
            ctx.out(self.usage)
            return 2
        return handler(ctx)

    # -- rule management --------------------------------------------------
    def _list(self, ctx: PluginContext) -> int:
        rules = ctx.store.load_alerts()
        if not rules:
            ctx.out("no alert rules configured")
            ctx.out("hint: alert --action init   (install a starter set)")
            return 0
        ctx.out(f"{len(rules)} rule(s) in {ctx.store.alerts_path}")
        for rule in rules:
            flag = "on " if rule.enabled else "off"
            note = f"  # {rule.description}" if rule.description else ""
            ctx.out(f"  [{flag}] {rule.name:<18} {rule.severity:<8} {rule.render()}{note}")
        return 0

    def _metrics(self, ctx: PluginContext) -> int:
        ctx.out("available metrics:")
        for metric in alerts.METRICS:
            ctx.out(f"  {metric}")
        ctx.out("")
        ctx.out(f"operators: {', '.join(alerts.OPERATORS)}")
        ctx.out(f"dimensions: {', '.join(alerts.DIMENSIONS)}")
        ctx.out(f"severities: {', '.join(alerts.SEVERITIES)}")
        return 0

    def _add(self, ctx: PluginContext) -> int:
        args = ctx.args
        name = args.get("name")
        metric = args.get("metric")
        operator = args.get("operator")
        if not name or not metric or not operator or args.get("threshold") is None:
            ctx.warn("--action add requires --name, --metric, --operator and --threshold")
            ctx.out(self.usage)
            return 2

        scope = {key: args.get(key) for key in ("project", "model", "tag")
                 if args.get(key)}
        try:
            rule = alerts.parse_rule_args(
                name=name, metric=metric, operator=operator,
                threshold=args.get_float("threshold", 0.0),
                dimension=args.get("dimension", "total"),
                scope=scope, period=args.get("period", "total"),
                severity=args.get("severity", "warning"),
                description=args.get("description", ""))
        except alerts.AlertError as exc:
            ctx.warn(str(exc))
            return 2

        rules = alerts.upsert_rule(ctx.store.load_alerts(), rule)
        ctx.store.save_alerts(rules)
        ctx.out(f"saved rule {rule.name!r}: {rule.render()}")
        return 0

    def _remove(self, ctx: PluginContext) -> int:
        name = ctx.args.get("name")
        if not name:
            ctx.warn("--action remove requires --name")
            return 2
        remaining, removed = alerts.remove_rule(ctx.store.load_alerts(), name)
        if removed is None:
            ctx.warn(f"rule {name!r} not found")
            return 1
        ctx.store.save_alerts(remaining)
        ctx.out(f"removed rule {name!r}")
        return 0

    def _enable(self, ctx: PluginContext) -> int:
        return self._toggle(ctx, True)

    def _disable(self, ctx: PluginContext) -> int:
        return self._toggle(ctx, False)

    def _toggle(self, ctx: PluginContext, enabled: bool) -> int:
        name = ctx.args.get("name")
        if not name:
            ctx.warn("--name is required")
            return 2
        rules = ctx.store.load_alerts()
        rule = alerts.find_rule(rules, name)
        if rule is None:
            ctx.warn(f"rule {name!r} not found")
            return 1
        rule.enabled = enabled
        ctx.store.save_alerts(rules)
        ctx.out(f"{'enabled' if enabled else 'disabled'} rule {name!r}")
        return 0

    def _init(self, ctx: PluginContext) -> int:
        existing = ctx.store.load_alerts()
        if existing and not ctx.args.get_bool("force"):
            ctx.warn(f"{len(existing)} rule(s) already exist; use --force to replace")
            return 1
        rules = alerts.default_rules()
        ctx.store.save_alerts(rules)
        ctx.out(f"installed {len(rules)} starter rule(s)")
        for rule in rules:
            ctx.out(f"  {rule.name:<18} {rule.render()}")
        return 0

    # -- evaluation -------------------------------------------------------
    def _check(self, ctx: PluginContext) -> int:
        rules = ctx.store.load_alerts()
        if not rules:
            ctx.out("no alert rules configured; nothing to check")
            return 0

        records = ctx.records()
        results = alerts.evaluate_all(rules, records, ctx.catalog)
        counts = alerts.summarize(results)

        ctx.out(f"checked {len(rules)} rule(s) against {len(records)} record(s)")
        ctx.out("")
        for result in results:
            mark = _STATUS_MARK.get(result["status"], "[????]")
            value = "-" if result["value"] is None else f"{result['value']:.4f}"
            ctx.out(f"  {mark} {result['rule']:<18} {result['bucket']:<16} "
                    f"{result['metric']:<18} {value:>12}  {result['detail']}")

        ctx.out("")
        summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        ctx.out(f"summary: {summary}")

        fail_on = ctx.args.get("fail_on", "critical")
        try:
            code = alerts.exit_code_for(results, fail_on=fail_on)
        except alerts.AlertError as exc:
            ctx.warn(str(exc))
            return 2
        if code:
            worst = alerts.worst_severity(results)
            ctx.warn(f"firing at severity {worst!r} (fail-on={fail_on})")
        return code


register(AlertPlugin())
