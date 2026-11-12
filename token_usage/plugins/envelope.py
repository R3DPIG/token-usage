"""``envelope`` -- review whether allocated budgets were actually honoured.

    token-usage envelope --action set --name api --limit 40 --scope project:api
    token-usage envelope --action list
    token-usage envelope --action review
    token-usage envelope --action review --view next --fail-on over
    token-usage envelope --action remove --name api

``quota`` answers "how should I split a budget?"; ``budget`` answers "am I about
to break one?"; this command answers the question that only becomes askable
later: *the period is half over -- is each allocation actually being spent the
way we planned, and was the plan itself sane?*

Envelopes live in ``<home>/envelopes.json`` as plain data. ``review`` is
read-only; only ``set``/``remove`` write.
"""

from __future__ import annotations

from token_usage import analytics, envelope
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_usd

_STATUS_LABEL = {envelope.STATUS_OK: "ok",
                 envelope.STATUS_WATCH: "watch",
                 envelope.STATUS_OVER: "OVER",
                 envelope.STATUS_HEADING_OVER: "heading over",
                 envelope.STATUS_NO_ACTIVITY: "no activity"}
# Which statuses a --fail-on gate treats as disqualifying.
_FAIL_ON = {"over": {envelope.STATUS_OVER},
            "attention": {envelope.STATUS_OVER, envelope.STATUS_HEADING_OVER,
                          envelope.STATUS_NO_ACTIVITY},
            "never": set()}


class EnvelopePlugin(Plugin):
    name = "envelope"
    help = "Allocate spending envelopes and review them after the fact."
    aliases = ["envelopes", "env"]
    usage = ("envelope [--action list|set|remove|review] [--name <n>] "
             "[--limit <usd>] [--period daily|weekly|monthly|total] "
             "[--scope all|project:<p>|model:<m>|tag:<t>] [--note <text>] "
             "[--view table|next] [--warn-at <0-1>] [--fail-on over|attention|never]")

    def run(self, ctx: PluginContext) -> int:
        action = ctx.args.get("action", "review")
        handler = {"list": self._list, "ls": self._list,
                   "set": self._set, "add": self._set,
                   "remove": self._remove, "rm": self._remove,
                   "review": self._review, "check": self._review}.get(action)
        if handler is None:
            ctx.warn(f"unknown --action {action!r}")
            ctx.out(self.usage)
            return 2
        return handler(ctx)

    # -- management --------------------------------------------------------
    def _list(self, ctx: PluginContext) -> int:
        envelopes = ctx.store.load_envelopes()
        if not envelopes:
            ctx.out("no spending envelopes defined")
            ctx.out("hint: envelope --action set --name api --limit 40 "
                    "--scope project:api")
            return 0
        ctx.out(f"{len(envelopes)} envelope(s) in {ctx.store.envelopes_path}")
        for item in envelopes:
            note = f"  # {item.note}" if item.note else ""
            ctx.out(f"  {item.name:<16}{human_usd(item.limit_usd):>12}  "
                    f"{item.period:<8}{item.scope_label():<20}{note}")
        return 0

    def _set(self, ctx: PluginContext) -> int:
        args = ctx.args
        name = args.get("name")
        limit = args.get_float("limit", 0.0)
        if not name:
            ctx.warn("--action set requires --name")
            ctx.out(self.usage)
            return 2
        if limit <= 0:
            ctx.warn("--limit must be a positive number of USD")
            return 2

        scope_type, scope_value = self._parse_scope(ctx, args.get("scope", "all"))
        if scope_type is None:
            return 2

        try:
            item = envelope.Envelope(
                name=str(name), limit_usd=limit,
                period=args.get("period", "monthly"), scope_type=scope_type,
                scope_value=scope_value, note=str(args.get("note") or ""))
        except ValueError as exc:
            ctx.warn(str(exc))
            return 2

        existing_all = ctx.store.load_envelopes()
        envelopes = [existing for existing in existing_all if existing.name != item.name]
        replaced = len(envelopes) != len(existing_all)
        envelopes.append(item)
        ctx.store.save_envelopes(sorted(envelopes, key=lambda e: e.name))
        verb = "replaced" if replaced else "added"
        ctx.out(f"{verb} envelope {item.name!r}: {human_usd(item.limit_usd)} "
                f"per {item.period} on {item.scope_label()}")
        return 0

    @staticmethod
    def _parse_scope(ctx: PluginContext, raw: str):
        """Parse ``project:api`` into ``(type, value)``; ``None`` on bad input."""
        text = str(raw or "all").strip()
        if text == "all":
            return "all", ""
        if ":" not in text:
            ctx.warn("--scope must be 'all' or '<type>:<value>' "
                     "(e.g. project:api, model:gpt-4o, tag:nightly)")
            return None, None
        scope_type, value = text.split(":", 1)
        scope_type, value = scope_type.strip(), value.strip()
        if scope_type not in envelope.SCOPE_TYPES or not value:
            ctx.warn(f"--scope type must be one of "
                     f"{[s for s in envelope.SCOPE_TYPES if s != 'all']}")
            return None, None
        return scope_type, value

    def _remove(self, ctx: PluginContext) -> int:
        name = ctx.args.get("name")
        if not name:
            ctx.warn("--action remove requires --name")
            return 2
        envelopes = ctx.store.load_envelopes()
        kept = [item for item in envelopes if item.name != name]
        if len(kept) == len(envelopes):
            ctx.warn(f"no envelope named {name!r}")
            return 1
        ctx.store.save_envelopes(kept)
        ctx.out(f"removed envelope {name!r} "
                f"({len(envelopes) - len(kept)} removed, {len(kept)} left)")
        return 0

    # -- review ------------------------------------------------------------
    def _review(self, ctx: PluginContext) -> int:
        envelopes = ctx.store.load_envelopes()
        if not envelopes:
            ctx.out("no spending envelopes defined; nothing to review")
            ctx.out("hint: envelope --action set --name api --limit 40 "
                    "--scope project:api")
            return 0

        # Every option is validated before any output, so a typo never buries a
        # full report under an error message.
        warn_at = ctx.args.get_float("warn_at", envelope.DEFAULT_WARN_AT)
        if not 0.0 < warn_at <= 1.0:
            ctx.warn("--warn-at must be in (0, 1]")
            return 2
        view = ctx.args.get("view", "table")
        if view not in ("table", "next"):
            ctx.warn("--view must be one of ('table', 'next')")
            return 2
        fail_on = str(ctx.args.get("fail_on", "never"))
        if fail_on not in _FAIL_ON:
            ctx.warn(f"--fail-on must be one of {tuple(_FAIL_ON)}")
            return 2

        records = analytics.filter_records(ctx.records())
        result = envelope.review(envelopes, records, ctx.catalog, warn_at=warn_at)
        (self._next if view == "next" else self._table)(ctx, result)

        triggered = _FAIL_ON[fail_on] & {row["status"] for row in result["rows"]}
        if triggered:
            ctx.warn(f"--fail-on {fail_on}: {len(triggered)} kind(s) of problem present")
            return 1
        return 0

    def _header(self, ctx: PluginContext, result) -> None:
        counts = result["counts"]
        ctx.out(f"envelope review ({result['status']})")
        ctx.out("=" * 78)
        ctx.out(f"  envelopes  {len(result['rows'])} "
                f"({counts['ok']} ok, {counts['watch']} watch, "
                f"{counts['over']} over, {counts['heading_over']} heading over, "
                f"{counts['no_activity']} no activity)")
        ctx.out(f"  limits     {human_usd(result['total_limits_usd'])} "
                f"spent against them {human_usd(result['total_spent_usd'])}")
        ctx.out(f"  coverage   {result['coverage'] * 100:.1f}% of workspace spend "
                f"({human_usd(result['governed_usd'])} of "
                f"{human_usd(result['workspace_spend_usd'])})")

    @staticmethod
    def _warnings(ctx: PluginContext, result) -> None:
        for text in result["warnings"]:
            ctx.out("")
            ctx.warn(text)

    def _table(self, ctx: PluginContext, result) -> None:
        self._header(ctx, result)
        ctx.out("")
        ctx.out(f"  {'envelope':<14}{'scope':<20}{'limit':>11}{'spent':>11}"
                f"{'used':>8}{'pace':>11}  status")
        for row in result["rows"]:
            pace = ("n/a" if row["pace_usd"] is None
                    else human_usd(row["pace_usd"]))
            ctx.out(f"  {row['envelope']:<14}{row['scope']:<20}"
                    f"{human_usd(row['limit_usd']):>11}"
                    f"{human_usd(row['spent_usd']):>11}"
                    f"{row['consumed'] * 100:>7.1f}%{pace:>11}"
                    f"  {_STATUS_LABEL[row['status']]}")
        ctx.out("")
        ctx.out("  pace is what an even spender would have used by now; "
                "'n/a' means the envelope has no reset")
        for row in result["rows"]:
            if row["status"] != envelope.STATUS_OK and row["note"]:
                ctx.out(f"  {row['envelope']}: {row['note']}")
        self._warnings(ctx, result)

    def _next(self, ctx: PluginContext, result) -> None:
        self._header(ctx, result)
        ctx.out("")
        steps = envelope.next_steps(result)
        if not steps:
            ctx.out("  every envelope is within its limit and tracking activity")
            return
        ctx.out(f"  {'envelope':<14}{'scope':<20}{'spent':>11}{'limit':>11}"
                f"{'over':>10}  status")
        for step in steps:
            ctx.out(f"  {step['envelope']:<14}{step['scope']:<20}"
                    f"{human_usd(step['spent_usd']):>11}"
                    f"{human_usd(step['limit_usd']):>11}"
                    f"{human_usd(step['over_usd']):>10}"
                    f"  {_STATUS_LABEL[step['status']]}")
        ctx.out("")
        for step in steps:
            ctx.out(f"  {step['envelope']}: {step['action']}")
        self._warnings(ctx, result)


register(EnvelopePlugin())
