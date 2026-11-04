"""``chargeback`` -- who pays for this spend?

    token-usage chargeback
    token-usage chargeback --view owners
    token-usage chargeback --view orphans
    token-usage chargeback --view suggest
    token-usage chargeback --view invoice --owner team-api
    token-usage chargeback --view rules
    token-usage chargeback --view rules --action add --owner team-api --project api
    token-usage chargeback --view rules --action remove --index 0

Attribution rules live in ``owners.json`` and are matched **first-match-wins**,
so the order in the file is the contract. A record no rule claims is reported as
unattributed rather than dropped -- a report that hides spend is worse than none.
"""

from __future__ import annotations

from token_usage import analytics, attribution
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import human_int, human_usd

_VIEWS = ("summary", "owners", "orphans", "suggest", "invoice", "rules", "reconcile")
_ACTIONS = ("list", "add", "remove")


class ChargebackPlugin(Plugin):
    name = "chargeback"
    help = "Attribute spend to owners with auditable rules."
    aliases = ["owners", "attribution"]
    usage = ("chargeback [--view summary|owners|orphans|suggest|invoice|rules|reconcile] "
             "[--owner <name>] [--action list|add|remove] [--index <n>] "
             "[--project <p>] [--model <m>] [--tag <t>] [--note-contains <text>] "
             "[--since <date>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2

        if view == "rules":
            return self._rules(ctx)

        rules = ctx.store.load_owners()
        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            tag=args.get("tag"), since=args.get("since"), until=args.get("until"))
        if not records:
            ctx.out("no records in scope; nothing to attribute")
            return 0

        handler = {"summary": self._summary, "owners": self._owners,
                   "orphans": self._orphans, "suggest": self._suggest,
                   "invoice": self._invoice, "reconcile": self._reconcile}[view]
        return handler(ctx, records, rules)

    # -- views ------------------------------------------------------------
    def _header(self, ctx: PluginContext, title: str) -> None:
        ctx.out(title)
        ctx.out("=" * 62)

    def _summary(self, ctx: PluginContext, records, rules) -> int:
        data = attribution.summarize(records, rules, ctx.catalog)
        rec = data["reconcile"]

        self._header(ctx, f"cost attribution over {len(records)} record(s)")
        ctx.out(f"  total spend    {human_usd(data['total_usd'])}")
        ctx.out(f"  attributed     {human_usd(data['attributed_usd'])} "
                f"({data['coverage'] * 100:.1f}% coverage)")
        ctx.out(f"  unattributed   {human_usd(data['unattributed_usd'])} "
                f"across {human_int(data['unattributed_records'])} call(s)")
        ctx.out(f"  balances       {'yes' if rec['balanced'] else 'NO'} "
                f"(drift {human_usd(rec['drift_usd'])})")

        if not rules:
            ctx.out("")
            ctx.out("  no rules defined yet -- every record is unattributed")
            ctx.out("  start with: chargeback --view suggest")
            return 0

        if data["owners"]:
            ctx.out("")
            ctx.out("  owners:")
            for row in data["owners"]:
                ctx.out(f"    {row['owner']:<20} {human_usd(row['cost_usd']):>12} "
                        f"{row['share'] * 100:>5.1f}%  "
                        f"{human_int(row['records']):>5} calls")

        if data["suggestions"]:
            ctx.out("")
            ctx.out(f"  {len(data['suggestions'])} project(s) still need a rule; "
                    "run --view suggest for proposals")
        return 0

    def _owners(self, ctx: PluginContext, records, rules) -> int:
        data = attribution.attribute(records, rules, ctx.catalog)
        self._header(ctx, "spend by owner")
        if not data["owners"]:
            ctx.out("  no owner matched any record")
            ctx.out(f"  unattributed: {human_usd(data['unattributed_usd'])}")
            return 0
        for row in data["owners"]:
            ctx.out(f"  {row['owner']:<20} {human_usd(row['cost_usd']):>12} "
                    f"{row['share'] * 100:>5.1f}%  "
                    f"{human_int(row['tokens']):>10} tokens")
            ctx.out(f"     {human_int(row['records'])} call(s) via "
                    f"{', '.join(row['rules'])}")
        ctx.out("")
        ctx.out(f"  unattributed {human_usd(data['unattributed_usd'])} "
                f"({(1 - data['coverage']) * 100:.1f}% of spend)")
        return 0

    def _orphans(self, ctx: PluginContext, records, rules) -> int:
        data = attribution.unattributed_records(records, rules, ctx.catalog,
                                                limit=20)
        self._header(ctx, "records no rule claimed")
        ctx.out(f"  {human_int(data['count'])} call(s) costing "
                f"{human_usd(data['cost_usd'])}")
        if not data["records"]:
            ctx.out("  every record is attributed")
            return 0
        ctx.out("")
        for item in data["records"]:
            tags = ",".join(item["tags"]) or "-"
            ctx.out(f"  {item['timestamp']}  {item['project']:<14} "
                    f"{item['model']:<22} {tags:<10} "
                    f"{human_usd(item['cost_usd'])}")
        return 0

    def _suggest(self, ctx: PluginContext, records, rules) -> int:
        items = attribution.suggest_rules(records, rules, ctx.catalog)
        self._header(ctx, "proposed rules for unattributed spend")
        if not items:
            ctx.out("  nothing to suggest -- coverage is complete")
            return 0
        for item in items:
            rule = item["proposed_rule"]
            ctx.out(f"  {item['project']:<16} {human_usd(item['cost_usd']):>12} "
                    f"{human_int(item['records']):>5} call(s)")
            ctx.out(f"     chargeback --view rules --action add "
                    f"--owner {rule['owner']} --project {rule['project']}")
            ctx.out(f"     models: {', '.join(item['models'])}")
        ctx.out("")
        ctx.out("  suggestions are proposals only; nothing was written")
        return 0

    def _invoice(self, ctx: PluginContext, records, rules) -> int:
        owner = ctx.args.get("owner")
        if not owner:
            ctx.warn("--view invoice requires --owner <name>")
            return 2
        data = attribution.invoice(records, rules, ctx.catalog, owner)
        self._header(ctx, f"invoice for {owner}")
        if not data["found"]:
            ctx.out(f"  no rule assigns any spend to {owner!r}")
            return 0
        ctx.out(f"  total  {human_usd(data['cost_usd'])} "
                f"over {human_int(data['records'])} call(s)")
        ctx.out("")
        for label, key, rows in (("by project", "project", data["by_project"]),
                                 ("by model", "model", data["by_model"])):
            ctx.out(f"  {label}:")
            for row in rows:
                ctx.out(f"    {row[key]:<22} {human_usd(row['cost_usd']):>12} "
                        f"{row['share'] * 100:>5.1f}%  "
                        f"{human_int(row['tokens']):>10} tokens")
        return 0

    def _reconcile(self, ctx: PluginContext, records, rules) -> int:
        data = attribution.reconcile(records, rules, ctx.catalog)
        self._header(ctx, "attribution reconciliation")
        ctx.out(f"  true total      {human_usd(data['total_usd'])}")
        ctx.out(f"  attributed      {human_usd(data['attributed_usd'])}")
        ctx.out(f"  unattributed    {human_usd(data['unattributed_usd'])}")
        ctx.out(f"  accounted       {human_usd(data['accounted_usd'])}")
        ctx.out(f"  drift           {human_usd(data['drift_usd'])}")
        ctx.out("")
        if data["balanced"]:
            ctx.out("  balanced: attributed + unattributed equals the true total")
            return 0
        ctx.warn("attribution does not balance; a split weight or rounding is "
                 "losing money somewhere")
        return 1

    # -- rule management --------------------------------------------------
    def _rules(self, ctx: PluginContext) -> int:
        args = ctx.args
        action = args.get("action", "list")
        if action not in _ACTIONS:
            ctx.warn(f"--action must be one of {_ACTIONS}")
            return 2

        rules = ctx.store.load_owners()
        if action == "list":
            return self._list_rules(ctx, rules)
        if action == "add":
            return self._add_rule(ctx, rules)
        return self._remove_rule(ctx, rules)

    def _list_rules(self, ctx: PluginContext, rules) -> int:
        self._header(ctx, "attribution rules (first match wins)")
        if not rules:
            ctx.out("  no rules defined")
            ctx.out("  add one: chargeback --view rules --action add "
                    "--owner team-api --project api")
            return 0
        for index, rule in enumerate(rules):
            predicates = " ".join(f"{name}={getattr(rule, name)}"
                                  for name in attribution.MATCH_FIELDS
                                  if getattr(rule, name) is not None)
            target = f"split {rule.split}" if rule.split else rule.owner
            ctx.out(f"  [{index}] {target:<24} {predicates or '(match all)'}")
        return 0

    def _add_rule(self, ctx: PluginContext, rules) -> int:
        args = ctx.args
        owner = args.get("owner")
        if not owner:
            ctx.warn("--action add requires --owner <name>")
            return 2
        try:
            rule = attribution.OwnerRule(
                owner=owner, project=args.get("project"),
                model=args.get("model"), tag=args.get("tag"),
                note_contains=args.get("note_contains"),
                label=args.get("label", ""))
        except ValueError as exc:
            ctx.warn(str(exc))
            return 2

        # A rule that matches nothing is almost always a typo; say so instead of
        # letting the user believe coverage improved.
        hits = sum(1 for record in ctx.records() if rule.matches(record))
        rules.append(rule)
        ctx.store.save_owners(rules)
        ctx.out(f"added rule [{len(rules) - 1}] -> {owner}")
        if hits == 0:
            ctx.warn("this rule matches no stored record; check the predicate")
        else:
            ctx.out(f"it currently claims {human_int(hits)} record(s)")
        return 0

    def _remove_rule(self, ctx: PluginContext, rules) -> int:
        index = ctx.args.get_int("index", -1)
        if index < 0 or index >= len(rules):
            ctx.warn(f"--index must be between 0 and {len(rules) - 1}")
            return 2
        removed = rules.pop(index)
        ctx.store.save_owners(rules)
        ctx.out(f"removed rule [{index}] -> {removed.owner}")
        return 0


register(ChargebackPlugin())
