"""``quality`` -- how much should you trust these numbers?

    token-usage quality
    token-usage quality --view checks --min-severity warning
    token-usage quality --view detail --check duplicates
    token-usage quality --view next --top 3
    token-usage quality --project api --strict

``doctor`` answers "is anything obviously broken" with a pass/fail per probe.
This command answers the harder question a person about to publish a figure has:
*how good is this dataset, which specific defect costs the most, and what command
fixes it?* Every point of the score is attributable to one named check, so the
number can always be expanded into reasons.

Exit codes: 0 reportable, 1 critical defects or a score below 60, 2 bad
arguments. ``--strict`` fails on any failing check, which is what a CI gate wants.
"""

from __future__ import annotations

from token_usage import analytics, quality
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import bar, human_int, human_usd

_VIEWS = ("summary", "checks", "detail", "next")
_MARK = {"critical": "[CRIT]", "warning": "[WARN]", "notice": "[note]"}
# Below this score the dataset is not a sound basis for a spend report.
_FAIL_SCORE = 60.0


class QualityPlugin(Plugin):
    name = "quality"
    help = "Score data quality and name the defects behind the score."
    aliases = ["qa", "audit"]
    usage = ("quality [--view summary|checks|detail|next] [--check <id>] "
             "[--min-severity critical|warning|notice] [--top <n>] "
             "[--project <p>] [--model <m>] [--since <date>] [--strict]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2
        min_severity = args.get("min_severity")
        if min_severity is not None and min_severity not in quality.SEVERITIES:
            ctx.warn(f"--min-severity must be one of {quality.SEVERITIES}")
            return 2
        check_id = args.get("check")
        if check_id is not None and check_id not in quality.CHECK_IDS:
            ctx.warn(f"--check must be one of {quality.CHECK_IDS}")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            since=args.get("since"), until=args.get("until"))
        result = quality.audit(records, ctx.catalog)
        if result["empty"]:
            ctx.out("no records in scope; nothing to audit")
            ctx.out("hint: run 'sample' for demo data, or 'import --file <path>'")
            return 0

        handler = {"summary": self._summary, "checks": self._checks,
                   "detail": self._detail, "next": self._next}[view]
        return handler(ctx, result, check_id, min_severity,
                       args.get_int("top", 5), args)

    # -- shared rendering --------------------------------------------------
    def _header(self, ctx: PluginContext, result) -> None:
        ctx.out("data quality audit")
        ctx.out("=" * 66)
        ctx.out(f"  records   {human_int(result['total_records'])}")
        ctx.out(f"  spend     {human_usd(result['total_usd'])}")
        ctx.out(f"  score     {result['score']:.1f} / 100  ({result['grade']})")
        ctx.out(f"  {'':<9}  {bar(result['score'], 100.0, width=34)}")
        ctx.out(f"  findings  {result['critical']} critical, "
                f"{result['warnings']} warning, {result['notices']} notice")

    @staticmethod
    def _sample_text(sample: dict) -> str:
        """Render a sample row as ``k=v`` pairs, skipping empty values."""
        return "  ".join(f"{key}={value}" for key, value in sample.items()
                         if value not in ("", None))

    # -- views -------------------------------------------------------------
    def _summary(self, ctx: PluginContext, result, check_id, min_severity,
                 top, args) -> int:
        self._header(ctx, result)
        ctx.out("")
        if not result["failing"]:
            ctx.out("  every check passed; this dataset is safe to report from")
            return 0
        ctx.out("  worst findings (fix in this order):")
        for item in quality.next_steps(result, limit=3):
            ctx.out(f"    {_MARK[item['severity']]} {item['summary']}")
            ctx.out(f"           -{item['penalty']:.1f} pts  {item['fix']}")
        ctx.out("")
        ctx.out("  run 'quality --view checks' for the full list")
        return self._exit_code(result, args)

    def _checks(self, ctx: PluginContext, result, check_id, min_severity,
                top, args) -> int:
        checks = quality.filter_checks(result["checks"], check_id=check_id,
                                       min_severity=min_severity)
        self._header(ctx, result)
        ctx.out("")
        if not checks:
            ctx.out(f"  no check matches id={check_id!r} "
                    f"severity>={min_severity!r}")
            return 0
        for check in checks:
            mark = "[ ok ]" if check["ok"] else _MARK[check["severity"]]
            cost = "" if check["ok"] else f"  -{check['penalty']:.1f}"
            ctx.out(f"  {mark} {check['id']:<20} {check['summary']}{cost}")
        if len(checks) != len(result["checks"]):
            ctx.out("")
            ctx.out(f"  showing {len(checks)} of {len(result['checks'])} check(s); "
                    f"the score still counts all of them")
        return self._exit_code(result, args)

    def _detail(self, ctx: PluginContext, result, check_id, min_severity,
                top, args) -> int:
        if check_id is None:
            ctx.warn("--view detail requires --check <id>")
            ctx.out(f"known checks: {', '.join(quality.CHECK_IDS)}")
            return 2
        check = next(c for c in result["checks"] if c["id"] == check_id)
        self._header(ctx, result)
        ctx.out("")
        ctx.out(f"  check     {check['id']} -- {check['title']}")
        ctx.out(f"  severity  {check['severity']} (max deduction "
                f"{quality.MAX_PENALTY[check['severity']]:.0f} pts)")
        ctx.out(f"  status    {'pass' if check['ok'] else 'fail'}")
        ctx.out(f"  affected  {human_int(check['affected'])} record(s), "
                f"{check['share'] * 100:.1f}% of the scoped data")
        ctx.out(f"  points    -{check['penalty']:.1f}")
        ctx.out("")
        ctx.out(f"  {check['summary']}")
        if check["ok"]:
            return 0
        ctx.out("")
        ctx.out(f"  fix       {check['fix']}")
        for sample in check["samples"]:
            ctx.out(f"  sample    {self._sample_text(sample)}")
        return self._exit_code(result, args)

    def _next(self, ctx: PluginContext, result, check_id, min_severity,
              top, args) -> int:
        self._header(ctx, result)
        ctx.out("")
        if not result["failing"]:
            ctx.out("  nothing to fix")
            return 0
        steps = quality.next_steps(result, limit=max(1, top))
        total = sum(item["penalty"] for item in steps)
        ctx.out(f"  ranked worklist ({len(steps)} of {len(result['failing'])}):")
        for rank, item in enumerate(steps, start=1):
            ctx.out("")
            ctx.out(f"  {rank}. {_MARK[item['severity']]} {item['id']}"
                    f"  -{item['penalty']:.1f} pts")
            ctx.out(f"     {item['summary']}")
            ctx.out(f"     fix: {item['fix']}")
        ctx.out("")
        ctx.out(f"  completing these recovers {total:.1f} point(s), "
                f"to {min(100.0, result['score'] + total):.1f}/100")
        if len(steps) < len(result["failing"]):
            ctx.out(f"  {len(result['failing']) - len(steps)} further finding(s) "
                    f"remain; raise --top to see them")
        return self._exit_code(result, args)

    # -- exit policy -------------------------------------------------------
    @staticmethod
    def _exit_code(result, args) -> int:
        """Non-zero when the dataset is not reportable.

        ``--strict`` fails on any finding at all. Without it, only critical
        defects or a score below the reporting floor fail, so a few advisory
        notices do not break an existing pipeline.
        """
        if args.get_bool("strict"):
            return 1 if result["failing"] else 0
        return 1 if result["critical"] > 0 or result["score"] < _FAIL_SCORE else 0


register(QualityPlugin())
