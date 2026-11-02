"""``pattern`` -- when does usage happen?

    token-usage pattern
    token-usage pattern --view weekday
    token-usage pattern --view hourly
    token-usage pattern --view trend --window 14
    token-usage pattern --view weekend

Answers scheduling questions that totals cannot: which weekday is expensive,
which hours carry the load, whether spend is trending, and whether usage is
steady or bursty.

Hour-of-day buckets are **UTC** (see :mod:`token_usage.patterns`).
"""

from __future__ import annotations

from token_usage import analytics, patterns
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import bar, human_int, human_usd

_VIEWS = ("summary", "weekday", "hourly", "trend", "weekend", "streaks", "burst")


class PatternPlugin(Plugin):
    name = "pattern"
    help = "Analyse when usage happens: weekday, hour, trend, burstiness."
    aliases = ["patterns", "when"]
    usage = ("pattern [--view summary|weekday|hourly|trend|weekend|streaks|burst] "
             "[--window <days>] [--project <p>] [--model <m>] [--since <date>]")

    def run(self, ctx: PluginContext) -> int:
        args = ctx.args
        view = args.get("view", "summary")
        if view not in _VIEWS:
            ctx.warn(f"--view must be one of {_VIEWS}")
            return 2

        records = analytics.filter_records(
            ctx.records(), project=args.get("project"), model=args.get("model"),
            since=args.get("since"), until=args.get("until"))
        if not records:
            ctx.out("no records in scope; nothing to analyse")
            return 0

        window = args.get_int("window", 7)
        if window < 1:
            ctx.warn("--window must be >= 1")
            return 2

        handler = {"summary": self._summary, "weekday": self._weekday,
                   "hourly": self._hourly, "trend": self._trend,
                   "weekend": self._weekend, "streaks": self._streaks,
                   "burst": self._burst}[view]
        return handler(ctx, records, window)

    # -- views ------------------------------------------------------------
    def _summary(self, ctx: PluginContext, records, window: int) -> int:
        data = patterns.summarize(records, ctx.catalog, window=window)
        ctx.out(f"temporal pattern over {data['days']} day(s)")
        ctx.out("=" * 56)

        trend = data["trend"]
        ctx.out(f"  trend        {trend['direction']:<8} "
                f"{trend['pct_per_day']:+.2f}%/day "
                f"(r2={trend['r_squared']:.2f}, {trend['samples']} days)")
        burst = data["burstiness"]
        ctx.out(f"  variability  {burst['classification']:<8} "
                f"cv={burst['cv']:.2f} (mean {human_usd(burst['mean_usd'])})")
        season = data["seasonality"]
        ratio = "n/a" if season["ratio"] is None else f"{season['ratio']:.2f}x"
        ctx.out(f"  seasonality  {season['strength']:<8} {ratio} "
                f"across weekdays")

        weekend = data["weekend"]
        ctx.out(f"  weekend      {human_usd(weekend['weekend_per_day_usd'])}/day vs "
                f"{human_usd(weekend['weekday_per_day_usd'])}/day on weekdays "
                f"({weekend['weekend_ratio']:.2f}x)")

        ctx.out("")
        ctx.out("  busiest weekdays:")
        for bucket in data["peak_weekdays"]:
            ctx.out(f"    {bucket['label']:<5} {human_usd(bucket['cost_usd']):>12} "
                    f"{human_int(bucket['records']):>6} calls")
        ctx.out("  busiest hours (UTC):")
        for bucket in data["peak_hours"]:
            ctx.out(f"    {bucket['label']:<5} {human_usd(bucket['cost_usd']):>12} "
                    f"{human_int(bucket['records']):>6} calls")

        streak = data["streaks"]
        if streak["current_direction"] != "none":
            ctx.out("")
            ctx.out(f"  current run  {streak['current_' + streak['current_direction']]} "
                    f"day(s) {streak['current_direction']} the mean "
                    f"({human_usd(streak['mean_usd'])})")
        return 0

    def _weekday(self, ctx: PluginContext, records, window: int) -> int:
        profile = patterns.weekday_profile(records, ctx.catalog)
        self._render_profile(ctx, profile, "weekday")
        season = patterns.seasonality_strength(profile)
        ctx.out("")
        if season["strength"] == "insufficient-data":
            ctx.out("  not enough distinct weekdays to judge seasonality")
        else:
            ctx.out(f"  seasonality {season['strength']}: {season['strongest']} is "
                    f"the most expensive, {season['weakest']} the cheapest")
        return 0

    def _hourly(self, ctx: PluginContext, records, window: int) -> int:
        profile = patterns.hourly_profile(records, ctx.catalog)
        self._render_profile(ctx, profile, "hour (UTC)")
        peaks = patterns.peak_buckets(profile, top=3)
        if peaks:
            ctx.out("")
            ctx.out("  peak hours: " + ", ".join(
                f"{b['label']} ({human_usd(b['cost_usd'])})" for b in peaks))
        return 0

    def _render_profile(self, ctx: PluginContext, profile, label: str) -> None:
        maximum = max((bucket["cost_usd"] for bucket in profile), default=0.0)
        ctx.out(f"cost by {label}")
        ctx.out("=" * 56)
        for bucket in profile:
            if bucket["records"] == 0:
                continue
            ctx.out(f"  {bucket['label']:<6} {human_usd(bucket['cost_usd']):>12} "
                    f"{human_int(bucket['records']):>6} calls  "
                    f"{bar(bucket['cost_usd'], maximum, width=20)}")
        empty = sum(1 for bucket in profile if bucket["records"] == 0)
        if empty:
            ctx.out(f"  ({empty} bucket(s) with no usage omitted)")

    def _trend(self, ctx: PluginContext, records, window: int) -> int:
        daily = patterns.daily_series(records, ctx.catalog)
        series = patterns.moving_average(daily, window=window)
        trend = patterns.trend_direction(daily, window=window)

        ctx.out(f"daily cost with {window}-day moving average")
        ctx.out("=" * 56)
        maximum = max((point["cost_usd"] for point in series), default=0.0)
        for point in series:
            ctx.out(f"  {point['day']}  {human_usd(point['cost_usd']):>12}  "
                    f"avg {human_usd(point['average_usd']):>12}  "
                    f"{bar(point['cost_usd'], maximum, width=16)}")

        ctx.out("")
        ctx.out(f"  direction   {trend['direction']}")
        ctx.out(f"  slope       {human_usd(trend['slope_usd_per_day'])}/day "
                f"({trend['pct_per_day']:+.2f}%)")
        ctx.out(f"  fit         r2={trend['r_squared']:.3f} over "
                f"{trend['samples']} day(s)")
        if trend["direction"] != "stable" and trend["r_squared"] < 0.3:
            ctx.warn("slope is steep but the fit is weak; treat this as noise "
                     "rather than a trend")
        return 0

    def _weekend(self, ctx: PluginContext, records, window: int) -> int:
        split = patterns.weekend_split(records, ctx.catalog)
        ctx.out("weekday vs weekend")
        ctx.out("=" * 56)
        ctx.out(f"  weekdays  {human_usd(split['weekday_cost_usd']):>12} over "
                f"{split['weekday_days']} day(s) = "
                f"{human_usd(split['weekday_per_day_usd'])}/day")
        ctx.out(f"  weekends  {human_usd(split['weekend_cost_usd']):>12} over "
                f"{split['weekend_days']} day(s) = "
                f"{human_usd(split['weekend_per_day_usd'])}/day")
        ctx.out("")
        if split["weekday_days"] == 0 or split["weekend_days"] == 0:
            ctx.out("  only one side of the week has data; ratio not meaningful")
            return 0
        ratio = split["weekend_ratio"]
        if ratio > 1.0:
            ctx.out(f"  weekends cost {ratio:.2f}x a weekday -- off-hours usage "
                    "is heavier than working hours")
        else:
            ctx.out(f"  weekends cost {ratio:.2f}x a weekday -- usage is "
                    "concentrated in the working week")
        return 0

    def _streaks(self, ctx: PluginContext, records, window: int) -> int:
        daily = patterns.daily_series(records, ctx.catalog)
        result = patterns.streaks(daily)
        ctx.out("consecutive runs above/below the daily mean")
        ctx.out("=" * 56)
        ctx.out(f"  mean daily cost   {human_usd(result['mean_usd'])}")
        ctx.out(f"  longest above     {result['longest_above']} day(s)")
        ctx.out(f"  longest below     {result['longest_below']} day(s)")
        if result["current_direction"] == "none":
            ctx.out("  current run       none")
        else:
            count = result["current_" + result["current_direction"]]
            ctx.out(f"  current run       {count} day(s) "
                    f"{result['current_direction']} the mean")
        if result["longest_above"] >= 5:
            ctx.warn(f"{result['longest_above']} consecutive expensive days "
                     "suggests a sustained change, not a one-off spike")
        return 0

    def _burst(self, ctx: PluginContext, records, window: int) -> int:
        daily = patterns.daily_series(records, ctx.catalog)
        result = patterns.burstiness(daily)
        ctx.out("daily variability")
        ctx.out("=" * 56)
        ctx.out(f"  classification  {result['classification']}")
        ctx.out(f"  mean            {human_usd(result['mean_usd'])}")
        ctx.out(f"  stdev           {human_usd(result['stdev_usd'])}")
        ctx.out(f"  cv              {result['cv']:.3f}")
        ctx.out(f"  samples         {result['samples']} day(s)")
        if result["classification"] == "bursty":
            ctx.out("")
            ctx.out("  bursty usage means averages hide the peaks; check "
                    "'anomaly' for the specific days")
        return 0


register(PatternPlugin())
