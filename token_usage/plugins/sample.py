"""``sample`` -- generate deterministic demo data (seeded, so tests are stable).

    token-usage sample --days 14 --per-day 6 --project demo
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from token_usage.models import UsageRecord
from token_usage.registry import Plugin, PluginContext, register
from token_usage.utils import day_key

MODELS = [
    ("gpt-4o", 800, 2400, 300, 90),
    ("gpt-4o-mini", 500, 1500, 200, 60),
    ("claude-3-5-sonnet", 1200, 3200, 400, 120),
    ("claude-3-5-haiku", 400, 1100, 150, 50),
    ("gpt-3.5-turbo", 300, 900, 100, 40),
]
TAG_POOL = ["api", "batch", "chat", "eval", "rag", "agent"]


class SamplePlugin(Plugin):
    name = "sample"
    help = "Generate deterministic demo usage data."
    aliases = ["seed"]
    usage = "sample [--days <n>] [--per-day <n>] [--project <p>] [--seed <n>]"

    def run(self, ctx: PluginContext) -> int:
        days = ctx.args.get_int("days", 7)
        per_day = ctx.args.get_int("per_day", 5)
        if days <= 0 or per_day <= 0:
            ctx.warn("--days and --per-day must be > 0")
            return 2

        rng = random.Random(ctx.args.get_int("seed", 42))
        project = ctx.args.get("project", "demo")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        records = []
        for offset in range(days):
            day = now - timedelta(days=days - 1 - offset)
            # Gentle ramp plus weekend bump so forecast has something to show.
            factor = 1.0 + offset * 0.04 + (0.2 if day.weekday() >= 5 else 0.0)
            for _ in range(per_day):
                model, base_in, base_out, base_cached, spread = rng.choice(MODELS)
                records.append(UsageRecord(
                    model=model,
                    input_tokens=max(1, int((base_in + rng.randint(0, spread)) * factor)),
                    output_tokens=max(0, int((base_out + rng.randint(0, spread * 2)) * factor)),
                    cached_input_tokens=int(base_cached * rng.uniform(0.2, 0.9)),
                    timestamp=day.replace(hour=rng.randint(8, 20), minute=rng.randint(0, 59),
                                          second=rng.randint(0, 59)).strftime("%Y-%m-%dT%H:%M:%S"),
                    project=project,
                    tags=rng.sample(TAG_POOL, k=rng.randint(1, 2)),
                    note="generated sample",
                ))

        inserted = ctx.store.append_records(records)
        total = sum(ctx.catalog.cost_for_record(r) for r in records)
        ctx.out(f"inserted {inserted} sample record(s) across {days} day(s) "
                f"in project {project!r} (~${total:.4f})")
        ctx.out(f"date range: {day_key(now - timedelta(days=days - 1))} .. {day_key(now)}")
        return 0


register(SamplePlugin())
