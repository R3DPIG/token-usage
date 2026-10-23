"""Small dependency-free helpers shared across the package.

Generic utilities only -- no LLM or pricing knowledge -- so edge cases (timezone
offsets, period boundaries, rounding) can be tested in isolation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


def utcnow_iso() -> str:
    """Current UTC time as a second-precision ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(ISO_FORMAT)


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string into an aware UTC datetime.

    Accepts a trailing ``Z`` and naive strings (assumed UTC), so values copied
    straight from logs work without pre-processing.
    """
    if value.endswith("Z"):
        value = value[:-1]
    try:
        dt = datetime.strptime(value, ISO_FORMAT)
    except ValueError:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def week_key(dt: datetime) -> str:
    """ISO week key, e.g. ``2024-W05``."""
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def period_start(period: str, now: Optional[datetime] = None) -> datetime:
    """Inclusive start of the named period; ``weekly`` starts Monday (ISO)."""
    now = now or datetime.now(timezone.utc)
    if period == "daily":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "weekly":
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=midnight.weekday())
    if period == "monthly":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if period == "total":
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    raise ValueError(f"unknown period: {period!r}")


def round_usd(value: float, places: int = 6) -> float:
    """Round a USD amount; micro-cent precision avoids tiny costs showing as 0."""
    return round(value + 0.0, places)


def human_int(value: int) -> str:
    return f"{value:,}"


def human_usd(value: float) -> str:
    return f"${value:,.4f}"


def bar(value: float, maximum: float, width: int = 24) -> str:
    """Render an ASCII bar; empty when ``maximum`` is non-positive."""
    if maximum <= 0:
        return ""
    filled = max(0, min(width, int(round(width * (value / maximum)))))
    return "#" * filled
