"""Core data structures: records, pricing entries and budgets.

Plain dataclasses with explicit ``to_dict`` / ``from_dict`` round-tripping. The
persistence format is deliberately human-readable. Money is ``float`` USD;
timestamps are ISO-8601 UTC strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelPricing:
    """Published price for one model, USD per one million tokens."""

    model: str
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float = 0.0
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("ModelPricing.model must be a non-empty string")
        for name, value in (("input_per_1m", self.input_per_1m),
                            ("output_per_1m", self.output_per_1m),
                            ("cached_input_per_1m", self.cached_input_per_1m)):
            if value < 0:
                raise ValueError(f"ModelPricing.{name} must be >= 0, got {value}")

    def per_1k(self) -> Dict[str, float]:
        """Return per-1K prices for human-readable tables."""
        return {
            "input_per_1k": self.input_per_1m / 1000.0,
            "output_per_1k": self.output_per_1m / 1000.0,
            "cached_input_per_1k": self.cached_input_per_1m / 1000.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelPricing":
        return cls(model=data["model"], input_per_1m=float(data["input_per_1m"]),
                   output_per_1m=float(data["output_per_1m"]),
                   cached_input_per_1m=float(data.get("cached_input_per_1m", 0.0)),
                   currency=data.get("currency", "USD"))


@dataclass
class UsageRecord:
    """One observed unit of LLM consumption.

    Deliberately flat: hierarchies (by project/tag/day) are computed in
    :mod:`token_usage.analytics`, so storage never changes when a roll-up is added.
    """

    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str
    cached_input_tokens: int = 0
    project: str = "default"
    tags: List[str] = field(default_factory=list)
    note: str = ""
    cost_usd: Optional[float] = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be >= 0")
        if self.cached_input_tokens < 0:
            raise ValueError("cached_input_tokens must be >= 0")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(
                "cached_input_tokens cannot exceed input_tokens "
                f"({self.cached_input_tokens} > {self.input_tokens})"
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def billable_input_tokens(self) -> int:
        """Input tokens not served from cache (billed at the standard rate)."""
        return self.input_tokens - self.cached_input_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageRecord":
        return cls(model=data["model"], input_tokens=int(data["input_tokens"]),
                   output_tokens=int(data["output_tokens"]), timestamp=data["timestamp"],
                   cached_input_tokens=int(data.get("cached_input_tokens", 0)),
                   project=data.get("project", "default"), tags=list(data.get("tags", [])),
                   note=data.get("note", ""), cost_usd=data.get("cost_usd"))


@dataclass
class Budget:
    """A spending ceiling, optionally scoped to a project and a period."""

    limit_usd: float
    period: str = "monthly"
    project: Optional[str] = None
    label: str = ""

    VALID_PERIODS = ("daily", "weekly", "monthly", "total")

    def __post_init__(self) -> None:
        if self.limit_usd <= 0:
            raise ValueError("Budget.limit_usd must be > 0")
        if self.period not in self.VALID_PERIODS:
            raise ValueError(
                f"Budget.period must be one of {self.VALID_PERIODS}, got {self.period!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Budget":
        return cls(limit_usd=float(data["limit_usd"]), period=data.get("period", "monthly"),
                   project=data.get("project"), label=data.get("label", ""))
