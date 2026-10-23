"""Model pricing catalog and cost computation.

The catalog is the source of truth for "what does a token cost". It ships with a
handful of well-known public prices and can be extended at runtime from JSON or
by a plugin. Prices are illustrative defaults for a demo tool -- confirm against
your provider's pricing page before billing on them.

All rates are USD per one million tokens. Cached input is billed at its own
(usually lower) rate; the non-cached remainder uses the standard input rate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from token_usage.models import ModelPricing, UsageRecord
from token_usage.utils import round_usd


class PricingCatalog:
    """In-memory map of model name -> :class:`ModelPricing`.

    Unknown models fall back to a configured default rather than raising, since
    dropping a record silently is worse than pricing it approximately.
    """

    def __init__(self, entries: Optional[Iterable[ModelPricing]] = None,
                 default_model: Optional[str] = None) -> None:
        self._entries: Dict[str, ModelPricing] = {}
        self.default_model = default_model
        for entry in entries or []:
            self.add(entry)

    def add(self, pricing: ModelPricing) -> None:
        self._entries[pricing.model] = pricing

    def remove(self, model: str) -> bool:
        return self._entries.pop(model, None) is not None

    def get(self, model: str) -> Optional[ModelPricing]:
        if model in self._entries:
            return self._entries[model]
        if self.default_model and self.default_model in self._entries:
            return self._entries[self.default_model]
        return None

    def require(self, model: str) -> ModelPricing:
        pricing = self.get(model)
        if pricing is None:
            raise KeyError(f"no pricing for model {model!r} and no default configured")
        return pricing

    def models(self) -> List[str]:
        return sorted(self._entries)

    def all(self) -> List[ModelPricing]:
        return [self._entries[m] for m in self.models()]

    def cost_for(self, model: str, input_tokens: int, output_tokens: int,
                 cached_input_tokens: int = 0) -> float:
        """Cost in USD for one call, splitting cached from standard input."""
        pricing = self.require(model)
        non_cached_input = max(0, input_tokens - cached_input_tokens)
        cost = (
            non_cached_input / 1_000_000.0 * pricing.input_per_1m
            + cached_input_tokens / 1_000_000.0 * pricing.cached_input_per_1m
            + output_tokens / 1_000_000.0 * pricing.output_per_1m
        )
        return round_usd(cost)

    def cost_for_record(self, record: UsageRecord) -> float:
        """Cost for a record, honouring a pre-computed override if present."""
        if record.cost_usd is not None:
            return record.cost_usd
        return self.cost_for(record.model, record.input_tokens,
                             record.output_tokens, record.cached_input_tokens)

    def to_dict(self) -> Dict:
        return {"default_model": self.default_model,
                "entries": [e.to_dict() for e in self.all()]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict) -> "PricingCatalog":
        entries = [ModelPricing.from_dict(e) for e in data.get("entries", [])]
        return cls(entries, default_model=data.get("default_model"))

    @classmethod
    def from_json(cls, text: str) -> "PricingCatalog":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Path) -> "PricingCatalog":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


# model, input/1M, output/1M, cached/1M
_DEFAULT_ENTRIES = [
    ("gpt-4o",                 2.50,  10.00, 1.25),
    ("gpt-4o-mini",            0.15,   0.60, 0.075),
    ("gpt-4-turbo",           10.00,  30.00, 5.00),
    ("gpt-3.5-turbo",          0.50,   1.50, 0.25),
    ("claude-3-5-sonnet",      3.00,  15.00, 0.30),
    ("claude-3-5-haiku",       0.80,   4.00, 0.08),
    ("claude-3-opus",         15.00,  75.00, 1.50),
    ("text-embedding-3-small", 0.02,   0.00, 0.00),
    ("text-embedding-3-large", 0.13,   0.00, 0.00),
]


def default_catalog() -> PricingCatalog:
    """Build the shipped catalog with a sensible default fallback model."""
    entries = [ModelPricing(model=m, input_per_1m=i, output_per_1m=o,
                            cached_input_per_1m=c)
               for (m, i, o, c) in _DEFAULT_ENTRIES]
    return PricingCatalog(entries, default_model="gpt-4o-mini")
