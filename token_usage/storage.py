"""Persistence layer.

Usage records live in a JSONL file (one JSON object per line) -- append-friendly,
diff-friendly, streamable. Small singletons (pricing, budgets, config) live in
their own JSON documents. Layout under the workspace root (``--home`` or
``TOKEN_USAGE_HOME``)::

    <home>/
      usage.jsonl     # append-only usage records
      pricing.json    # pricing catalog override (optional)
      budgets.json    # list of budgets (optional)
      plugins/        # drop-in external plugins (*.py)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, List, Optional

from token_usage.models import Budget, UsageRecord
from token_usage.pricing import PricingCatalog, default_catalog

DEFAULT_HOME = ".token-usage"
USAGE_FILE = "usage.jsonl"
PRICING_FILE = "pricing.json"
BUDGETS_FILE = "budgets.json"


def resolve_home(explicit: Optional[str] = None) -> Path:
    """Resolve the workspace root: explicit arg > ``TOKEN_USAGE_HOME`` >
    ``.token-usage`` in the cwd."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("TOKEN_USAGE_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / DEFAULT_HOME


class JsonStore:
    """Filesystem-backed store rooted at a home directory."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)

    @property
    def usage_path(self) -> Path:
        return self.home / USAGE_FILE

    @property
    def pricing_path(self) -> Path:
        return self.home / PRICING_FILE

    @property
    def budgets_path(self) -> Path:
        return self.home / BUDGETS_FILE

    # -- usage records ----------------------------------------------------
    def append_record(self, record: UsageRecord) -> None:
        """Append one record as a ``\\n``-terminated JSON line."""
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self.usage_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def append_records(self, records: List[UsageRecord]) -> int:
        for record in records:
            self.append_record(record)
        return len(records)

    def iter_records(self) -> Iterator[UsageRecord]:
        """Yield stored records, skipping blank/malformed lines."""
        if not self.usage_path.exists():
            return
        with self.usage_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield UsageRecord.from_dict(json.loads(raw))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue

    def load_records(self) -> List[UsageRecord]:
        return list(self.iter_records())

    def rewrite_records(self, records: List[UsageRecord]) -> None:
        """Atomically replace the usage file (write to tmp, then rename)."""
        tmp = self.usage_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self.usage_path)

    def count_records(self) -> int:
        return sum(1 for _ in self.iter_records())

    # -- pricing ----------------------------------------------------------
    def load_pricing(self) -> PricingCatalog:
        """Load pricing, falling back to shipped defaults on missing/corrupt file."""
        if self.pricing_path.exists():
            try:
                return PricingCatalog.load(self.pricing_path)
            except (json.JSONDecodeError, KeyError, ValueError):
                return default_catalog()
        return default_catalog()

    def save_pricing(self, catalog: PricingCatalog) -> None:
        catalog.save(self.pricing_path)

    # -- budgets ----------------------------------------------------------
    def load_budgets(self) -> List[Budget]:
        if not self.budgets_path.exists():
            return []
        try:
            return [Budget.from_dict(item)
                    for item in json.loads(self.budgets_path.read_text(encoding="utf-8"))]
        except json.JSONDecodeError:
            return []

    def save_budgets(self, budgets: List[Budget]) -> None:
        self.budgets_path.write_text(
            json.dumps([b.to_dict() for b in budgets], indent=2, ensure_ascii=False),
            encoding="utf-8")
