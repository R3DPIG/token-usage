"""Built-in plugin package.

Importing this package imports every built-in plugin module; each registers
itself with :data:`token_usage.registry.registry` on import. To add a built-in
command, write the module and append it to ``_BUILTIN_MODULES``. Imports are
wrapped defensively so one broken plugin cannot stop the CLI from starting.
"""

from __future__ import annotations

import importlib
from typing import List

_BUILTIN_MODULES = (
    "token_usage.plugins.record",
    "token_usage.plugins.list_records",
    "token_usage.plugins.report",
    "token_usage.plugins.export",
    "token_usage.plugins.budget",
    "token_usage.plugins.forecast",
    "token_usage.plugins.pricing",
    "token_usage.plugins.stats",
    "token_usage.plugins.sample",
    # Iteration 2: data ingestion and derived insights.
    "token_usage.plugins.import_usage",
    "token_usage.plugins.compare",
    "token_usage.plugins.anomaly",
    "token_usage.plugins.tags",
    "token_usage.plugins.doctor",
    # Iteration 3: data lifecycle governance.
    "token_usage.plugins.dedupe",
    "token_usage.plugins.prune",
    "token_usage.plugins.edit",
    "token_usage.plugins.merge",
    "token_usage.plugins.archive",
    # Iteration 4: configuration and presets.
    "token_usage.plugins.config",
    "token_usage.plugins.preset",
    # Iteration 5: declarative alert rules.
    "token_usage.plugins.alert",
)


def load_builtin_plugins() -> List[str]:
    """Import every built-in plugin module; return the ones that loaded."""
    loaded: List[str] = []
    for module_name in _BUILTIN_MODULES:
        try:
            importlib.import_module(module_name)
            loaded.append(module_name)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not stop the rest
            print(f"warning: failed to load built-in plugin {module_name!r}: {exc}")
    return loaded


__all__ = ["load_builtin_plugins"]
