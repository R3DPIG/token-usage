"""Plugin contract and registry -- the seam that makes the tool iterable.

The core knows nothing about commands; it discovers and invokes :class:`Plugin`
instances. A new feature is a new module defining a :class:`Plugin` subclass --
no core edits. Discovery: built-in modules imported by
:mod:`token_usage.plugins` (self-registering on import), plus external modules
named in ``TOKEN_USAGE_PLUGIN_MODULES`` or ``.py`` files in ``<home>/plugins/``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from token_usage.pricing import PricingCatalog
from token_usage.storage import JsonStore


class PluginContext:
    """Everything a plugin needs, passed in explicitly (the sole channel between
    core and plugin, which keeps plugins testable)."""

    def __init__(self, store: JsonStore, catalog: PricingCatalog,
                 args: "PluginArgs") -> None:
        self.store = store
        self.catalog = catalog
        self.args = args

    def records(self):
        """All stored usage records, loaded on demand."""
        return self.store.load_records()

    def out(self, message: str = "") -> None:
        print(message)

    def warn(self, message: str) -> None:
        print(f"warning: {message}")


class PluginArgs:
    """A tolerant bag of ``--key value`` options with typed accessors; the CLI
    parses generically so the core never learns any command's flags."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None,
                 positionals: Optional[List[str]] = None) -> None:
        self.raw = dict(raw or {})
        self.positionals = list(positionals or [])

    def get(self, key: str, default=None):
        value = self.raw.get(key)
        return default if value is None else value

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key)
        if value is None or isinstance(value, bool):
            return default if value is None else value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def get_list(self, key: str, default=None) -> List[str]:
        value = self.get(key)
        if value is None:
            return list(default or [])
        if isinstance(value, list):
            return [str(v) for v in value]
        return [part.strip() for part in str(value).split(",") if part.strip()]


class Plugin:
    """Base class for every command: set ``name``/``help``, implement ``run``.

    ``aliases`` and ``usage`` are optional.
    """

    name: str = ""
    help: str = ""
    aliases: List[str] = []
    usage: str = ""

    def run(self, ctx: PluginContext) -> int:  # pragma: no cover - abstract
        raise NotImplementedError(f"plugin {type(self).__name__} must implement run()")


class PluginRegistry:
    """Ordered collection of plugins with name/alias lookup."""

    def __init__(self) -> None:
        self._plugins: Dict[str, Plugin] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, plugin: Plugin) -> None:
        if not plugin.name:
            raise ValueError("plugin must define a non-empty name")
        if plugin.name in self._plugins:
            raise ValueError(f"duplicate plugin name: {plugin.name!r}")
        self._plugins[plugin.name] = plugin
        for alias in plugin.aliases:
            if alias in self._aliases:
                raise ValueError(f"duplicate plugin alias: {alias!r}")
            self._aliases[alias] = plugin.name

    def get(self, name: str) -> Optional[Plugin]:
        if name in self._plugins:
            return self._plugins[name]
        resolved = self._aliases.get(name)
        return self._plugins[resolved] if resolved else None

    def names(self) -> List[str]:
        return list(self._plugins)

    def all(self) -> List[Plugin]:
        return [self._plugins[n] for n in self.names()]

    def __contains__(self, name: str) -> bool:
        return name in self._plugins or name in self._aliases

    def __len__(self) -> int:
        return len(self._plugins)

    def render_help(self) -> str:
        lines = ["Available commands:"]
        width = max((len(p.name) for p in self.all()), default=0)
        for plugin in sorted(self.all(), key=lambda p: p.name):
            lines.append(f"  {plugin.name:<{width}}  {plugin.help}")
            if plugin.aliases:
                lines.append(f"  {' ' * width}  aliases: {', '.join(plugin.aliases)}")
        return "\n".join(lines)


# Process-wide registry; built-in plugins register here on import.
registry = PluginRegistry()


def register(plugin: Plugin) -> Plugin:
    """Register a plugin instance with the process-wide registry."""
    registry.register(plugin)
    return plugin


def discover_external(store: JsonStore) -> List[str]:
    """Import external plugins from ``TOKEN_USAGE_PLUGIN_MODULES`` and
    ``<home>/plugins/*.py``; return what loaded. Errors are reported, never fatal.
    """
    loaded: List[str] = []

    env = os.environ.get("TOKEN_USAGE_PLUGIN_MODULES", "")
    for module_name in [m.strip() for m in env.split(",") if m.strip()]:
        try:
            importlib.import_module(module_name)
            loaded.append(module_name)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash
            print(f"warning: failed to import plugin module {module_name!r}: {exc}")

    plugins_dir = store.home / "plugins"
    for path in sorted(plugins_dir.glob("*.py")) if plugins_dir.is_dir() else []:
        if path.name.startswith("_"):
            continue
        try:
            _load_from_path(path)
            loaded.append(str(path))
        except Exception as exc:  # noqa: BLE001 - surface, don't crash
            print(f"warning: failed to load plugin file {path}: {exc}")

    return loaded


def _load_from_path(path: Path) -> None:
    module_name = f"token_usage_external_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
