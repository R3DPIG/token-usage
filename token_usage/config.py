"""Configuration and presets.

Two related ideas live here:

* **Defaults** -- a mapping of option name to value that the CLI merges into
  every command's arguments when the user did not pass that option explicitly.
  Set ``project`` once and ``report``/``list``/``record`` all pick it up.
* **Presets** -- named bundles of options for a specific command, so a long
  invocation can be replayed as ``preset --use weekly-review``.

Both are stored in ``<home>/config.json``. This module is pure: it validates and
transforms plain dicts; :class:`~token_usage.storage.JsonStore` does the I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Options that may be set as defaults, with a validator per key. Restricting the
# set keeps a typo like ``--set projct demo`` from silently doing nothing.
KNOWN_DEFAULTS = {
    "project": str,
    "model": str,
    "group": str,
    "period": str,
    "format": str,
    "limit": int,
    "top": int,
    "days": int,
    "tags": str,
    "no_bars": bool,
    "oldest": bool,
}

# Values that must come from a fixed set, checked after type coercion.
ENUM_VALUES = {
    "group": ("model", "project", "tag", "day", "week", "month"),
    "period": ("daily", "weekly", "monthly", "total"),
    "format": ("csv", "json"),
}

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


class ConfigError(ValueError):
    """Raised when a configuration value is unknown or malformed."""


def coerce(key: str, value: Any) -> Any:
    """Coerce ``value`` to the type declared for ``key``, validating enums.

    Accepts strings (as they arrive from the CLI) as well as native types (as
    they arrive from JSON), so the same function serves both entry points.
    """
    if key not in KNOWN_DEFAULTS:
        raise ConfigError(f"unknown setting {key!r}; known settings are "
                          f"{sorted(KNOWN_DEFAULTS)}")
    expected = KNOWN_DEFAULTS[key]
    if expected is bool:
        if isinstance(value, bool):
            result = value
        else:
            text = str(value).strip().lower()
            if text in _TRUE:
                result = True
            elif text in _FALSE:
                result = False
            else:
                raise ConfigError(f"{key} expects a boolean, got {value!r}")
    elif expected is int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{key} expects an integer, got {value!r}") from None
    else:
        result = str(value)

    allowed = ENUM_VALUES.get(key)
    if allowed and result not in allowed:
        raise ConfigError(f"{key} must be one of {list(allowed)}, got {result!r}")
    return result


@dataclass
class Preset:
    """A named bundle of options for replaying a command."""

    name: str
    options: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"options": dict(self.options), "description": self.description}

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "Preset":
        return cls(name=name, options=dict(data.get("options", {})),
                   description=data.get("description", ""))

    def render(self) -> str:
        pairs = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in sorted(self.options.items()))
        return pairs or "(no options)"


class Config:
    """Validated view over the stored configuration document."""

    def __init__(self, defaults: Optional[Dict[str, Any]] = None,
                 presets: Optional[Dict[str, Preset]] = None) -> None:
        self.defaults: Dict[str, Any] = dict(defaults or {})
        self.presets: Dict[str, Preset] = dict(presets or {})

    # -- defaults ---------------------------------------------------------
    def set_default(self, key: str, value: Any) -> Any:
        coerced = coerce(key, value)
        self.defaults[key] = coerced
        return coerced

    def unset_default(self, key: str) -> bool:
        return self.defaults.pop(key, None) is not None

    def clear_defaults(self) -> int:
        count = len(self.defaults)
        self.defaults.clear()
        return count

    def merge_into(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``options`` with defaults filled in for absent keys.

        Explicit options always win, so a one-off ``--project other`` still
        overrides the configured default.
        """
        merged = dict(self.defaults)
        merged.update(options)
        return merged

    # -- presets ----------------------------------------------------------
    def save_preset(self, name: str, options: Dict[str, Any],
                    description: str = "") -> Preset:
        if not name:
            raise ConfigError("preset name must not be empty")
        preset = Preset(name=name, options=dict(options), description=description)
        self.presets[name] = preset
        return preset

    def get_preset(self, name: str) -> Optional[Preset]:
        return self.presets.get(name)

    def delete_preset(self, name: str) -> bool:
        return self.presets.pop(name, None) is not None

    def preset_names(self) -> List[str]:
        return sorted(self.presets)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"defaults": dict(self.defaults),
                "presets": {name: preset.to_dict()
                            for name, preset in self.presets.items()}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a config, dropping any entry that fails validation.

        A hand-edited config file with one bad line should not brick the CLI, so
        invalid defaults are skipped rather than raising.
        """
        config = cls()
        for key, value in (data.get("defaults") or {}).items():
            try:
                config.set_default(key, value)
            except ConfigError:
                continue
        for name, payload in (data.get("presets") or {}).items():
            if isinstance(payload, dict):
                config.presets[name] = Preset.from_dict(name, payload)
        return config


def options_from_args(args) -> Dict[str, Any]:
    """Extract a preset-able option dict from a :class:`PluginArgs`.

    Only known default keys are kept, so a preset never captures transient flags
    like ``--apply`` or ``--dry-run``.
    """
    captured = {}
    for key in KNOWN_DEFAULTS:
        value = args.get(key)
        if value is None or isinstance(value, bool) and not value:
            continue
        captured[key] = value
    return captured


def describe_defaults(config: Config) -> List[Tuple[str, Any]]:
    """Sorted ``(key, value)`` pairs for display."""
    return sorted(config.defaults.items())
