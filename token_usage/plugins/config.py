"""``config`` -- view and edit workspace defaults.

    token-usage config --action list
    token-usage config --action set --key project --value demo
    token-usage config --action unset --key project
    token-usage config --action clear

Defaults are merged into every command's options, so setting ``project`` once
means ``report``/``list``/``record`` all use it without repeating ``--project``.
"""

from __future__ import annotations

from token_usage.config import ConfigError, KNOWN_DEFAULTS, describe_defaults
from token_usage.registry import Plugin, PluginContext, register


class ConfigPlugin(Plugin):
    name = "config"
    help = "View or edit workspace defaults."
    aliases = ["settings"]
    usage = ("config [--action list|set|unset|clear|keys] "
             "[--key <name>] [--value <v>]")

    def run(self, ctx: PluginContext) -> int:
        action = ctx.args.get("action", "list")
        handler = {"list": self._list, "ls": self._list, "show": self._list,
                   "set": self._set, "unset": self._unset, "remove": self._unset,
                   "clear": self._clear, "keys": self._keys}.get(action)
        if handler is None:
            ctx.warn(f"unknown --action {action!r}")
            ctx.out(self.usage)
            return 2
        return handler(ctx)

    def _list(self, ctx: PluginContext) -> int:
        config = ctx.config
        pairs = describe_defaults(config)
        if not pairs:
            ctx.out("no defaults configured")
            ctx.out("hint: config --action set --key project --value demo")
            return 0
        ctx.out(f"{len(pairs)} default(s) in {ctx.store.config_path}")
        for key, value in pairs:
            ctx.out(f"  {key:<12} {value!r}")
        return 0

    def _keys(self, ctx: PluginContext) -> int:
        ctx.out("settable keys:")
        for key in sorted(KNOWN_DEFAULTS):
            ctx.out(f"  {key:<12} ({KNOWN_DEFAULTS[key].__name__})")
        return 0

    def _set(self, ctx: PluginContext) -> int:
        key = ctx.args.get("key")
        value = ctx.args.get("value")
        if not key or value is None:
            ctx.warn("--action set requires both --key and --value")
            return 2
        config = ctx.config
        try:
            coerced = config.set_default(key, value)
        except ConfigError as exc:
            ctx.warn(str(exc))
            return 2
        ctx.store.save_config(config)
        ctx.out(f"{key} = {coerced!r}")
        return 0

    def _unset(self, ctx: PluginContext) -> int:
        key = ctx.args.get("key")
        if not key:
            ctx.warn("--action unset requires --key")
            return 2
        config = ctx.config
        if not config.unset_default(key):
            ctx.warn(f"{key!r} was not set")
            return 1
        ctx.store.save_config(config)
        ctx.out(f"unset {key}")
        return 0

    def _clear(self, ctx: PluginContext) -> int:
        config = ctx.config
        count = config.clear_defaults()
        if not count:
            ctx.out("no defaults to clear")
            return 0
        ctx.store.save_config(config)
        ctx.out(f"cleared {count} default(s)")
        return 0


register(ConfigPlugin())
