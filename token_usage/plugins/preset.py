"""``preset`` -- save and replay named option bundles.

    token-usage preset --action save --name weekly --group day --period weekly
    token-usage preset --action list
    token-usage report --preset weekly
    token-usage preset --action delete --name weekly

A preset captures the option flags you passed, so a long invocation becomes a
short one. Only recognised option keys are stored -- transient flags such as
``--apply`` are never captured.
"""

from __future__ import annotations

from token_usage.config import ConfigError, options_from_args
from token_usage.registry import Plugin, PluginContext, register


class PresetPlugin(Plugin):
    name = "preset"
    help = "Save, list, and delete named option bundles."
    aliases = ["presets"]
    usage = ("preset [--action save|list|show|delete] [--name <n>] "
             "[--description <text>] [<option flags to capture>]")

    def run(self, ctx: PluginContext) -> int:
        action = ctx.args.get("action", "list")
        handler = {"list": self._list, "ls": self._list, "save": self._save,
                   "add": self._save, "show": self._show, "delete": self._delete,
                   "rm": self._delete}.get(action)
        if handler is None:
            ctx.warn(f"unknown --action {action!r}")
            ctx.out(self.usage)
            return 2
        return handler(ctx)

    def _list(self, ctx: PluginContext) -> int:
        config = ctx.config
        names = config.preset_names()
        if not names:
            ctx.out("no presets saved")
            ctx.out("hint: preset --action save --name weekly --group day --period weekly")
            return 0
        ctx.out(f"{len(names)} preset(s):")
        for name in names:
            preset = config.get_preset(name)
            note = f"  # {preset.description}" if preset.description else ""
            ctx.out(f"  {name:<16} {preset.render()}{note}")
        return 0

    def _show(self, ctx: PluginContext) -> int:
        name = ctx.args.get("name")
        if not name:
            ctx.warn("--action show requires --name")
            return 2
        preset = ctx.config.get_preset(name)
        if preset is None:
            ctx.warn(f"preset {name!r} not found")
            return 1
        ctx.out(f"preset {name}")
        if preset.description:
            ctx.out(f"  description  {preset.description}")
        if not preset.options:
            ctx.out("  (no options)")
        for key, value in sorted(preset.options.items()):
            ctx.out(f"  {key:<12} {value!r}")
        ctx.out("")
        ctx.out(f"use it with: token-usage <command> --preset {name}")
        return 0

    def _save(self, ctx: PluginContext) -> int:
        name = ctx.args.get("name")
        if not name:
            ctx.warn("--action save requires --name")
            return 2
        captured = options_from_args(ctx.args)
        if not captured:
            ctx.warn("no recognised options to capture; pass option flags to save")
            ctx.out(self.usage)
            return 2
        config = ctx.config
        try:
            preset = config.save_preset(name, captured,
                                        description=ctx.args.get("description", ""))
        except ConfigError as exc:
            ctx.warn(str(exc))
            return 2
        ctx.store.save_config(config)
        ctx.out(f"saved preset {name!r}: {preset.render()}")
        return 0

    def _delete(self, ctx: PluginContext) -> int:
        name = ctx.args.get("name")
        if not name:
            ctx.warn("--action delete requires --name")
            return 2
        config = ctx.config
        if not config.delete_preset(name):
            ctx.warn(f"preset {name!r} not found")
            return 1
        ctx.store.save_config(config)
        ctx.out(f"deleted preset {name!r}")
        return 0


register(PresetPlugin())
