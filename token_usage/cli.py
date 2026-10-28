"""Command-line entry point.

    token-usage [--home <path>] <command> [--key value ...] [positionals...]
    token-usage --help | --version

The CLI never learns what options a command takes: it parses ``--flag`` /
``--key value`` tokens into a :class:`PluginArgs` bag and hands them to the
plugin, so adding a plugin requires no change here. Globals (before the
command): ``--home <path>``, ``--help``/``-h``, ``--version``.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

from token_usage import __version__
from token_usage.plugins import load_builtin_plugins
from token_usage.registry import (
    PluginArgs,
    PluginContext,
    discover_external,
    registry,
)
from token_usage.storage import JsonStore, resolve_home

# Flags that never take a value; other flags consume the next token.
_BOOLEAN_FLAGS = {"no-bars", "oldest", "verbose", "quiet", "force", "dry-run"}


def parse_argv(argv: List[str]) -> Tuple[str, Dict[str, object], List[str], Dict[str, str]]:
    """Split argv into (command, options, positionals, globals).

    ``--key=value`` sets ``key``; ``--key value`` sets ``key`` to ``value``
    unless ``key`` is a known boolean flag or the next token is another flag.
    """
    globals_: Dict[str, str] = {}
    options: Dict[str, object] = {}
    positionals: List[str] = []

    index = 0
    while index < len(argv) and argv[index].startswith("-"):
        token = argv[index]
        if token in ("--help", "-h"):
            globals_["help"] = "1"
            index += 1
        elif token == "--version":
            globals_["version"] = "1"
            index += 1
        elif token == "--home" and index + 1 < len(argv):
            globals_["home"] = argv[index + 1]
            index += 2
        elif token.startswith("--home="):
            globals_["home"] = token.split("=", 1)[1]
            index += 1
        else:
            break

    if index >= len(argv):
        return "", {}, [], globals_

    command = argv[index]
    index += 1

    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            body = token[2:]
            if "=" in body:
                key, value = body.split("=", 1)
                options[key.replace("_", "-")] = value
                index += 1
                continue
            key = body.replace("_", "-")
            nxt = argv[index + 1] if index + 1 < len(argv) else None
            if key in _BOOLEAN_FLAGS or nxt is None or nxt.startswith("--"):
                options[key] = True
                index += 1
            else:
                options[key] = nxt
                index += 2
        else:
            positionals.append(token)
            index += 1

    # Normalise keys to snake_case so plugins can use ``get_int("per_day")``.
    return command, {k.replace("-", "_"): v for k, v in options.items()}, positionals, globals_


def apply_config_defaults(store: JsonStore, options: Dict[str, object]) -> Dict[str, object]:
    """Fill in configured defaults for options the user did not pass.

    Explicit CLI options always win. A corrupt or absent config yields the
    options unchanged, so this can never break a command.
    """
    try:
        config = store.load_config()
    except Exception:  # noqa: BLE001 - config must never block a command
        return options
    return config.merge_into(options)


def _print_help() -> int:
    print("token-usage -- track LLM token usage and cost\n")
    print("usage: token-usage [--home <path>] <command> [--key value ...]\n")
    print(registry.render_help())
    print("\nglobal options:\n"
          "  --home <path>   workspace root (default: .token-usage)\n"
          "  --preset <name> replay a saved option bundle\n"
          "  --version       print version\n"
          "  -h, --help      show this help")
    print("\nenvironment:\n"
          "  TOKEN_USAGE_HOME             override the workspace root\n"
          "  TOKEN_USAGE_PLUGIN_MODULES   extra importable plugin modules")
    print("\nplugins may also be dropped into <home>/plugins/*.py")
    return 0


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    load_builtin_plugins()
    command, options, positionals, globals_ = parse_argv(argv)

    if globals_.get("version"):
        print(f"token-usage {__version__}")
        return 0

    discover_external(JsonStore(resolve_home(globals_.get("home"))))

    if globals_.get("help") or not command:
        return _print_help()

    if command in ("help", "?"):
        target = positionals[0] if positionals else None
        if not target:
            return _print_help()
        plugin = registry.get(target)
        if plugin is None:
            print(f"unknown command: {target}")
            return 1
        print(f"{plugin.name} -- {plugin.help}\n")
        print(f"usage: token-usage {plugin.usage or plugin.name}")
        if plugin.aliases:
            print(f"aliases: {', '.join(plugin.aliases)}")
        return 0

    plugin = registry.get(command)
    if plugin is None:
        print(f"unknown command: {command!r}\n", file=sys.stderr)
        print(registry.render_help(), file=sys.stderr)
        return 127

    store = JsonStore(resolve_home(globals_.get("home", "")))
    options = _resolve_options(store, options)
    ctx = PluginContext(store, store.load_pricing(), PluginArgs(options, positionals))
    try:
        return plugin.run(ctx)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - a plugin bug must not leak a traceback
        print(f"error: command {command!r} failed: {exc}", file=sys.stderr)
        return 1


def _resolve_options(store: JsonStore, options: Dict[str, object]) -> Dict[str, object]:
    """Expand ``--preset <name>`` then fill in configured defaults.

    Precedence, lowest to highest: configured defaults < preset < explicit CLI
    options. A missing preset is reported but does not abort the command, since
    the user may simply have mistyped the name.
    """
    preset_name = options.pop("preset", None)
    if preset_name:
        preset = store.load_config().get_preset(str(preset_name))
        if preset is None:
            print(f"warning: preset {preset_name!r} not found; ignoring",
                  file=sys.stderr)
        else:
            merged = dict(preset.options)
            merged.update(options)
            options = merged
    return apply_config_defaults(store, options)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
