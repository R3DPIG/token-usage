"""Storage, registry and CLI tests; each uses an isolated tmp workspace."""

from __future__ import annotations

import json

import pytest

from token_usage import cli
from token_usage.models import Budget, ModelPricing, UsageRecord
from token_usage.plugins import load_builtin_plugins
from token_usage.registry import Plugin, PluginRegistry, registry
from token_usage.storage import JsonStore


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path / "home")


def _record() -> UsageRecord:
    return UsageRecord(model="gpt-4o", input_tokens=1, output_tokens=1,
                       timestamp="2024-05-01T00:00:00")


def test_store_roundtrip_and_malformed(store):
    store.append_record(_record())
    store.append_record(_record())
    assert store.count_records() == 2
    store.usage_path.write_text(
        '{"model":"gpt-4o","input_tokens":1,"output_tokens":1,'
        '"timestamp":"2024-05-01T00:00:00"}\nnot json\n{"missing":"fields"}\n', encoding="utf-8")
    assert store.count_records() == 1


def test_store_rewrite_atomic(store):
    store.append_record(_record())
    store.rewrite_records([])
    assert store.count_records() == 0


def test_pricing_and_budget_persistence(store):
    catalog = store.load_pricing()
    catalog.add(ModelPricing("custom", 1.0, 2.0))
    store.save_pricing(catalog)
    assert "custom" in store.load_pricing().models()

    store.save_budgets([Budget(limit_usd=10.0, period="monthly", project="x")])
    assert store.load_budgets()[0].project == "x"


def test_registry_alias_and_duplicates():
    reg = PluginRegistry()

    class Dummy(Plugin):
        name = "dummy"
        help = "test"
        aliases = ["d"]

        def run(self, ctx):
            return 0

    reg.register(Dummy())
    assert reg.get("d") is not None and "d" in reg and len(reg) == 1
    with pytest.raises(ValueError):
        reg.register(Dummy())


def test_builtin_plugins_load():
    assert len(load_builtin_plugins()) >= 8
    assert all(registry.get(n) for n in
               ("record", "list", "report", "export", "budget", "forecast"))


# -- CLI parsing ------------------------------------------------------------

def test_parse_argv():
    cmd, opts, pos, _ = cli.parse_argv(["record", "--model", "gpt-4o", "--input", "100", "x"])
    assert (cmd, opts["model"], opts["input"], pos) == ("record", "gpt-4o", "100", ["x"])
    _, opts, _, _ = cli.parse_argv(["report", "--group=day", "--no-bars"])
    assert opts["group"] == "day" and opts["no_bars"] is True
    _, _, _, glob = cli.parse_argv(["--home", "/tmp/x", "list"])
    assert glob["home"] == "/tmp/x"


# -- CLI end to end ---------------------------------------------------------

def test_cli_record_then_report(store, capsys):
    home = str(store.home)
    assert cli.main(["--home", home, "record", "--model", "gpt-4o",
                     "--input", "1000000", "--output", "1000000"]) == 0
    capsys.readouterr()
    assert cli.main(["--home", home, "report", "--group", "model"]) == 0
    out = capsys.readouterr().out
    assert "gpt-4o" in out and "usage summary" in out


def test_cli_sample_stats_export(store, capsys):
    home = str(store.home)
    assert cli.main(["--home", home, "sample", "--days", "3", "--per-day", "2"]) == 0
    capsys.readouterr()
    assert cli.main(["--home", home, "stats"]) == 0
    assert "busiest day" in capsys.readouterr().out
    assert cli.main(["--home", home, "export", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and rows[0]["model"]


def test_cli_budget_exceeded_exit_code(store, capsys):
    home = str(store.home)
    cli.main(["--home", home, "record", "--model", "gpt-4o",
              "--input", "10000000", "--output", "0"])
    capsys.readouterr()
    code = cli.main(["--home", home, "budget", "--action", "set",
                     "--limit", "0.001", "--period", "total"])
    assert code == 1


def test_cli_help_and_unknown(capsys):
    assert cli.main(["--help"]) == 0
    assert "Available commands" in capsys.readouterr().out
    assert cli.main(["definitely-not-a-command"]) == 127


def test_external_plugin_discovery(store, capsys):
    plugins_dir = store.home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    (plugins_dir / "hello.py").write_text(
        "from token_usage.registry import Plugin, register\n"
        "class Hello(Plugin):\n"
        "    name = 'hello'\n    help = 'say hi'\n"
        "    def run(self, ctx):\n        ctx.out('hi from external plugin')\n        return 0\n"
        "register(Hello())\n", encoding="utf-8")
    assert cli.main(["--home", str(store.home), "hello"]) == 0
    assert "hi from external plugin" in capsys.readouterr().out
