"""Tests for the iteration-4 configuration layer (pure logic)."""

from __future__ import annotations

import pytest

from token_usage.config import (
    Config,
    ConfigError,
    coerce,
    describe_defaults,
    options_from_args,
)
from token_usage.registry import PluginArgs


# -- coercion ---------------------------------------------------------------

def test_coerce_string_and_int():
    assert coerce("project", "demo") == "demo"
    assert coerce("limit", "25") == 25
    assert coerce("limit", 25) == 25


def test_coerce_bool_accepts_common_spellings():
    for text in ("1", "true", "YES", "y", "on"):
        assert coerce("no_bars", text) is True
    for text in ("0", "false", "No", "n", "off"):
        assert coerce("no_bars", text) is False


def test_coerce_bool_rejects_nonsense():
    with pytest.raises(ConfigError):
        coerce("no_bars", "maybe")


def test_coerce_rejects_unknown_key():
    with pytest.raises(ConfigError):
        coerce("projct", "demo")  # typo must not silently succeed


def test_coerce_rejects_bad_int():
    with pytest.raises(ConfigError):
        coerce("limit", "many")


def test_coerce_validates_enums():
    assert coerce("group", "day") == "day"
    with pytest.raises(ConfigError):
        coerce("group", "hour")
    with pytest.raises(ConfigError):
        coerce("period", "yearly")


# -- defaults ---------------------------------------------------------------

def test_set_and_unset_default():
    config = Config()
    assert config.set_default("project", "demo") == "demo"
    assert config.defaults == {"project": "demo"}
    assert config.unset_default("project") is True
    assert config.unset_default("project") is False


def test_clear_defaults_returns_count():
    config = Config()
    config.set_default("project", "a")
    config.set_default("limit", 5)
    assert config.clear_defaults() == 2
    assert config.defaults == {}


def test_merge_into_explicit_options_win():
    config = Config()
    config.set_default("project", "demo")
    config.set_default("limit", 10)
    merged = config.merge_into({"project": "other"})
    assert merged["project"] == "other"   # explicit wins
    assert merged["limit"] == 10          # default fills the gap


def test_merge_into_does_not_mutate_input():
    config = Config()
    config.set_default("project", "demo")
    options = {"limit": 3}
    config.merge_into(options)
    assert options == {"limit": 3}


# -- presets ----------------------------------------------------------------

def test_save_and_get_preset():
    config = Config()
    preset = config.save_preset("weekly", {"group": "day"}, description="weekly view")
    assert preset.name == "weekly"
    assert config.get_preset("weekly").options == {"group": "day"}
    assert config.preset_names() == ["weekly"]


def test_save_preset_rejects_empty_name():
    with pytest.raises(ConfigError):
        Config().save_preset("", {"group": "day"})


def test_delete_preset():
    config = Config()
    config.save_preset("x", {"group": "day"})
    assert config.delete_preset("x") is True
    assert config.delete_preset("x") is False


def test_preset_render():
    preset = Config().save_preset("p", {"group": "day", "top": 5})
    assert "--group day" in preset.render() and "--top 5" in preset.render()
    assert Config().save_preset("empty", {}).render() == "(no options)"


# -- serialisation ----------------------------------------------------------

def test_config_roundtrip():
    config = Config()
    config.set_default("project", "demo")
    config.save_preset("weekly", {"group": "day"}, description="d")
    restored = Config.from_dict(config.to_dict())
    assert restored.defaults == config.defaults
    assert restored.get_preset("weekly").description == "d"


def test_from_dict_skips_invalid_defaults():
    """A hand-edited config with one bad line must not brick the CLI."""
    restored = Config.from_dict({
        "defaults": {"project": "demo", "group": "hour", "bogus": 1},
        "presets": {"ok": {"options": {"limit": 5}}, "bad": "not-a-dict"},
    })
    assert restored.defaults == {"project": "demo"}
    assert restored.preset_names() == ["ok"]


def test_from_dict_handles_empty():
    restored = Config.from_dict({})
    assert restored.defaults == {} and restored.presets == {}


# -- helpers ----------------------------------------------------------------

def test_options_from_args_captures_known_keys_only():
    args = PluginArgs({"group": "day", "limit": "5", "apply": True, "name": "x"})
    captured = options_from_args(args)
    assert captured == {"group": "day", "limit": "5"}  # apply/name not captured


def test_options_from_args_skips_false_booleans():
    args = PluginArgs({"no_bars": False, "oldest": True})
    assert options_from_args(args) == {"oldest": True}


def test_describe_defaults_sorted():
    config = Config()
    config.set_default("project", "demo")
    config.set_default("limit", 5)
    assert describe_defaults(config) == [("limit", 5), ("project", "demo")]
