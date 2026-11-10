"""CLI tests for the iteration-13 peers plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from token_usage import cli
from token_usage.models import UsageRecord
from token_usage.storage import JsonStore

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def ts(hours_ago: int = 1) -> str:
    return (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def build(home, *, model="gpt-4o", inp=1_000_000, out=0, calls=3):
    """Seed a workspace with ``calls`` identical records."""
    store = JsonStore(home)
    store.append_records([
        UsageRecord(model=model, input_tokens=inp, output_tokens=out,
                    timestamp=ts(i + 1), project="api", tags=["api"])
        for i in range(calls)])
    return store


@pytest.fixture
def fleet(tmp_path):
    """Three peers: cheap (no output), mid, and expensive (all output)."""
    stores = {"cheap": build(tmp_path / "cheap", out=0),
              "mid": build(tmp_path / "mid", out=500_000),
              "dear": build(tmp_path / "dear", out=1_000_000)}
    homes = ",".join(str(store.home) for store in stores.values())
    return stores["cheap"], homes, ",".join(stores)


def run(store, *args):
    return cli.main(["--home", str(store.home), "peers", *args])


# -- empty and error paths --------------------------------------------------

def test_cli_peers_without_sources(tmp_path, capsys):
    store = JsonStore(tmp_path / "solo")
    assert run(store) == 0
    out = capsys.readouterr().out
    assert "no peers in scope" in out and "--from" in out


def test_cli_peers_skips_a_missing_workspace(tmp_path, capsys):
    store = JsonStore(tmp_path / "solo")
    assert run(store, "--from", str(tmp_path / "nope")) == 0
    # ctx.warn goes to stdout, so the notice lands beside the normal output.
    assert "skipping" in capsys.readouterr().out


def test_cli_peers_include_self(tmp_path, capsys):
    store = build(tmp_path / "mine")
    other = build(tmp_path / "other", out=500_000)
    assert run(store, "--from", str(other.home), "--include-self",
               "--view", "table") == 0
    out = capsys.readouterr().out
    assert "self" in out and "other" in out


# -- views ------------------------------------------------------------------

def test_cli_peers_summary(fleet, capsys):
    cheap, homes, labels = fleet
    assert run(cheap, "--from", homes, "--label", labels) == 0
    out = capsys.readouterr().out
    assert "peer comparison" in out and "unit cost" in out
    assert "priciest peer pays" in out


def test_cli_peers_table_shows_normalised_columns(fleet, capsys):
    cheap, homes, labels = fleet
    assert run(cheap, "--from", homes, "--view", "table") == 0
    out = capsys.readouterr().out
    assert "$/1M" in out and "$/call" in out and "out%" in out and "cache" in out


def test_cli_peers_table_marks_ineligible_peers(tmp_path, capsys):
    big = build(tmp_path / "big")
    tiny = build(tmp_path / "tiny", inp=100, calls=1)
    assert run(big, "--from", f"{big.home},{tiny.home}",
               "--label", "big,tiny", "--view", "table") == 0
    out = capsys.readouterr().out
    assert "*tiny" in out and "below the token floor" in out


@pytest.mark.parametrize("metric,first,last", [
    ("cost_per_1m", "cheap", "dear"),      # lower is better
    ("output_ratio", "dear", "cheap"),     # higher is better
])
def test_cli_peers_rank_respects_metric_direction(fleet, capsys, metric,
                                                  first, last):
    cheap, homes, labels = fleet
    assert run(cheap, "--from", homes, "--label", labels,
               "--view", "rank", "--metric", metric) == 0
    out = capsys.readouterr().out
    assert out.index(f"#1   {first}") < out.index(f"#3   {last}")


def test_cli_peers_spread_lists_every_metric(fleet, capsys):
    cheap, homes, labels = fleet
    assert run(cheap, "--from", homes, "--label", labels, "--view", "spread") == 0
    out = capsys.readouterr().out
    for metric in ("cost_per_1m", "cost_per_call", "output_ratio", "cache_ratio"):
        assert metric in out
    assert "cv" in out


def test_cli_peers_spread_needs_two_eligible_peers(tmp_path, capsys):
    big = build(tmp_path / "big")
    tiny = build(tmp_path / "tiny", inp=100, calls=1)
    assert run(big, "--from", f"{big.home},{tiny.home}",
               "--label", "big,tiny", "--view", "spread") == 0
    assert "fewer than two eligible peers" in capsys.readouterr().out


def test_cli_peers_savings_states_its_premise(fleet, capsys):
    cheap, homes, labels = fleet
    assert run(cheap, "--from", homes, "--label", labels, "--view", "savings") == 0
    out = capsys.readouterr().out
    assert "repricing every peer" in out and "benchmark: cheap" in out
    assert "assumes" in out          # the premise is always stated


def test_cli_peers_savings_rejects_a_ratio_metric(fleet, capsys):
    cheap, homes, _ = fleet
    assert run(cheap, "--from", homes, "--view", "savings",
               "--metric", "output_ratio") == 2


def test_cli_peers_outliers_needs_three_peers(tmp_path, capsys):
    a = build(tmp_path / "a")
    b = build(tmp_path / "b", out=500_000)
    assert run(a, "--from", f"{a.home},{b.home}", "--view", "outliers") == 0
    assert "at least three" in capsys.readouterr().out


@pytest.mark.parametrize("extra_peer,expected", [
    (None, "no peer is an outlier"),          # three identical peers
    (("huge", "claude-3-opus"), "huge"),      # one peer far above the rest
])
def test_cli_peers_outliers(tmp_path, capsys, extra_peer, expected):
    peers_to_build = {"a": build(tmp_path / "a", out=0),
                      "b": build(tmp_path / "b", out=0),
                      "c": build(tmp_path / "c", out=0)}
    if extra_peer:
        name, model = extra_peer
        peers_to_build[name] = build(tmp_path / name, model=model, out=1_000_000)
    stores = list(peers_to_build.values())
    homes = ",".join(str(s.home) for s in stores)
    labels = ",".join(peers_to_build)
    assert run(stores[0], "--from", homes, "--label", labels,
               "--view", "outliers") == 0
    assert expected in capsys.readouterr().out


# -- comparability ----------------------------------------------------------

@pytest.fixture
def divergent(tmp_path):
    """A chat peer and an embedding peer: comparable totals, incomparable work."""
    chat = build(tmp_path / "chat", out=1_000_000)
    embed = build(tmp_path / "embed", model="text-embedding-3-small", out=0)
    return chat, f"{chat.home},{embed.home}"


def test_cli_peers_warns_about_divergent_workloads(divergent, capsys):
    chat, homes = divergent
    assert run(chat, "--from", homes, "--label", "chat,embed") == 0
    out = capsys.readouterr().out
    assert "not doing the same kind of work" in out
    assert "shown for completeness only" in out


def test_cli_peers_savings_repeats_the_comparability_caveat(divergent, capsys):
    """A 99.7% 'saving' must never be printed without its caveat."""
    chat, homes = divergent
    assert run(chat, "--from", homes, "--label", "chat,embed",
               "--view", "savings") == 0
    assert "arithmetic ceiling, not an achievable target" in capsys.readouterr().out


# -- validation, filtering, regression --------------------------------------

@pytest.mark.parametrize("args", [
    ("--view", "nonsense"),
    ("--metric", "nonsense"),
])
def test_cli_peers_rejects_bad_args(fleet, capsys, args):
    cheap, homes, labels = fleet
    assert run(cheap, "--from", homes, *args) == 2


def test_cli_peers_min_tokens_override(tmp_path, capsys):
    big = build(tmp_path / "big")
    tiny = build(tmp_path / "tiny", inp=100, calls=1)
    assert run(big, "--from", f"{big.home},{tiny.home}", "--view", "table") == 0
    assert "*tiny" in capsys.readouterr().out
    # Lowering the floor admits the small peer as a benchmark.
    capsys.readouterr()
    assert run(big, "--from", f"{big.home},{tiny.home}", "--min-tokens", "1",
               "--view", "table") == 0
    assert "*tiny" not in capsys.readouterr().out


def test_cli_peers_reads_peers_without_writing(tmp_path, capsys):
    mine = build(tmp_path / "mine")
    other = build(tmp_path / "other", out=500_000)
    before = other.usage_path.read_text(encoding="utf-8")
    assert run(mine, "--from", str(other.home), "--apply") == 0
    assert other.usage_path.read_text(encoding="utf-8") == before


def test_existing_commands_still_work(tmp_path, capsys):
    store = build(tmp_path / "regress")
    for command in ("report", "stats", "quality", "simulate", "drivers",
                    "optimize", "plan", "doctor"):
        assert cli.main(["--home", str(store.home), command]) in (0, 1)


def test_new_plugin_registered():
    from token_usage.plugins import load_builtin_plugins
    from token_usage.registry import registry
    load_builtin_plugins()
    for name in ("peers", "bench", "benchmark"):
        assert registry.get(name) is not None
