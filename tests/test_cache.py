"""Content-addressed caching behavior of the DAG runner.

These tests are the load-bearing proof of the reproducibility thesis: a stage is
skipped when nothing it depends on changed, re-run when its config subtree or an
upstream result changes, and forced when asked.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import medchem.stages  # noqa: F401  (registers the demo pipeline)
from medchem.config import Config, DataConfig, DemoConfig
from medchem.pipeline import runner


def _run(cfg: Config, tmp, **kw):
    return runner.run(
        "demo",
        cfg,
        workdir=str(tmp / "work"),
        cache_dir=str(tmp / "cache"),
        **kw,
    )


def test_second_run_hits_cache(tmp_path):
    r1 = _run(Config(), tmp_path)
    assert all(not r.from_cache for r in r1)
    r2 = _run(Config(), tmp_path)
    assert all(r.from_cache for r in r2)


def test_config_change_invalidates(tmp_path):
    _run(Config(), tmp_path)
    r = _run(Config(demo=DemoConfig(n=9)), tmp_path)
    # both stages read the "demo" section, so both must re-run
    assert all(not x.from_cache for x in r)


def test_unrelated_config_change_keeps_cache(tmp_path):
    _run(Config(), tmp_path)
    # neither demo stage reads the "data" section -> cache subtree unchanged.
    # Uses a REAL DataConfig field: the config is now extra="forbid", so an invented key
    # would fail validation instead of quietly proving nothing.
    r = _run(Config(data=DataConfig(chembl_release="35")), tmp_path)
    assert all(x.from_cache for x in r)


def test_upstream_change_propagates(tmp_path):
    _run(Config(), tmp_path)
    # changing "seed" only appears in the seed stage's config_keys, but the
    # double stage depends on seed's cache_key, so it must re-run too
    r = _run(Config(seed=99), tmp_path)
    assert all(not x.from_cache for x in r)


def test_force_reruns(tmp_path):
    _run(Config(), tmp_path)
    r = _run(Config(), tmp_path, force=True)
    assert all(not x.from_cache for x in r)


def test_artifacts_are_key_scoped_across_config_switch(tmp_path):
    # Regression for the artifact-clobbering bug: run n=5, then n=9, then n=5
    # again. The n=5 re-run must be a cache hit AND its on-disk artifacts must
    # still hold the n=5 values (not the n=9 values from the intervening run).
    _run(Config(demo=DemoConfig(n=5)), tmp_path)
    _run(Config(demo=DemoConfig(n=9)), tmp_path)
    r = _run(Config(demo=DemoConfig(n=5)), tmp_path)

    assert all(x.from_cache for x in r)
    seed = next(x for x in r if x.name == "seed")
    double = next(x for x in r if x.name == "double")
    assert json.loads(Path(seed.outputs["values"]).read_text(encoding="utf-8")) == [0, 1, 2, 3, 4]
    assert json.loads(Path(double.outputs["values"]).read_text(encoding="utf-8")) == [0, 2, 4, 6, 8]


def test_only_runs_selected_and_skips_unrelated_downstream(tmp_path):
    # `only=["seed"]` runs seed and skips the unrelated downstream `double`
    # entirely (it must not require double to be cached).
    r = runner.run(
        "demo", Config(), workdir=str(tmp_path / "w"), cache_dir=str(tmp_path / "c"), only=["seed"]
    )
    assert [x.name for x in r] == ["seed"]


def test_only_downstream_requires_cached_upstream(tmp_path):
    # `only=["double"]` needs seed (an upstream) cached; fail fast if it isn't.
    with pytest.raises(RuntimeError):
        runner.run(
            "demo", Config(), workdir=str(tmp_path / "w"), cache_dir=str(tmp_path / "c"), only=["double"]
        )


def test_missing_config_section_raises(tmp_path, monkeypatch):
    """A stage declaring a config section the config does not have must fail BEFORE anything runs.

    Rewritten because the original was both stale and non-hermetic. It asserted that a default
    ``Config`` lacks the "features"/"model"/"eval" sections; those have since gained defaults, so
    validation passed, the real ``data_pull`` stage executed, and the test made a LIVE ChEMBL REQUEST
    inside the unit suite. It then passed only because some later stage raised ``KeyError`` for an
    unrelated reason — and failed outright whenever the API was down or served an error page, which is
    how it was found.

    So the contract is asserted directly: an absent declared section raises, and NO stage body runs.
    """
    from medchem.pipeline.stage import get_pipeline

    ran: list[str] = []
    stages = dict(get_pipeline("discovery"))
    first = next(iter(stages))
    # Declare a section that cannot exist, and record whether any stage body is entered.
    stages[first] = replace(stages[first], config_keys=("no_such_section",))
    for name, st in stages.items():
        stages[name] = replace(st, fn=_recording(st.fn, name, ran))
    # Patch where the RUNNER looks it up: it does `from ... import get_pipeline`, so the name is
    # bound in the runner module and patching the source module has no effect.
    monkeypatch.setattr("medchem.pipeline.runner.get_pipeline", lambda _p: stages)

    with pytest.raises(KeyError, match="no_such_section"):
        runner.run(
            "discovery",
            Config(),
            workdir=str(tmp_path / "work"),
            cache_dir=str(tmp_path / "cache"),
        )
    assert ran == [], f"validation must precede execution, but these ran: {ran}"


def _recording(fn, name: str, log: list[str]):
    """Wrap a stage body so the test can prove it was never entered."""
    def wrapper(*a, **k):
        log.append(name)
        return fn(*a, **k)
    wrapper.__module__ = getattr(fn, "__module__", "medchem")
    wrapper.__qualname__ = getattr(fn, "__qualname__", name)
    return wrapper


def test_no_cache_entry_points_outside_the_repository():
    """A cache entry naming a path outside the repo makes provenance unresolvable — or worse, wrong.

    The content cache is shared across runs and lives outside any ``--workdir``, so a run launched with
    ``--workdir /tmp/somewhere`` writes entries whose outputs point there. Those entries then satisfy
    later cache lookups: ``scripts/make_run_manifest_v2.py`` and ``scripts/publish_provenance.py`` both
    resolve published artifacts by recomputing a stage key and reading the entry, so they would resolve
    to a scratch directory that may since have been deleted — and if it has NOT been deleted, they would
    publish records describing artifacts nobody can see.

    Measured once: 30 of 204 entries pointed into scratch directories from earlier reruns. The manifest
    tool already refuses such a path with a clear message; this catches the condition before a tool has
    to.

    Skipped where no cache exists, which is every fresh clone and CI.
    """
    repo = Path(__file__).resolve().parent.parent
    cache = repo / ".medchem_cache"
    if not cache.is_dir():
        pytest.skip("no local content cache (expected in CI and in a fresh clone)")
    outside = []
    for f in sorted(cache.glob("*.json")):
        try:
            outs = (json.loads(f.read_text()) or {}).get("outputs") or {}
        except json.JSONDecodeError:
            continue
        for v in (outs.values() if isinstance(outs, dict) else outs):
            p = Path(str(v))
            if p.is_absolute() and not p.is_relative_to(repo):
                outside.append(f"{f.name[:12]}… -> {p}")
                break
    assert not outside, (
        f"{len(outside)} cache entr(ies) point outside the repository; a stage key would resolve to "
        f"one of them and provenance would name an artifact that is not published:\n  "
        + "\n  ".join(outside[:10])
    )
