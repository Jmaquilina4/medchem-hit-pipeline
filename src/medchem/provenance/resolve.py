"""Resolve which artifacts a given config's run actually produced, by recomputing stage cache keys.

Why this exists
---------------
Provenance tooling repeatedly had to answer "which of these identical-looking files did the run
consume?", and every heuristic answer was wrong in a different way:

* Globbing ``*_raw.csv`` and keying by filename took whichever directory sorted last, which pointed at
  a stale pull output lacking assay identity.
* Selecting an ``eval_report.json`` by matching the manifest's gate value worked only while gate values
  were unique. Once a cohort correction re-ran every panel, the three *unchanged* panels reproduced
  their scaffold-CV R² **exactly**, so two era-split reports matched and the rule could not choose. It
  refused rather than guessing, which was correct and also a dead end.

The exact answer is available without heuristics: a stage's output directory IS its cache key, and that
key is a pure function of the code version, the stage's source closure, its config subtree, and its
upstream keys. Replaying the runner's own key computation over the stage graph therefore reproduces the
exact path for any (config, stage) pair.

This module is imported only by scripts, never by ``medchem.stages`` or ``medchem.cli``, so it stays out
of the scientific-source digest's closure — resolving provenance must not perturb the identity of the
thing being resolved. It is also the outermost architectural layer; see ``medchem/provenance/__init__``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["resolve_stage_keys", "resolve_stage_outputs"]


def resolve_stage_keys(config: Any, *, pipeline: str = "discovery") -> dict[str, str]:
    """Map stage name -> cache key, for every stage in the graph, exactly as the runner would.

    Mirrors ``medchem.pipeline.runner.run``'s key computation, including the subtlety that OPTIONAL
    upstream dependencies contribute when present — a stage's key ignoring an optional upstream was a
    real defect once, and a resolver that disagreed with the runner would silently return wrong paths.
    """
    import medchem.stages  # noqa: F401  (registration side-effect)
    from medchem import __version__
    from medchem.pipeline.cache import hash_source, stage_cache_key
    from medchem.pipeline.runner import _config_subtree, _topo_sort
    from medchem.pipeline.stage import get_pipeline

    stages = get_pipeline(pipeline)
    keys: dict[str, str] = {}
    for name in _topo_sort(stages):
        st = stages[name]
        upstream = [d for d in (*st.deps, *st.optional_deps) if d in keys]
        keys[name] = stage_cache_key(
            code_version=__version__,
            source_hash=hash_source(st.fn),
            config_subtree=_config_subtree(config, st.config_keys),
            upstream_hashes={d: keys[d] for d in upstream},
        )
    return keys


def resolve_stage_outputs(
    repo: Path, config_path: Path, stage: str, *, pipeline: str = "discovery"
) -> dict[str, Path]:
    """The output paths a stage produced for this config, read from its cache manifest.

    Raises rather than returning a best guess: a provenance record built from the wrong file is worse
    than no record, because it looks authoritative.
    """
    from medchem.config import load_config

    keys = resolve_stage_keys(load_config(config_path), pipeline=pipeline)
    if stage not in keys:
        raise SystemExit(f"unknown stage {stage!r}; known: {sorted(keys)}")
    key = keys[stage]
    manifest = repo / ".medchem_cache" / f"{key}.json"
    if not manifest.exists():
        raise SystemExit(
            f"no cache manifest for stage {stage!r} key {key[:12]}… under {config_path.name}. The stage "
            f"has not run at this code and config, so there is nothing to record."
        )
    import json

    outs = json.loads(manifest.read_text())["outputs"]
    items = outs.items() if isinstance(outs, dict) else enumerate(outs)
    resolved: dict[str, Path] = {}
    for label, value in items:
        p = Path(str(value))
        resolved[str(label)] = p if p.is_absolute() else repo / p
    return resolved
