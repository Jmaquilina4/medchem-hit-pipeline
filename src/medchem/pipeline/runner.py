"""The DAG runner: topological ordering + content-addressed caching.

Each stage's outputs are written to a **key-scoped** directory (``workdir/<key>/``)
so a cache hit always points at the artifacts that key produced — never a file
clobbered by a later run under a different config. Manifests record each output's
hash and are verified on load, so a missing or altered artifact forces a re-run.
Because a stage's key includes its upstreams' keys, a change anywhere propagates
downstream automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medchem import __version__
from medchem.pipeline.cache import hash_file, hash_source, stage_cache_key
from medchem.pipeline.stage import Stage, StageContext, StageResult, get_pipeline


def _topo_sort(stages: dict[str, Stage]) -> list[str]:
    order: list[str] = []
    temp: set[str] = set()
    perm: set[str] = set()

    def visit(name: str) -> None:
        if name in perm:
            return
        if name in temp:
            raise ValueError(f"dependency cycle detected at stage {name!r}")
        temp.add(name)
        for dep in stages[name].deps:
            if dep not in stages:
                raise KeyError(f"stage {name!r} depends on unknown stage {dep!r}")
            visit(dep)
        # Optional deps affect ORDER when present and are simply absent otherwise.
        for dep in stages[name].optional_deps:
            if dep in stages:
                visit(dep)
        temp.discard(name)
        perm.add(name)
        order.append(name)

    for name in stages:
        visit(name)
    return order


def _filtered_stages(pipeline: str, disable: list[str] | None) -> dict[str, Stage]:
    """Pipeline stages minus any disabled ones (ADR 0006). Errors if a *kept* stage
    depends on a disabled one, rather than breaking the DAG silently."""
    stages = get_pipeline(pipeline)
    if not disable:
        return stages
    drop = set(disable)
    # A name that is not in this pipeline disabled nothing and said so nowhere. `disable_stages:
    # [selectivty]` ran the full graph and reported success -- the exact shape of failure this
    # project keeps finding, where a config expresses an intent the code cannot honour and no one
    # is told. The config validator checks the list's shape; only here is the graph known.
    if unknown := sorted(drop - set(stages)):
        raise ValueError(
            f"cannot disable {unknown}: no such stage in pipeline {pipeline!r}. Its stages are "
            f"{sorted(stages)}. A name that matches nothing disables nothing, so it is rejected "
            f"rather than ignored."
        )
    kept = {k: v for k, v in stages.items() if k not in drop}
    for name, st in kept.items():
        # Only REQUIRED deps block a disable. Optional ones are the whole point: a stage that declares
        # one must run without it.
        bad = sorted({d for d in st.deps if d in drop})
        if bad:
            raise ValueError(
                f"cannot disable {bad}: stage {name!r} requires it. If {name!r} should tolerate its "
                f"absence, move it to optional_deps and handle ctx.upstream.get(...) returning None."
            )
    return kept


def plan(pipeline: str, disable: list[str] | None = None) -> list[Stage]:
    """Return a pipeline's (optionally filtered) stages in topological order."""
    stages = _filtered_stages(pipeline, disable)
    return [stages[name] for name in _topo_sort(stages)]


def _ancestors(stages: dict[str, Stage], names: set[str]) -> set[str]:
    """All transitive upstream dependencies of ``names``."""
    seen: set[str] = set()
    stack = list(names)
    while stack:
        st = stages[stack.pop()]
        for dep in (*st.deps, *st.optional_deps):
            if dep in stages and dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


def _config_dump(config: Any) -> dict[str, Any]:
    return config.model_dump() if hasattr(config, "model_dump") else dict(config)


def _config_subtree(config: Any, keys: tuple[str, ...]) -> Any:
    dump = _config_dump(config)
    if not keys:
        return dump
    return {k: dump.get(k) for k in keys}


def _write_manifest(path: Path, res: StageResult, output_hashes: dict[str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                "name": res.name,
                "outputs": res.outputs,
                "output_hashes": output_hashes,
                "metrics": res.metrics,
                "gate_status": res.gate_status,
                "cache_key": res.cache_key,
            },
            default=str,
        ),
        encoding="utf-8",
    )


def _load_manifest(path: Path) -> tuple[StageResult, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    res = StageResult(
        name=data["name"],
        outputs=data["outputs"],
        metrics=data["metrics"],
        gate_status=data.get("gate_status"),
        from_cache=True,
        cache_key=data["cache_key"],
    )
    return res, data.get("output_hashes", {})


def _artifacts_valid(outputs: dict[str, str], output_hashes: dict[str, str]) -> bool:
    """A cached result is usable only if every artifact still exists and matches its hash."""
    for name, path in outputs.items():
        p = Path(path)
        if not p.exists():
            return False
        if name in output_hashes and hash_file(p) != output_hashes[name]:
            return False
    return True


def run(
    pipeline: str,
    config: Any,
    *,
    workdir: str,
    cache_dir: str = ".medchem_cache",
    only: list[str] | None = None,
    from_stage: str | None = None,
    force: bool = False,
    disable: list[str] | None = None,
) -> list[StageResult]:
    """Execute ``pipeline`` with caching; return results in topological order."""
    stages = _filtered_stages(pipeline, disable)
    order = _topo_sort(stages)

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    if from_stage is not None and from_stage not in order:
        raise KeyError(f"--from stage {from_stage!r} not in pipeline {pipeline!r}")
    if only:
        unknown = set(only) - set(order)
        if unknown:
            raise KeyError(f"--stage names not in pipeline {pipeline!r}: {sorted(unknown)}")

    if only:
        selected = set(only)
    elif from_stage is not None:
        selected = set(order[order.index(from_stage):])
    else:
        selected = set(order)

    # We touch only the selected stages and their transitive upstreams; unrelated
    # downstream stages are skipped entirely (not required to be cached).
    needed = selected | _ancestors(stages, selected)

    # A stage must declare the config sections it reads; catch typos / missing
    # sections up front rather than silently hashing None and never invalidating.
    known = set(_config_dump(config))
    for name in selected:
        missing = [k for k in stages[name].config_keys if k not in known]
        if missing:
            raise KeyError(
                f"stage {name!r} declares config_keys absent from the config: {missing}"
            )

    results: dict[str, StageResult] = {}
    out: list[StageResult] = []
    for name in order:
        if name not in needed:
            continue
        st = stages[name]
        # Optional dependencies count when they are PRESENT. Excluding them here meant a stage's key
        # ignored its optional upstream entirely: a changed (or newly enabled) selectivity model would
        # not invalidate the screening or generative stages that consume it.
        upstream_names = [d for d in (*st.deps, *st.optional_deps) if d in results]
        upstream_hashes = {d: (results[d].cache_key or "") for d in upstream_names}
        key = stage_cache_key(
            code_version=__version__,
            source_hash=hash_source(st.fn),
            config_subtree=_config_subtree(config, st.config_keys),
            upstream_hashes=upstream_hashes,
        )
        manifest = cache / f"{key}.json"

        cached_res: StageResult | None = None
        cached_ok = False
        if manifest.exists():
            cached_res, output_hashes = _load_manifest(manifest)
            cached_ok = _artifacts_valid(cached_res.outputs, output_hashes)

        if name not in selected:
            if not cached_ok or cached_res is None:
                raise RuntimeError(
                    f"stage {name!r} is needed upstream but has no valid cached result; "
                    "run the full pipeline first"
                )
            res = cached_res
        elif cached_ok and not force and cached_res is not None:
            res = cached_res
        else:
            stage_out = work / key
            stage_out.mkdir(parents=True, exist_ok=True)
            ctx = StageContext(
                config=config,
                workdir=str(stage_out),
                # Optional deps MUST be here too. Passing only `st.deps` meant
                # `ctx.upstream.get("selectivity")` was always None, so every consumer silently took
                # the selectivity-absent path even when the stage had run and produced a model --
                # the optional-dependency feature was ordered correctly and never actually delivered.
                upstream={d: results[d] for d in upstream_names},
            )
            res = st.fn(ctx)
            res.name = name
            res.cache_key = key
            res.from_cache = False
            output_hashes = {n: hash_file(p) for n, p in res.outputs.items()}
            _write_manifest(manifest, res, output_hashes)

        results[name] = res
        out.append(res)
    return out
