"""Stage contract and registry.

A *stage* is a function ``fn(ctx) -> StageResult`` registered under a named
pipeline with an explicit list of upstream dependencies and the config sections
it reads. The runner topologically orders stages and content-addresses their
results so unchanged stages are skipped on re-run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

StageFn = Callable[["StageContext"], "StageResult"]


@dataclass(slots=True)
class StageResult:
    """What a stage returns; also what the cache persists."""

    name: str
    outputs: dict[str, str] = field(default_factory=dict)  # artifact name -> path
    metrics: dict[str, Any] = field(default_factory=dict)
    gate_status: str | None = None  # "pass" | "fail" | None
    from_cache: bool = False
    cache_key: str | None = None


@dataclass(slots=True)
class StageContext:
    """What a stage receives at run time."""

    config: Any  # a medchem.config.Config (typed Any to avoid an import cycle)
    workdir: str
    upstream: dict[str, StageResult] = field(default_factory=dict)


@dataclass(slots=True)
class Stage:
    """Registry entry describing a single stage."""

    name: str
    pipeline: str
    fn: StageFn
    deps: tuple[str, ...] = ()
    # Dependencies that are USED IF PRESENT but do not block disabling. A stage declaring an optional
    # dep must read it with ctx.upstream.get(name), never ctx.upstream[name], and must produce a
    # meaningful result without it. This is what makes `disable_stages` an honest capability rather
    # than a documented one: previously the screening and generative stages hard-required selectivity,
    # so disabling it raised, while the README claimed the compose worked.
    optional_deps: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()  # top-level config sections this stage reads
    description: str = ""

    # NOTE: an `all_deps` convenience property was removed here. It had zero callers -- the runner
    # spreads `(*st.deps, *st.optional_deps)` at each of the three sites that need it. An unused
    # helper on a core dataclass reads as API and invites divergence from the real logic.


_REGISTRY: dict[str, dict[str, Stage]] = {}


def stage(
    pipeline: str,
    name: str,
    *,
    deps: tuple[str, ...] = (),
    optional_deps: tuple[str, ...] = (),
    config_keys: tuple[str, ...] = (),
    description: str = "",
) -> Callable[[StageFn], StageFn]:
    """Decorator registering ``fn`` as a stage of ``pipeline``."""

    def deco(fn: StageFn) -> StageFn:
        bucket = _REGISTRY.setdefault(pipeline, {})
        if name in bucket:
            raise ValueError(f"stage {name!r} already registered for pipeline {pipeline!r}")
        bucket[name] = Stage(
            name=name,
            pipeline=pipeline,
            fn=fn,
            deps=tuple(deps),
            optional_deps=tuple(optional_deps),
            config_keys=tuple(config_keys),
            description=description,
        )
        return fn

    return deco


# Back-compatible aliases. The flagship case study was previously invoked as `-p jak1`; that keeps
# working and resolves to the generic graph, so documented commands do not silently break.
PIPELINE_ALIASES = {"jak1": "discovery"}


def resolve_pipeline(pipeline: str) -> str:
    """Map a caller-supplied pipeline name through the alias table."""
    return PIPELINE_ALIASES.get(pipeline, pipeline)


def get_pipeline(pipeline: str) -> dict[str, Stage]:
    """Return a copy of the stage map for ``pipeline`` (aliases resolved)."""
    pipeline = resolve_pipeline(pipeline)
    if pipeline not in _REGISTRY:
        raise KeyError(f"unknown pipeline {pipeline!r}; registered: {sorted(_REGISTRY)}")
    return dict(_REGISTRY[pipeline])


def list_pipelines() -> list[str]:
    """Return the names of all registered pipelines."""
    return sorted(_REGISTRY)
