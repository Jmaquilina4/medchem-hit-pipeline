"""Multi-objective scoring for constrained generative design (ADR 0003).

The reward-hacking → recovery centerpiece, and it's provable **without a GPU**: a naive
weighted *sum* (mean) of component scores is gameable — a molecule that maxes one objective
and fails the rest still scores high — whereas a weighted *geometric mean* forces every
objective to be satisfied at once (any near-zero component tanks the whole product).

This is the reward the generative stage (GPU, REINVENT4) optimizes; keeping it here — pure,
deterministic, CPU-only — means the reward logic is unit-tested independent of the sampler.
Components (each transformed to [0,1]) come from the pipeline's own models: QSAR pIC50,
direct-Δ selectivity, RDKit physchem (MW/SlogP/TPSA/QED), and an applicability-domain term.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

# Transforms live in medchem.transforms (a leaf module) because medchem.config also needs
# them for validation, and config must not import from this layer. Re-exported here so
# existing callers and tests keep working.
from medchem.transforms import (  # noqa: F401
    _TRANSFORMS,
    apply_transform,
    double_sigmoid,
    reverse_sigmoid,
    sigmoid,
)


def aggregate(scores: Sequence[float], weights: Sequence[float] | None = None,
              method: str = "geometric_mean") -> float:
    """Combine per-component [0,1] scores into a scalar in [0,1].

    ``sum`` = weighted mean (a normalized sum; rank-equivalent to v1's raw sum, and
    gameable). ``geometric_mean`` = weighted product (forces ALL objectives).
    """
    s = np.asarray(list(scores), dtype=float)
    w = np.ones_like(s) if weights is None else np.asarray(list(weights), dtype=float)
    if s.size == 0 or w.sum() == 0:
        raise ValueError("no scores, or weights sum to zero")
    if method == "sum":
        return float(np.sum(w * s) / np.sum(w))
    if method in ("geometric_mean", "product"):
        s = np.clip(s, 1e-9, 1.0)  # a true zero would annihilate the product; floor it
        return float(np.exp(np.sum(w * np.log(s)) / np.sum(w)))
    raise ValueError(f"unknown aggregation {method!r}")


def score_components(
    components: Mapping[str, float],
    spec: Iterable[Mapping],
    *,
    aggregation: str = "geometric_mean",
) -> tuple[float, dict[str, float]]:
    """Score one molecule from raw component values + a spec (config ``generative.scoring``).

    ``spec`` items are ``{name, transform, weight?, ...transform params}``. A missing
    component scores 0 (so an un-computable objective can't be silently ignored). Returns
    ``(aggregate_score, per_component_transformed)``.
    """
    names: list[str] = []
    weights: list[float] = []
    transformed: dict[str, float] = {}
    for comp in spec:
        name = comp["name"]
        params = {k: v for k, v in comp.items() if k not in ("name", "transform", "weight")}
        raw = components.get(name)
        transformed[name] = 0.0 if raw is None else apply_transform(comp["transform"], raw, **params)
        names.append(name)
        weights.append(float(comp.get("weight", 1.0)))
    agg = aggregate([transformed[n] for n in names], weights, method=aggregation)
    return agg, transformed
