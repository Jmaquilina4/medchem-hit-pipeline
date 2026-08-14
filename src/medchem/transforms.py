"""Score-shaping transforms: pure functions mapping a raw property onto [0, 1].

These live at the bottom of the dependency graph on purpose. Two very different layers need them:
the generative scorer, which applies them, and :mod:`medchem.config`, which validates that a
configured component supplies the parameters its transform actually requires. Before this module
existed, ``config`` imported from ``generative.scoring`` — the innermost layer reaching into an outer
one, which an architecture check flagged as the package's only layering violation.

Nothing here imports from the rest of the package, and nothing here should.

The registry is the single source of truth for *which* transforms exist and *what each requires*.
Config validation introspects these signatures rather than duplicating a requirements list, so adding
a required parameter to a transform automatically invalidates every config that omits it.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x, center: float, k: float = 1.0):
    """0→1 rising through ``center``. Rewards larger x (potency, selectivity, QED).

    There is deliberately no default ``center``: a sigmoid without a centre is not a shape, and
    silently defaulting it would place the knee somewhere the author never chose. A shipped config
    once omitted it for three components and could not have run.
    """
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(x, dtype=float) - center)))


def reverse_sigmoid(x, center: float, k: float = 1.0):
    """1→0 falling through ``center``. Rewards smaller x (e.g. an applicability-domain distance)."""
    return 1.0 - sigmoid(x, center, k)


def double_sigmoid(x, low: float, high: float, k: float = 1.0):
    """~1 inside [low, high], →0 outside — penalises BOTH extremes (MW/SlogP/TPSA windows).

    Use this wherever the property has an optimum rather than a direction. Writing a one-sided
    ``sigmoid`` for such a property tells an optimiser that more is always better, and a reinforcement
    learner will duly overshoot — which is exactly what happened with fraction-sp3 carbon.
    """
    return sigmoid(x, low, k) * reverse_sigmoid(x, high, k)


_TRANSFORMS = {
    "sigmoid": sigmoid,
    "reverse_sigmoid": reverse_sigmoid,
    "double_sigmoid": double_sigmoid,
}


def apply_transform(name: str, value: float, **params) -> float:
    if name not in _TRANSFORMS:
        raise ValueError(f"unknown transform {name!r}; known: {sorted(_TRANSFORMS)}")
    return float(_TRANSFORMS[name](value, **params))
