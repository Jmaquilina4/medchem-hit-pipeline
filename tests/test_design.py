"""Generative design orchestration — the reward-hacking → recovery story at the SELECTION level.

A sum reward is gamed by an out-of-domain molecule that (falsely) maxes potency + selectivity;
the geometric-mean reward pushes it out of the top and keeps an in-domain molecule.
"""

from __future__ import annotations

import numpy as np

from medchem.features.featurize import compute_features
from medchem.generative.design import generate_and_select
from medchem.generative.samplers import MockSampler

_SPEC = [
    {"name": "qsar_pic50", "transform": "sigmoid", "center": 7.5},
    {"name": "selectivity_delta", "transform": "sigmoid", "center": 1.0},
    {"name": "applicability_domain", "transform": "reverse_sigmoid", "center": 0.4, "k": 10.0},
]


def test_sum_selects_the_ood_hack_but_geomean_recovers():
    train = ["c1ccccc1", "c1ccncc1", "CC(=O)Nc1ccccc1"]  # in-domain "training" chemistry
    tfp, *_ = compute_features(train)

    good = ["c1ccccc1", "c1ccncc1", "CC(=O)Nc1ccccc1"]    # modest, in-domain
    hack = "CCCCCCCCCCCC"                                 # long aliphatic → far out-of-domain
    sampler = MockSampler(good + [hack])

    # positional mock models: the hack is (falsely) predicted super-potent + selective — exactly
    # what an unconstrained generator games the QSAR into producing; the good ones are modest.
    pot = lambda X: np.array([5.0, 5.5, 6.0, 10.0])[: len(X)]  # noqa: E731
    sel = lambda X: np.array([0.5, 0.6, 0.7, 3.0])[: len(X)]   # noqa: E731

    res = generate_and_select(
        sampler, n=4, potency_predict=pot, selectivity_predict=sel,
        train_fp=tfp, spec=_SPEC, top_k=1,
    )

    # the sum reward is gamed — the out-of-domain hack ranks #1
    assert res["naive_top"][0]["smiles"] == hack
    # the geometric mean recovers — #1 is an in-domain molecule, not the hack
    assert res["constrained_top"][0]["smiles"] != hack
    assert res["constrained_top"][0]["smiles"] in good
    # the constrained pick is far more in-domain than the naive pick
    assert res["constrained_top_ad"][0] < res["naive_top_ad"][0]
