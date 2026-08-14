"""Active-learning loop orchestration (mock Sampler + ComputationalScorer, CPU)."""

from __future__ import annotations

import numpy as np

from medchem.features.featurize import compute_features
from medchem.generative.active_learning import active_learning_loop
from medchem.generative.samplers import MockSampler
from medchem.generative.scorers import MockScorer

_SPEC = [
    {"name": "qsar_pic50", "transform": "sigmoid", "center": 7.5},
    {"name": "selectivity_delta", "transform": "sigmoid", "center": 1.0},
    {"name": "applicability_domain", "transform": "reverse_sigmoid", "center": 0.4, "k": 10.0},
]


def test_loop_runs_rounds_and_validates_batches_with_scorer():
    train = ["c1ccccc1", "c1ccncc1", "CC(=O)Nc1ccccc1", "c1ccc(cc1)C(=O)O"]
    tfp, *_ = compute_features(train)
    out = active_learning_loop(
        MockSampler(train), MockScorer(), rounds=3,
        potency_predict=lambda X: np.full(len(X), 7.0),
        selectivity_predict=lambda X: np.full(len(X), 1.2),
        train_fp=tfp, spec=_SPEC, n_per_round=4, batch_size=2,
    )
    assert out["rounds"] == 3 and len(out["history"]) == 3 and len(out["trace"]) == 3
    for h in out["history"]:
        assert h["n_validated"] == 2
        assert all("structure_score" in b["scorer"] for b in h["batch"])


def test_retrain_hook_is_called_each_round():
    train = ["c1ccccc1", "c1ccncc1"]
    tfp, *_ = compute_features(train)
    calls = {"n": 0}

    def retrain(validated, pot, sel):
        calls["n"] += 1
        return pot, sel  # no-op refit for the test

    active_learning_loop(
        MockSampler(train), MockScorer(), rounds=2,
        potency_predict=lambda X: np.full(len(X), 7.0),
        selectivity_predict=lambda X: np.full(len(X), 1.0),
        train_fp=tfp, spec=_SPEC, n_per_round=2, batch_size=1, retrain=retrain,
    )
    assert calls["n"] == 2
