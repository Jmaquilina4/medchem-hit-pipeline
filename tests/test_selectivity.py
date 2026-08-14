"""Unit tests for the selectivity scoring helpers (deterministic, offline)."""

from __future__ import annotations

import numpy as np

from medchem.models.selectivity import _classification, _reg


def test_reg_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = _reg(y, y.copy())
    assert m["r2"] == 1.0
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0


def test_classification_perfect_ranking():
    # two selective (Δ≥1), two not; score ranks them perfectly
    delta_true = np.array([2.0, 1.5, 0.5, 0.0])
    score = np.array([2.0, 1.5, 0.5, 0.0])
    m = _classification(delta_true, score, 1.0)
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0
    # top-10% -> top-1 compound is selective; base rate 0.5 -> enrichment 2.0
    assert m["top10_enrichment"] == 2.0


def test_classification_degenerate_all_one_class():
    delta_true = np.array([0.0, 0.1, 0.2])  # none selective
    m = _classification(delta_true, np.array([0.0, 0.1, 0.2]), 1.0)
    assert np.isnan(m["pr_auc"])
    assert np.isnan(m["roc_auc"])
