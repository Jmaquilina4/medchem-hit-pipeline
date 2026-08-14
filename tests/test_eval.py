"""Unit tests for evaluation-harness helpers (deterministic, offline)."""

from __future__ import annotations

import numpy as np

from medchem.eval.harness import _nn_tanimoto, _reg


def test_reg_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert _reg(y, y.copy())["r2"] == 1.0


def test_nn_tanimoto_max():
    fp_test = np.array([[1.0, 1.0, 0.0]])
    fp_train = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # to [1,1,0] -> 2/2 = 1.0 ; to [1,0,0] -> 1/2 = 0.5 ; max = 1.0
    assert _nn_tanimoto(fp_test, fp_train)[0] == 1.0


def test_nn_tanimoto_partial():
    fp_test = np.array([[1.0, 0.0, 1.0, 0.0]])
    fp_train = np.array([[1.0, 1.0, 0.0, 0.0]])
    # inter=1, union=3 -> 1/3
    assert abs(_nn_tanimoto(fp_test, fp_train)[0] - (1.0 / 3.0)) < 1e-6
