"""The generative scoring engine + the reward-hacking → recovery property (ADR 0003)."""

from __future__ import annotations

from medchem.generative.scoring import (
    aggregate,
    double_sigmoid,
    score_components,
    sigmoid,
)


def test_sigmoid_monotonic_through_center():
    assert sigmoid(10.0, 7.5) > sigmoid(7.5, 7.5) > sigmoid(5.0, 7.5)
    assert abs(float(sigmoid(7.5, 7.5)) - 0.5) < 1e-9


def test_double_sigmoid_is_a_window():
    assert double_sigmoid(375.0, 300.0, 450.0, k=1.0) > 0.9   # inside the MW window
    assert double_sigmoid(150.0, 300.0, 450.0, k=1.0) < 0.1   # too small
    assert double_sigmoid(600.0, 300.0, 450.0, k=1.0) < 0.1   # too large


def test_geometric_mean_resists_reward_hacking():
    # A "reward hack": maxes four objectives, fails the fifth. A "balanced" molecule is
    # merely good at all five.
    hack = [0.99, 0.99, 0.99, 0.99, 0.02]
    balanced = [0.7, 0.7, 0.7, 0.7, 0.7]

    # Under a SUM/mean reward the hack WINS — one dominant objective carries it (v1's bug).
    assert aggregate(hack, method="sum") > aggregate(balanced, method="sum")

    # Under a GEOMETRIC MEAN the balanced molecule wins — the failed objective tanks the hack.
    assert aggregate(hack, method="geometric_mean") < aggregate(balanced, method="geometric_mean")
    assert aggregate(hack, method="geometric_mean") < 0.5
    assert abs(aggregate(balanced, method="geometric_mean") - 0.7) < 1e-6


def test_score_components_from_config_spec():
    spec = [
        {"name": "qsar_pic50", "transform": "sigmoid", "center": 7.5},
        {"name": "mw", "transform": "double_sigmoid", "low": 300.0, "high": 450.0},
        {"name": "applicability_domain", "transform": "reverse_sigmoid", "center": 0.4},
    ]
    agg, per = score_components({"qsar_pic50": 8.5, "mw": 375.0, "applicability_domain": 0.2}, spec)
    assert 0.0 <= agg <= 1.0
    assert per["qsar_pic50"] > 0.5 and per["mw"] > 0.9

    # A missing component scores 0 and (under geometric mean) tanks the whole score —
    # objectives can't be silently dropped.
    agg_missing, _ = score_components({"qsar_pic50": 8.5, "mw": 375.0}, spec)
    assert agg_missing < 1e-3
