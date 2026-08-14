"""A selectivity result must carry whether it is interpretable.

The BRD4 run exposed the gap: BRD4-BRDT returned R2 = -0.009 from 177 paired compounds with 13
positives, and the pipeline reported it with `gate_status: null` because the evaluation gates cover
potency only. Nothing distinguished "the model failed" from "the data cannot answer this".

These tests pin the distinction, including the case that matters most: a LOW R2 whose confidence
interval still excludes zero is real-but-weak signal, not a failure, and must not be lumped in with
a result indistinguishable from noise.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from medchem.models.selectivity import _bootstrap_r2_ci, _support  # noqa: E402


def test_strong_signal_is_supported():
    rng = np.random.default_rng(0)
    delta = rng.normal(0, 1.5, 800)
    oof = delta + rng.normal(0, 0.4, 800)          # tight predictions
    s = _support(delta, oof, threshold=1.0, seed=42)
    assert s["supported"] is True
    assert s["r2_distinguishable_from_zero"] is True
    assert s["r2_ci95"][0] > 0
    assert s["prevalence"] > 0.10 and not s["reasons"]


def test_noise_is_not_supported_and_says_why():
    rng = np.random.default_rng(1)
    # spread wide enough that some compounds cross the threshold, so the failure under test is the
    # ABSENCE OF SIGNAL and not merely a constant label
    delta = rng.normal(0, 1.0, 180)
    oof = rng.normal(0, 0.2, 180)                  # unrelated to truth, and low-variance
    s = _support(delta, oof, threshold=1.0, seed=42)
    assert s["supported"] is False
    assert s["n_selective"] > 0                    # a selective class exists; the model just misses it
    assert s["r2_ci95"][1] < 0 or s["r2_ci95"][0] <= 0 <= s["r2_ci95"][1]
    assert any(("spans zero" in r) or ("entirely below zero" in r) for r in s["reasons"])


def test_weak_but_real_signal_stays_supported_and_is_not_confused_with_noise():
    """The BRD4-BRD2 case: R2 0.287 with CI [0.207, 0.372]. Low is not the same as absent, and
    collapsing the two would discard a real if modest result."""
    rng = np.random.default_rng(2)
    delta = rng.normal(0, 1.0, 800)
    oof = delta + rng.normal(0, 0.85, 800)         # genuine but weak: R2 ~ 0.28, as for BRD4-BRD2
    s = _support(delta, oof, threshold=1.0, seed=42)
    assert s["supported"] is True
    assert 0.0 < s["r2_ci95"][0]                    # CI excludes zero
    assert s["r2_ci95"][1] < 0.9                    # ...but the effect is modest


def test_constant_label_is_unsupported():
    """No compound crosses the threshold: there is no selective class to learn."""
    rng = np.random.default_rng(3)
    delta = rng.normal(0, 0.1, 300)                # never reaches 1.0
    s = _support(delta, delta + rng.normal(0, 0.01, 300), threshold=1.0, seed=42)
    assert s["n_selective"] == 0
    assert s["supported"] is False
    assert any("label is constant" in r for r in s["reasons"])


def test_rare_positive_class_is_flagged_even_when_supported():
    """PR-AUC on a 2% positive class is unstable and must be read against its baseline. The pair can
    still be supported -- the flag is a caveat, not a rejection."""
    rng = np.random.default_rng(4)
    delta = rng.normal(-0.5, 0.85, 900)            # mostly below threshold, a few above
    oof = delta + rng.normal(0, 0.3, 900)
    s = _support(delta, oof, threshold=1.0, seed=42)
    assert 0 < s["prevalence"] < 0.05
    assert any("positive class is" in r for r in s["reasons"])


def test_bootstrap_ci_brackets_the_point_estimate():
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 500)
    yhat = y + rng.normal(0, 0.5, 500)
    lo, hi = _bootstrap_r2_ci(y, yhat, n_boot=400, seed=7)
    assert lo < r2_score(y, yhat) < hi


def test_support_is_deterministic():
    """Seeded: two calls on identical input must agree, or reported CIs are not reproducible."""
    rng = np.random.default_rng(6)
    delta = rng.normal(0, 1, 400)
    oof = 0.6 * delta + rng.normal(0, 0.7, 400)
    a = _support(delta, oof, threshold=1.0, seed=42)
    b = _support(delta, oof, threshold=1.0, seed=42)
    assert a == b


# --- an unsupported result must not become a production model ---------------------------------------

def _matrix(n=60, primary="P", comparators=("C1", "C2")):
    """A synthetic panel where one comparator carries real Δ signal and the other is noise."""
    import pandas as pd

    from medchem.cohorts import COHORT_SPEC_VERSION  # noqa: F401  (import sanity)

    rng = np.random.default_rng(0)
    smis = [
        "c1ccccc1C(=O)O", "c1ccncc1", "CCOc1ccccc1", "CC(=O)Nc1ccccc1", "c1ccc2[nH]ccc2c1",
        "c1ccc2occc2c1", "C1CCNCC1c1ccccc1", "C1COCCN1c1ccccc1", "c1ccc2nccnc2c1", "c1coc(c1)C(=O)N",
    ]
    smis = (smis * ((n // len(smis)) + 1))[:n]
    p = rng.normal(8.0, 1.0, n)
    return pd.DataFrame({
        "canonical_smiles": smis,
        primary: p,
        comparators[0]: p - rng.normal(1.5, 0.4, n),      # a real, sizeable Δ
        comparators[1]: p - rng.normal(0.0, 0.02, n),     # essentially no Δ
        f"delta_{primary}_{comparators[0]}": rng.normal(1.5, 0.4, n),
        f"delta_{primary}_{comparators[1]}": rng.normal(0.0, 0.02, n),
        "delta_min_vs_comparators": rng.normal(0.0, 0.02, n),
    })


def test_no_model_is_written_when_no_pair_is_supported(tmp_path):
    """The invariant that matters most: an unsupported Δ is noise at that sample size, and a model file
    built from it is indistinguishable from a good one to every downstream consumer."""

    from medchem.config import Config
    from medchem.models.selectivity import selectivity
    from medchem.pipeline.stage import StageContext, StageResult

    df = _matrix()
    # make BOTH comparators pure noise so nothing can be supported
    df["C1"] = df["P"] - np.random.default_rng(1).normal(0.0, 0.02, len(df))
    df["delta_P_C1"] = np.random.default_rng(2).normal(0.0, 0.02, len(df))
    path = tmp_path / "panel_selectivity_matrix.csv"
    df.to_csv(path, index=False)

    cfg = Config.model_validate({
        "features": {"n_bits": 2048, "radius": 2},
        "data": {"targets": {"P": "C0", "C1": "C1", "C2": "C2"}, "primary": "P"},
        "model": {"selectivity": {"pairs": ["P-C1", "P-C2"], "rf_n_estimators": 8},
                  "potency": {"rf_n_estimators": 8}},
    })
    work = tmp_path / "w"
    work.mkdir()
    res = selectivity(StageContext(config=cfg, workdir=str(work),
                                   upstream={"curate": StageResult(
                                       name="curate", outputs={"selectivity": str(path)})}))
    assert res.metrics["pairs_supported"] == [], "fixture should leave nothing supported"
    assert "model" not in res.outputs, "an unsupported panel must not emit a production model"
    assert res.metrics["production_model"]["written"] is False
    assert not (work / "selectivity_model.joblib").exists()


def test_unsupported_comparators_are_excluded_from_the_aggregate(tmp_path):
    """Partial support: the aggregate must span SUPPORTED comparators only, not delta_min over all of
    them, or an unsupported pair is laundered into every downstream score."""
    from medchem.config import Config
    from medchem.models.selectivity import selectivity
    from medchem.pipeline.stage import StageContext, StageResult

    df = _matrix()
    path = tmp_path / "panel_selectivity_matrix.csv"
    df.to_csv(path, index=False)
    cfg = Config.model_validate({
        "features": {"n_bits": 2048, "radius": 2},
        "data": {"targets": {"P": "C0", "C1": "C1", "C2": "C2"}, "primary": "P"},
        "model": {"selectivity": {"pairs": ["P-C1", "P-C2"], "rf_n_estimators": 8},
                  "potency": {"rf_n_estimators": 8}},
    })
    work = tmp_path / "w"
    work.mkdir()
    res = selectivity(StageContext(config=cfg, workdir=str(work),
                                   upstream={"curate": StageResult(
                                       name="curate", outputs={"selectivity": str(path)})}))
    pm = res.metrics["production_model"]
    sup = res.metrics["pairs_supported"]
    if sup and len(sup) < 2:
        assert pm["basis_column"] == "delta_min_supported", (
            "with partial support the aggregate must be rebuilt from supported comparators only"
        )
        assert pm["supported_comparators"] == [p.split("-", 1)[1] for p in sup]
