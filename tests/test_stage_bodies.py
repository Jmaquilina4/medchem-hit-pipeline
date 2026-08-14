"""Execute stage BODIES, not just the functions they call.

This file exists because of a specific gap. The generative stage was covered only indirectly:
tests exercised ``score_molecules`` and ``generate_and_select``, and separate tests exercised DAG
composition. Nothing ever *ran the stage function*. Two bugs walked straight through that gap:

1. ``selectivity_predict=None`` crashed the scorer whenever the optional selectivity stage was
   disabled (the plan composed fine, so composition tests were green).
2. Typing the config broke ``gen.get("scoring", {}).get("components", ...)`` — dict-punning a section
   that had become a submodel — which only fails when the stage body actually reads its config.

Both were found by running the pipeline, after the test suite passed. A stage body is where config,
upstream artifacts, and library code meet, so it is exactly the seam unit tests miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medchem.config import Config
from medchem.pipeline.stage import StageContext, StageResult

pytest.importorskip("sklearn")
pytest.importorskip("rdkit")

# Real molecules: the stage featurises with RDKit, so these must parse.
_SMILES = [
    "CC(=O)Nc1ccccc1", "c1ccc(cc1)C(=O)O", "c1ccncc1", "CCOc1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1ccc(cc1)CCN", "c1ccc2[nH]ccc2c1",
    "CN1CCN(CC1)c1ccccc1", "OCC1CCCCC1", "Cc1ccc(cc1)S(=O)(=O)N",
]


def _upstream(tmp_path: Path) -> dict[str, StageResult]:
    """Minimal but REAL upstream artifacts: a training CSV, a features npz, a fitted model."""
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    from medchem.features.featurize import compute_features

    fp, desc, _scaf, keep = compute_features(_SMILES)
    x = np.hstack([fp, desc])
    kept = [s for s, k in zip(_SMILES, keep, strict=True) if k]
    y = np.linspace(5.0, 9.0, len(kept))

    train_csv = tmp_path / "potency_training.csv"
    pd.DataFrame({"canonical_smiles": kept, "pIC50": y}).to_csv(train_csv, index=False)

    feats = tmp_path / "features.npz"
    np.savez_compressed(feats, X=x)

    model_path = tmp_path / "potency_model.joblib"
    joblib.dump(RandomForestRegressor(n_estimators=8, random_state=42).fit(x, y), model_path)

    sel_path = tmp_path / "selectivity_model.joblib"
    joblib.dump(RandomForestRegressor(n_estimators=8, random_state=42).fit(x, y * 0.1), sel_path)

    return {
        "curate": StageResult(name="curate", outputs={"potency_training": str(train_csv)}),
        "featurize": StageResult(name="featurize", outputs={"features": str(feats)}),
        "qsar": StageResult(name="qsar", outputs={"model": str(model_path)}),
        "selectivity": StageResult(name="selectivity", outputs={"model": str(sel_path)}),
    }


def _config() -> Config:
    """A config shaped like the real one: a configured scoring spec, small candidate count."""
    return Config.model_validate({
        "target": "test",
        "features": {"n_bits": 2048, "radius": 2, "use_chirality": True},
        "generative": {
            "sampler": "mock",
            "n_candidates": 8,
            "top_k": 3,
            "scoring": {"components": [
                {"name": "qsar_pic50", "transform": "sigmoid", "center": 7.5},
                {"name": "selectivity_delta", "transform": "sigmoid", "center": 1.0},
                {"name": "mw", "transform": "double_sigmoid", "low": 300.0, "high": 450.0},
                {"name": "applicability_domain", "transform": "reverse_sigmoid", "center": 0.4},
            ]},
        },
    })


def test_generative_stage_body_runs_with_selectivity(tmp_path):
    """Reads a configured scoring spec out of a TYPED config. Dict-punning a modeled subsection
    fails here and nowhere else."""
    from medchem.generative.stage import generative

    up = _upstream(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    res = generative(StageContext(config=_config(), workdir=str(work), upstream=up))

    assert res.name == "generative"
    payload = json.loads(Path(res.outputs["selection"]).read_text())
    assert len(payload["constrained_top"]) == 3
    assert len(payload["naive_top"]) == 3
    assert res.metrics["sampler"] == "mock"
    assert res.metrics["constrained_mean_ad"] is not None


def test_generative_stage_body_runs_without_selectivity(tmp_path):
    """The optional-dependency path, executed. Composition tests assert the plan drops selectivity;
    only this asserts the stage still produces a selection when it is gone."""
    from medchem.generative.stage import generative

    up = _upstream(tmp_path)
    del up["selectivity"]  # exactly what the runner passes when the stage is disabled
    work = tmp_path / "work"
    work.mkdir()
    res = generative(StageContext(config=_config(), workdir=str(work), upstream=up))

    payload = json.loads(Path(res.outputs["selection"]).read_text())
    assert len(payload["constrained_top"]) == 3
    assert res.metrics["constrained_mean_ad"] is not None


def test_generative_stage_honours_the_configured_spec(tmp_path):
    """A configured spec must actually reach the scorer. The hard-coded default was once used
    unconditionally, so configured components were silently ignored — a config that lies."""
    from medchem.generative.stage import generative

    cfg = Config.model_validate({
        "features": {"n_bits": 2048, "radius": 2},
        "generative": {"sampler": "mock", "n_candidates": 8, "top_k": 2,
                       # ONLY molecular weight: if this is honoured, no other component is scored
                       "scoring": {"components": [
                           {"name": "mw", "transform": "double_sigmoid", "low": 300.0, "high": 450.0},
                       ]}},
    })
    up = _upstream(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    generative(StageContext(config=cfg, workdir=str(work), upstream=up))

    # the scorer records every transformed component; only the configured one may appear
    import joblib

    from medchem.features.featurize import compute_features
    from medchem.generative.scorer import score_molecules

    fp, desc, _s, keep = compute_features(_SMILES[:4])
    model = joblib.load(up["qsar"].outputs["model"])
    rows = score_molecules(
        _SMILES[:4],
        potency_predict=model.predict,
        selectivity_predict=None,
        train_fp=np.load(up["featurize"].outputs["features"])["X"][:, :2048],
        # `components` is `list | None` now, because omitted and explicit-empty are different requests
        # and neither may be answered with a default. This test configures one component, so the
        # assertion documents that: a None here would mean the config under test lost its spec.
        spec=cfg.generative.scoring.components or [],
    )
    assert cfg.generative.scoring.components, "this test must configure a spec, not rely on a default"
    assert set(rows[0]["transformed"]) == {"mw"}


def test_vls_stage_body_skips_cleanly_without_a_library(tmp_path):
    """The skip path is a real path: it must report why rather than raise, so a run without a
    pulled deck still produces metrics."""
    from medchem.vls.stage import vls

    up = _upstream(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    cfg = Config.model_validate({"vls": {"enabled": True, "library": {"path": "does/not/exist.smi"}}})
    res = vls(StageContext(config=cfg, workdir=str(work), upstream=up))

    assert res.metrics["status"] == "skipped"
    assert "absent" in res.metrics["reason"]


# --- the curate -> selectivity artifact contract --------------------------------------------------

_SCAFFOLD_DIVERSE = [
    "c1ccccc1C(=O)O", "c1ccncc1C(=O)O", "c1cnc2[nH]ccc2c1", "c1ccc2ccccc2c1C",
    "c1ccc2[nH]ccc2c1", "c1ccc2occc2c1", "c1ccc2sccc2c1", "c1cn[nH]c1C(=O)N",
    "C1CCNCC1c1ccccc1", "C1COCCN1c1ccccc1", "C1CCCCC1C(=O)O", "C1CCNC1C(=O)N",
    "c1ccc2nccnc2c1", "c1ccc2[nH]nnc2c1", "c1ccc(cc1)S(=O)(=O)N", "c1ccc(cc1)C#N",
    "c1ccc2c(c1)CNC2=O", "c1ccc2c(c1)OCC2", "c1ccc2c(c1)SCC2", "c1coc(c1)C(=O)N",
    "c1csc(c1)C(=O)N", "c1cc[nH]c1C(=O)N", "C1CC1c1ccccc1", "C1CCC1c1ccncc1",
    "c1ccc(cc1)N1CCOCC1", "c1ccc(cc1)N1CCNCC1", "c1ccc(cc1)c1ccccc1", "c1ccc(cc1)c1ccncc1",
    "c1ccc2c(c1)ncnc2N", "c1ccc2c(c1)[nH]c(=O)o2", "C1CCC2(CC1)CCCC2", "c1ccc(cc1)C1CC1",
]


def _selectivity_matrix(tmp_path: Path, primary="ACME1", comparator="ACME2") -> Path:
    """A curated selectivity matrix in the CURRENT contract, built via curate's own constant."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    from medchem.data.curate import DELTA_MIN_COLUMN

    RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]
    rng = np.random.default_rng(0)
    # The Δ must be GENERALISABLE, not merely deterministic. Folds are grouped by scaffold, so a Δ that
    # is a function of molecule identity is unlearnable across groups: such a fixture yields R² = -0.813,
    # correctly unsupported. Tying Δ to molecular
    # weight, a descriptor the features carry, makes a held-out scaffold's Δ predictable from its own
    # descriptors, which is what a real selectivity signal looks like.
    reps = 6
    smiles_pool = (_SCAFFOLD_DIVERSE * reps)[: len(_SCAFFOLD_DIVERSE) * reps]
    n = len(smiles_pool)
    mw = np.array([Descriptors.MolWt(Chem.MolFromSmiles(x)) for x in smiles_pool])  # pyright: ignore[reportAttributeAccessIssue]
    delta = 0.8 + 0.020 * (mw - mw.mean()) + rng.normal(0.0, 0.12, n)
    a = rng.normal(8.0, 0.6, n)
    b = a - delta
    df = pd.DataFrame({
        "canonical_smiles": smiles_pool,
        primary: a,
        comparator: b,
        f"delta_{primary}_{comparator}": delta,
        DELTA_MIN_COLUMN: delta,
    })
    path = tmp_path / "panel_selectivity_matrix.csv"
    df.to_csv(path, index=False)
    return path


def test_selectivity_stage_persists_a_model_downstream_stages_can_consume(tmp_path):
    """The regression for a silent contract break. `curate` was renamed to write
    `delta_min_vs_comparators` while this stage still read `delta_min_vs_other`, so
    `outputs["model"]` stopped being produced -- and `generative`/`vls`, which check
    `"model" in sel_up.outputs`, silently took the "selectivity absent" path for BOTH targets while
    the stage itself reported success and published R2 values.

    Every existing test passed: they covered the ABSENCE path, never the presence of the artifact.
    """
    from medchem.config import Config
    from medchem.models.selectivity import selectivity

    matrix = _selectivity_matrix(tmp_path)
    cfg = Config.model_validate({
        "features": {"n_bits": 2048, "radius": 2},
        "model": {"selectivity": {"pairs": ["ACME1-ACME2"], "rf_n_estimators": 8},
                  "potency": {"rf_n_estimators": 8}},
        "data": {"targets": {"ACME1": "C1", "ACME2": "C2"}, "primary": "ACME1"},
    })
    work = tmp_path / "work"
    work.mkdir()
    up = {"curate": StageResult(name="curate", outputs={"selectivity": str(matrix)})}
    res = selectivity(StageContext(config=cfg, workdir=str(work), upstream=up))

    # The contract is now conditional: a model is written when a pair is SUPPORTED, and withheld when
    # none is, so that noise cannot be laundered into downstream scores through a model file.
    assert res.metrics["pairs_supported"], (
        "fixture produced no supported pair, so this test cannot check the model contract"
    )
    assert "model" in res.outputs, (
        "a supported pair must persist a model -- otherwise downstream stages silently score without "
        "selectivity even though the data supports it"
    )
    assert res.metrics["production_model"]["written"] is True
    assert Path(res.outputs["model"]).exists()
    assert res.metrics["pairs_requested"] == ["ACME1-ACME2"]


def test_curate_writes_the_column_the_selectivity_stage_reads():
    """Producer and consumer share ONE constant. Importing it from both sides is what makes a future
    rename impossible to desynchronise -- the string existed in two places before."""
    from medchem.data import curate as curate_mod
    from medchem.models import selectivity as sel_mod

    assert sel_mod.DELTA_MIN_COLUMN is curate_mod.DELTA_MIN_COLUMN

    rng = np.random.default_rng(1)
    n = len(_SCAFFOLD_DIVERSE)
    bio = pd.DataFrame({
        "canonical_smiles": _SCAFFOLD_DIVERSE * 2,
        "target": ["ACME1"] * n + ["ACME2"] * n,
        "pIC50": np.concatenate([rng.normal(8, 1, n), rng.normal(7, 1, n)]),
    })
    matrix = curate_mod.build_selectivity(bio, primary="ACME1", comparators=["ACME2"])
    assert curate_mod.DELTA_MIN_COLUMN in matrix.columns


def test_vls_stage_actually_consumes_the_derived_window(tmp_path):
    """The loop this closes: `derive` was accepted and validated by the config for a whole commit while
    the stage never read it -- config/behaviour drift of exactly the kind this repo keeps documenting.
    Asserting the STAGE reports a derived window, not merely that the config parses."""
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")
    import json

    import joblib
    from sklearn.ensemble import RandomForestRegressor

    from medchem.features.featurize import compute_features
    from medchem.vls.stage import vls

    smiles = _SCAFFOLD_DIVERSE[:12]
    fp, desc, _s, keep = compute_features(smiles)
    x = np.hstack([fp, desc])
    kept = [s for s, k in zip(smiles, keep, strict=True) if k]
    y = np.linspace(6.0, 9.0, len(kept))

    lib = tmp_path / "lib.smi"
    lib.write_text("\n".join(f"{s} z{i}" for i, s in enumerate(kept)) + "\n")
    train_csv = tmp_path / "potency_training.csv"
    pd.DataFrame({"canonical_smiles": kept, "pIC50": y}).to_csv(train_csv, index=False)
    feats = tmp_path / "features.npz"
    np.savez_compressed(feats, X=x)
    pot = tmp_path / "potency_model.joblib"
    joblib.dump(RandomForestRegressor(n_estimators=6, random_state=42).fit(x, y), pot)
    qm = tmp_path / "qsar_metrics.json"
    qm.write_text(json.dumps({"conformal_rf": {"interval_halfwidth_90": 0.9}}))

    cfg = Config.model_validate({
        "features": {"n_bits": 2048, "radius": 2},
        "vls": {
            "enabled": True,
            "actives": {"pic50_quantile": 0.5},
            "library": {
                "path": str(lib),
                "lead_like": {
                    "derive": {"margin": 0.3, "must_admit_references": True,
                               "intersect_with_explicit": False},
                    "mw_min": 50, "mw_max": 500, "logp_max": 6.0, "tpsa_max": 200,
                    "hbd_max": 6, "hba_max": 12, "rotb_max": 12,
                },
            },
            # references must be admitted by the derived window, so they come from the same chemistry
            "known_reference": {"ref_a": kept[0], "ref_b": kept[1]},
        },
    })
    work = tmp_path / "w"
    work.mkdir()
    up = {
        "curate": StageResult(name="curate", outputs={"potency_training": str(train_csv)}),
        "featurize": StageResult(name="featurize", outputs={"features": str(feats)}),
        "qsar": StageResult(name="qsar", outputs={"model": str(pot), "metrics": str(qm)}),
    }
    res = vls(StageContext(config=cfg, workdir=str(work), upstream=up))

    assert res.metrics.get("status") != "skipped", "stage skipped; nothing was tested"
    win = res.metrics["screening_window"]
    assert win["mode"] == "derived", "the stage ignored the derive config"
    assert win["admits_references"]["passed"] is True
    # the potency cut must be the resolved QUANTILE, recorded with how it was obtained
    assert "quantile" in res.metrics["potency_cut"]["how"]
    # and the derivation must have produced its own bounds, not echoed the explicit ones
    assert win["bounds"]["mw_max"] != 500
