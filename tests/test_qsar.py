"""Integration test: featurize -> qsar on a tiny in-memory dataset (offline).

Exercises the real stage code (RDKit fingerprints/descriptors, RF + XGB,
scaffold-CV, conformal) end to end so CI catches regressions; values are
meaningless on 16 compounds, so we assert structure, not accuracy.
"""

from __future__ import annotations

import pandas as pd

from medchem.config import Config
from medchem.data.curate import curate_activities  # noqa: F401  (import sanity)
from medchem.features.featurize import featurize
from medchem.models.qsar import qsar
from medchem.pipeline.stage import StageContext, StageResult

_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "c1ccncc1",
    "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "c1ccc2ncccc2c1", "c1ccc(-c2ccccc2)cc1",
    "CC(=O)Nc1ccc(O)cc1", "CN1CCCCC1c1cccnc1", "c1c[nH]cn1", "c1ccoc1",
]


def _make_training_csv(tmp_path):
    path = tmp_path / "JAK1_training_ic50.csv"
    pd.DataFrame(
        {"canonical_smiles": _SMILES, "pIC50": [5.0 + 0.2 * i for i in range(len(_SMILES))]}
    ).to_csv(path, index=False)
    return str(path)


def test_featurize_then_qsar(tmp_path):
    curate_res = StageResult(name="curate", outputs={"potency_training": _make_training_csv(tmp_path)})

    feat_ctx = StageContext(
        config=Config(), workdir=str(tmp_path / "feat"), upstream={"curate": curate_res}
    )
    (tmp_path / "feat").mkdir()
    feat_res = featurize(feat_ctx)
    assert feat_res.metrics["n_compounds"] == len(_SMILES)
    assert feat_res.metrics["n_features"] == 2048 + 9  # ECFP4 bits + descriptors

    qsar_ctx = StageContext(
        config=Config(), workdir=str(tmp_path / "qsar"), upstream={"featurize": feat_res}
    )
    (tmp_path / "qsar").mkdir()
    qsar_res = qsar(qsar_ctx)
    assert "random_forest" in qsar_res.metrics["random_split"]
    assert "xgboost" in qsar_res.metrics["random_split"]
    assert "r2" in qsar_res.metrics["random_split"]["random_forest"]
    assert qsar_res.metrics["scaffold_cv_rf"]  # non-empty (enough scaffolds)
    assert "empirical_coverage" in qsar_res.metrics["conformal_rf"]


def test_meta_carries_era_labels_for_the_temporal_split(tmp_path):
    """The third hop. Era labels travel curate -> featurize(meta.csv) -> harness, and meta.csv was
    dropping them, which left the leakage fix inert while the harness reported it as applied. Each hop
    of a multi-hop artifact contract has to be asserted; only the consumer can reveal a silent drop."""
    import numpy as np
    import pandas as pd
    import pytest

    pytest.importorskip("rdkit")

    from medchem.config import Config
    from medchem.features.featurize import featurize
    from medchem.pipeline.stage import StageContext, StageResult

    smis = ["CC(=O)Nc1ccccc1", "c1ccc(cc1)C(=O)O", "c1ccncc1", "CCOc1ccccc1",
            "COc1ccc(cc1)CCN", "c1ccc2[nH]ccc2c1", "OCC1CCCCC1", "CN1CCN(CC1)c1ccccc1"]
    train = tmp_path / "potency_training.csv"
    pd.DataFrame({
        "canonical_smiles": smis,
        "pIC50": np.linspace(6.0, 9.0, len(smis)),
        "document_year": [2015, 2016, 2017, 2018, 2023, 2024, 2019, 2020],
        "pIC50_pre": [6.0, 6.4, 6.9, 7.3, np.nan, np.nan, 8.1, 8.6],
        "n_pre": [2, 1, 3, 1, 0, 0, 2, 1],
        "pIC50_post": [np.nan, np.nan, 7.2, np.nan, 8.5, 9.0, np.nan, np.nan],
        "n_post": [0, 0, 1, 0, 2, 1, 0, 0],
    }).to_csv(train, index=False)

    work = tmp_path / "w"
    work.mkdir()
    res = featurize(StageContext(
        config=Config.model_validate({"features": {"n_bits": 512, "radius": 2}}),
        workdir=str(work),
        upstream={"curate": StageResult(name="curate",
                                        outputs={"potency_training": str(train)})}))
    meta = pd.read_csv(res.outputs["meta"])
    assert {"pIC50_pre", "n_pre", "pIC50_post", "n_post"} <= set(meta.columns), (
        "meta.csv is the only channel to the harness; a dropped column silently disables the fix"
    )
    assert meta["pIC50_pre"].notna().sum() >= 5
