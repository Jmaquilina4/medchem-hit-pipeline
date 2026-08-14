"""Unit tests for the curation logic (offline; no ChEMBL network)."""

from __future__ import annotations

import math

import pandas as pd

from medchem.data.curate import (
    POTENCY_TRAINING_SCHEMA,
    _unit_factor,
    build_selectivity,
    curate_activities,
)


def test_unit_factor_normalizes_to_nM():
    units = pd.Series(["nM", "uM", "mM", "M", "weird"])
    factors = _unit_factor(units).tolist()
    assert factors[0] == 1.0
    assert factors[1] == 1_000.0
    assert factors[2] == 1_000_000.0
    assert factors[3] == 1_000_000_000.0
    assert math.isnan(factors[4])


def test_curate_converts_units_and_dedups():
    # Two IC50 measurements of the same compound in different units that both
    # normalize to 100 nM (pIC50 = 7.0); a salt is stripped to its parent.
    raw = pd.DataFrame(
        {
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL1", "CHEMBL2"],
            "canonical_smiles": ["CCO.[Na+]", "CCO", "c1ccccc1"],
            "standard_type": ["IC50", "IC50", "IC50"],
            "standard_value": [100.0, 0.1, 1000.0],
            "standard_units": ["nM", "uM", "nM"],
            "standard_relation": ["=", "=", "="],
            "pchembl_value": [None, None, None],
        }
    )
    out = curate_activities(raw, "JAK1")

    # CHEMBL1's two measurements collapse to one row (median pIC50 = 7.0)
    row = out[out["molecule_chembl_id"] == "CHEMBL1"].iloc[0]
    assert row["assay_count"] == 2
    assert abs(row["pIC50"] - 7.0) < 1e-6
    assert row["canonical_smiles"] == "CCO"  # salt stripped
    assert (out["target"] == "JAK1").all()
    POTENCY_TRAINING_SCHEMA.validate(out[["canonical_smiles", "pIC50"]])


def test_build_selectivity_computes_direct_delta():
    all_bio = pd.DataFrame(
        {
            "target": ["JAK1", "JAK2"],
            "canonical_smiles": ["CCO", "CCO"],
            "pIC50": [8.0, 6.0],
        }
    )
    sel = build_selectivity(all_bio)
    assert "delta_JAK1_JAK2" in sel.columns
    assert abs(sel["delta_JAK1_JAK2"].iloc[0] - 2.0) < 1e-6


def test_curate_enforces_ic50_only_and_drops_flagged_rows():
    # Regression guard: a Ki row carrying a pChEMBL value must NOT leak into the IC50
    # set; a ChEMBL-flagged row and an inactive-commented row are dropped; and a clean
    # UNFLAGGED IC50 row must survive (guards the pandas-3.0 astype(str)-keeps-NaN bug
    # that once dropped every unflagged row).
    raw = pd.DataFrame(
        {
            "molecule_chembl_id": ["K1", "V1", "F1", "I1"],
            "canonical_smiles": ["CCN", "CCC", "CCCl", "CCF"],
            "standard_type": ["Ki", "IC50", "IC50", "IC50"],
            "standard_value": [None, None, 50.0, None],
            "standard_units": [None, None, "nM", None],
            "standard_relation": [None, None, "=", None],
            "pchembl_value": [8.5, 7.0, None, 6.0],
            "data_validity_comment": [None, None, "Potential author error", None],
            "activity_comment": [None, None, None, "Not Active"],
        }
    )
    out = curate_activities(raw, "JAK1")
    smi = set(out["canonical_smiles"])
    assert "CCN" not in smi   # Ki (pChEMBL) leak blocked — IC50-only enforced
    assert "CCCl" not in smi  # ChEMBL-flagged-invalid row dropped
    assert "CCF" not in smi   # inactive activity_comment dropped
    assert "CCC" in smi       # clean unflagged IC50 row survives (pandas-3.0 NaN guard)
    assert len(out) == 1


def test_era_split_labels_reach_the_training_set():
    """The leakage fix was INERT for a whole run: curation computed pIC50_pre and the stage then
    dropped it when selecting columns, so the harness never saw a pre-cutoff label and kept training on
    the all-years median. Computing a guard and discarding it is worse than not having it, because the
    metrics claim the fix is in effect."""
    import pandas as pd

    from medchem.data.curate import curate_activities

    raw = pd.DataFrame({
        "canonical_smiles": ["CCO"] * 4 + ["c1ccccc1"] * 2,
        "standard_type": ["IC50"] * 6,
        "standard_value": [100, 200, 50, 80, 300, 400],
        "standard_units": ["nM"] * 6,
        "document_year": [2015, 2016, 2023, 2024, 2016, 2017],
        "molecule_chembl_id": ["C1"] * 4 + ["C2"] * 2,
    })
    out = curate_activities(raw, "T", temporal_cutoff_year=2022)
    assert {"pIC50_pre", "n_pre", "pIC50_post", "n_post"} <= set(out.columns)
    ccо = out[out["canonical_smiles"] == "CCO"].iloc[0]
    assert ccо["n_pre"] == 2 and ccо["n_post"] == 2, "measurements must be partitioned by era"
    # the all-years median differs from the pre-cutoff median: that difference IS the leak
    assert ccо["pIC50"] != ccо["pIC50_pre"]
    # a compound with no post-cutoff data has no post label
    benzene = out[out["canonical_smiles"] == "c1ccccc1"].iloc[0]
    assert benzene["n_post"] == 0


def test_no_cutoff_means_no_era_columns():
    """Absent a configured cutoff, curation must not invent era labels -- their presence is what tells
    the harness it may train on pre-cutoff medians."""
    import pandas as pd

    from medchem.data.curate import curate_activities

    raw = pd.DataFrame({
        "canonical_smiles": ["CCO"] * 2,
        "standard_type": ["IC50"] * 2,
        "standard_value": [100, 200],
        "standard_units": ["nM"] * 2,
        "document_year": [2015, 2023],
        "molecule_chembl_id": ["C1"] * 2,
    })
    out = curate_activities(raw, "T")
    assert "pIC50_pre" not in out.columns
