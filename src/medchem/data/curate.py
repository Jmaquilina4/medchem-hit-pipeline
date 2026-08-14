"""Curation stage: raw ChEMBL IC50 activity tables -> clean, deduplicated pIC50 data.

Ports the v1 IC50-only curation, hardened per the 2026-07 audit:

- salt-strip to the **organic parent** (RDKit ``FragmentParent`` — removes known
  counterions instead of naively keeping the largest fragment) + isomeric
  (chirality-preserving) canonicalization
- unit normalization (nM/uM/mM/M -> nM) and ``pIC50 = 9 - log10(value_nM)``; units are
  converted from the ``standard_units`` column, never trusted from a label
- **IC50-only is enforced on every row** — a pChEMBL value alone no longer smuggles
  Ki/Kd/EC50 into the pIC50 column (standard_type must be IC50)
- ChEMBL-flagged rows (``data_validity_comment``) and inactive ``activity_comment`` rows
  are dropped
- **one** median over all measurements per (target, canonical_smiles) -> no
  median-of-medians; carries ``assay_count`` and the earliest ``document_year`` (the
  temporal-split key)
- selectivity matrix (direct-delta vs JAK2/JAK3/TYK2) + panel coverage
- the **full-range** JAK1 set is the model input (all quality IC50 data, best coverage
  for QSAR); a drug-like subset is emitted *separately* for downstream candidate triage
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from medchem.cohorts import COHORT_SPEC_VERSION, apply_to_activities, select_assays
from medchem.pipeline.stage import StageContext, StageResult, stage

_NAN_TOKENS = {"", "nan", "none", "null", "na", "<na>", "n/a"}
_DEFAULT_ACCEPT_TYPES = {"ic50"}
_REL_OK = {"=", "~", "≈", "~=", "==", ""}
# activity_comment values that mean "do not use this as a quantitative potency"
_INACTIVE_COMMENTS = {
    "not active", "inactive", "inconclusive", "not determined",
    "unable to determine", "no data", "not evaluated",
}

_COLUMN_ALIASES = {
    "smiles": "canonical_smiles",
    "type": "standard_type",
    "value": "standard_value",
    "units": "standard_units",
    "relation": "relation",
    "standard_relation": "relation",
}
_NEEDED = [
    "canonical_smiles", "molecule_chembl_id", "standard_type", "standard_value",
    "standard_units", "relation", "pchembl_value", "assay_type", "target_chembl_id",
    "activity_comment", "data_validity_comment", "document_year",
]

# The aggregate selectivity column, named ONCE. The selectivity stage imports this rather than
# repeating the string: when this was renamed from "delta_min_vs_other", the consumer kept the old
# name, so the production selectivity model silently stopped being persisted and every downstream
# stage took the "selectivity absent" path while the stage itself reported success.
DELTA_MIN_COLUMN = "delta_min_vs_comparators"

POTENCY_TRAINING_SCHEMA = pa.DataFrameSchema(
    {
        "canonical_smiles": pa.Column(str, nullable=False),
        "pIC50": pa.Column(float, pa.Check.in_range(3.0, 11.0), nullable=False),
    },
    strict=False,
    coerce=True,
)


def _is_nan_token(value: object) -> bool:
    try:
        return str(value).strip().lower() in _NAN_TOKENS
    except Exception:
        return True


def _prepare_smiles(raw_smiles: object) -> str | float:
    """Salt-strip to the organic parent and return isomeric (chiral) canonical SMILES.

    Uses RDKit ``FragmentParent`` (known-counterion-aware) rather than naive
    largest-fragment selection, so a large organic counterion can't outweigh the drug.
    """
    from rdkit import Chem

    if not isinstance(raw_smiles, str) or _is_nan_token(raw_smiles):
        return np.nan
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return np.nan

    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        parent = rdMolStandardize.FragmentParent(mol)
        if parent is not None and parent.GetNumHeavyAtoms() > 0:
            mol = parent
    except Exception:
        # Fallback: largest *carbon-containing* fragment (still counterion-aware-ish).
        from rdkit.Chem import rdMolDescriptors

        frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
        carbon = [f for f in frags if any(a.GetSymbol() == "C" for a in f.GetAtoms())]
        pool = carbon or frags
        if pool:
            mol = max(pool, key=lambda m: rdMolDescriptors.CalcNumHeavyAtoms(m))

    try:
        return Chem.MolToSmiles(mol, canonical=True)  # isomeric=True default -> stereo kept
    except Exception:
        return np.nan


def _unit_factor(units: pd.Series) -> pd.Series:
    u = units.astype(str).str.strip().str.lower()
    factor = pd.Series(np.nan, index=u.index, dtype="float64")
    factor[u.isin({"nm", "nanomolar"})] = 1.0
    factor[u.isin({"um", "µm", "micromolar"})] = 1_000.0
    factor[u.isin({"mm", "millimolar"})] = 1_000_000.0
    factor[u.isin({"m", "molar"})] = 1_000_000_000.0
    return factor


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    present = {k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns}
    df = df.rename(columns=present)
    df = df.loc[:, ~df.columns.duplicated()]
    for col in _NEEDED:
        if col not in df.columns:
            df[col] = np.nan
    return df


# Cohort classification lives in medchem.data.cohort as a VERSIONED spec. It was briefly
# implemented here as ad-hoc regex helpers; two implementations of the same rule is how the
# producer and consumer drift apart, which this repository has already been bitten by.

def curate_activities(
    df: pd.DataFrame,
    target_name: str,
    *,
    accept_types: Iterable[str] = _DEFAULT_ACCEPT_TYPES,
    temporal_cutoff_year: int | None = None,
) -> pd.DataFrame:
    """Curate one target's raw activity table into a clean pIC50 table.

    Returns one row per canonical structure: target, molecule_chembl_id (representative),
    canonical_smiles, pIC50 (median over ALL measurements), assay_count, document_year
    (earliest), source.
    """
    accept = {t.lower() for t in accept_types}
    df = _standardize_columns(df)

    # pandas-stubs is over-strict about apply's func overload here; runtime is fine.
    df["canonical_smiles"] = df["canonical_smiles"].apply(_prepare_smiles)  # pyright: ignore[reportCallIssue, reportArgumentType]
    df = df[df["canonical_smiles"].notna()]

    # Drop ChEMBL-flagged rows: any non-empty data_validity_comment is a quality flag
    # ("Potential author error", "Outside typical range", ...); keep only unflagged rows.
    # NB pandas 3.0 astype(str) PRESERVES NaN, so test the raw column's isna() too —
    # otherwise every unflagged (missing-comment) row is silently dropped.
    dvc = df["data_validity_comment"]
    dvc_str = dvc.astype(str).str.strip().str.lower()
    df = df[dvc.isna() | dvc_str.isin(_NAN_TOKENS)]
    # Drop rows explicitly commented inactive / not-determined (a missing comment is kept).
    ac = df["activity_comment"].astype(str).str.strip().str.lower()
    df = df[~ac.isin(_INACTIVE_COMMENTS)]

    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    stype = df["standard_type"].astype(str).str.lower().str.strip()
    rel = df["relation"].astype(str).str.strip()
    rel_missing = df["relation"].isna() | rel.str.lower().isin({"", "nan", "none"})
    std_val = pd.to_numeric(df["standard_value"], errors="coerce")
    value_nM = std_val * _unit_factor(df["standard_units"])
    value_nM = value_nM.where(np.isfinite(value_nM), np.nan)
    df["value_nM"] = value_nM

    # IC50-only, enforced on BOTH the pChEMBL branch and the value-conversion branch.
    is_accepted = stype.isin(accept)
    rel_ok = rel_missing | rel.isin(_REL_OK)
    keep = is_accepted & (df["pchembl_value"].notna() | (df["value_nM"].notna() & rel_ok))
    df = df[keep]

    df["pIC50"] = df["pchembl_value"]
    need = df["pIC50"].isna() & df["value_nM"].notna()
    nm = pd.to_numeric(df.loc[need, "value_nM"], errors="coerce")
    nm = nm[(nm > 0) & np.isfinite(nm)]
    if not nm.empty:
        df.loc[nm.index, "pIC50"] = 9.0 - np.log10(nm.to_numpy(dtype=float))
    df = df[df["pIC50"].between(3, 11)]

    df["document_year"] = pd.to_numeric(df["document_year"], errors="coerce")

    # ONE median over all measurements per structure (fixes v1 median-of-medians).
    agg = (
        df.groupby("canonical_smiles", dropna=False)
        .agg(
            pIC50=("pIC50", "median"),
            assay_count=("pIC50", "size"),
            document_year=("document_year", "min"),
            document_year_max=("document_year", "max"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
        )
        .reset_index()
    )

    # TEMPORAL LEAKAGE FIX. The median above spans every year while document_year is the EARLIEST,
    # so a compound first reported in 2015 and re-measured in 2023 lands in a pre-2022 training set
    # carrying a label informed by the 2023 measurement. Measured: 8.2% of JAK1 and 5.5% of BRD4
    # training compounds. Labels are therefore also computed per era, and the evaluation harness
    # trains on `pIC50_pre` rather than on a label that saw the future.
    if temporal_cutoff_year is not None:
        pre = df[df["document_year"] < temporal_cutoff_year]
        post = df[df["document_year"] >= temporal_cutoff_year]
        pre_agg = pre.groupby("canonical_smiles", dropna=False).agg(
            pIC50_pre=("pIC50", "median"), n_pre=("pIC50", "size")).reset_index()
        post_agg = post.groupby("canonical_smiles", dropna=False).agg(
            pIC50_post=("pIC50", "median"), n_post=("pIC50", "size")).reset_index()
        agg = agg.merge(pre_agg, on="canonical_smiles", how="left")
        agg = agg.merge(post_agg, on="canonical_smiles", how="left")
        agg[["n_pre", "n_post"]] = agg[["n_pre", "n_post"]].fillna(0).astype(int)
        agg["temporal_cutoff_year"] = int(temporal_cutoff_year)
    agg.insert(0, "target", target_name)
    agg["source"] = "ChEMBL"
    keep_cols = ["target", "molecule_chembl_id", "canonical_smiles", "pIC50",
                 "assay_count", "document_year", "document_year_max", "source"]
    keep_cols += [c for c in ("pIC50_pre", "n_pre", "pIC50_post", "n_post",
                              "temporal_cutoff_year") if c in agg.columns]
    return agg[keep_cols]


def build_selectivity(
    all_bio: pd.DataFrame,
    primary: str | None = None,
    comparators: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compound x target pIC50 matrix with direct-delta selectivity columns.

    ``primary`` and ``comparators`` come from config (``data.primary`` and the other configured
    targets). They were previously hardcoded to the JAK panel, which meant this function could not be
    reused for another target family even though the config already carried the information.

    Delta columns are named ``delta_<primary>_<comparator>`` and the aggregate is
    ``delta_min_vs_comparators`` -- target-neutral in structure, with the target names appearing only
    as data.
    """
    pivot = all_bio.pivot_table(
        index="canonical_smiles", columns="target", values="pIC50", aggfunc="median"
    )
    if primary is None:
        # fall back to the most-measured target rather than a hardcoded name
        counts = all_bio["target"].value_counts() if "target" in all_bio else pd.Series(dtype=int)
        primary = str(counts.index[0]) if len(counts) else ""
    if comparators is None:
        comparators = [c for c in pivot.columns if str(c) != primary]

    for other in comparators:
        if primary in pivot.columns and other in pivot.columns:
            pivot[f"delta_{primary}_{other}"] = pivot[primary] - pivot[other]
    prefix = f"delta_{primary}_"
    delta_cols = [c for c in pivot.columns if str(c).startswith(prefix)]
    if delta_cols:
        pivot[DELTA_MIN_COLUMN] = pivot[delta_cols].min(axis=1, skipna=True)
    return pivot.reset_index()


def _druglike_mask(train: pd.DataFrame) -> pd.Series:
    """Physchem window + PAINS/Brenk. Used for the *downstream* triage subset only."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)

    flags = []
    for smi in train["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            flags.append(False)
            continue
        alert = catalog.GetFirstMatch(mol) is not None
        mw = Descriptors.MolWt(mol)  # pyright: ignore[reportAttributeAccessIssue]
        logp = Crippen.MolLogP(mol)  # pyright: ignore[reportAttributeAccessIssue]
        hbd = Lipinski.NumHDonors(mol)  # pyright: ignore[reportAttributeAccessIssue]
        hba = Lipinski.NumHAcceptors(mol)  # pyright: ignore[reportAttributeAccessIssue]
        rotb = Lipinski.NumRotatableBonds(mol)  # pyright: ignore[reportAttributeAccessIssue]
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        ok = (
            not alert and 150 <= mw <= 650 and -2 <= logp <= 6
            and hbd <= 5 and hba <= 10 and tpsa <= 140 and rotb <= 10
        )
        flags.append(bool(ok))
    return pd.Series(flags, index=train.index)


@stage("discovery", "curate", deps=("data_pull",), config_keys=("data", "curation"))
def curate(ctx: StageContext) -> StageResult:
    """Curate all pulled targets, build selectivity + the primary-target training sets."""
    cfg_data = ctx.config.data
    # The VALIDATED value, with no second fallback. This read `... or ["IC50"]`, which meant an empty
    # configured list -- the one value that cannot work -- silently became the default, and it meant this
    # module carried a copy of a default that config.py already owns.
    accept_types = list(cfg_data.activity_types)
    # The primary target and its comparators are CONFIGURED, not assumed. Artifact names
    # below are target-neutral so a second target changes data, not the contract.
    # Typed reads. `str(x or "")` and `dict(x or {})` normalise None to an empty value rather than
    # substituting a default, so they were harmless -- but they are the same SHAPE as the two defects
    # just fixed, and the regression guard in tests/test_config.py has to be able to tell them apart.
    # Reading the typed fields directly removes the ambiguity from the file.
    primary = cfg_data.primary or ""
    configured = list(cfg_data.targets)
    comparators = [t for t in configured if t != primary]

    # Cohort + cutoff come from `curation`, not `data`: `data` is hashed into the pull key, so a
    # cohort living there would fetch a second copy of the same snapshot.
    cfg_cur = getattr(ctx.config, "curation", None)
    cohort_cfg = getattr(cfg_cur, "assay_cohort", None)
    cutoff = getattr(cfg_cur, "temporal_cutoff_year", None)

    upstream = ctx.upstream["data_pull"].outputs
    workdir = Path(ctx.workdir)

    frames: list[pd.DataFrame] = []
    cohort_report: dict[str, Any] = {}
    for name, path in upstream.items():
        # The pull emits one activity table AND one assay table per target, plus provenance. Only the
        # activity tables are targets; iterating everything treated "JAK1_assays" as a target, which
        # the cohort guard then correctly refused because that pseudo-target has no assay metadata of
        # its own. Caught by the guard rather than producing a bogus fifth target.
        if name == "provenance" or name.endswith("_assays"):
            continue
        raw = pd.read_csv(path, low_memory=False)
        # Assay metadata comes from the same pull, so descriptions and activities are one snapshot.
        assay_path = upstream.get(f"{name}_assays")
        cohort_name = getattr(cohort_cfg, "name", "target_associated")
        if assay_path and Path(assay_path).exists():
            adf = pd.read_csv(assay_path, low_memory=False)
            desc_col = "description" if "description" in adf.columns else None
            assays = dict(zip(adf["assay_chembl_id"],
                              adf[desc_col] if desc_col else [None] * len(adf), strict=False))
            # ChEMBL's own structured fields, passed alongside the descriptions. Spec-1.2 cohorts
            # require them and RAISE if they are missing, rather than quietly deciding on text alone --
            # the text-only decision admitted a cell-based reporter assay and 23 tandem-domain
            # constructs. Absent columns are passed through as None so the cohort's fail-closed branch
            # sees "cannot confirm" instead of an empty dict it might mistake for "nothing to check".
            structured = {
                str(r["assay_chembl_id"]): {
                    "bao_format": r.get("bao_format"),
                    "assay_type": r.get("assay_type"),
                    "confidence_score": r.get("confidence_score"),
                }
                for _, r in adf.iterrows()
            }
            sel = select_assays(assays, cohort_name, structured=structured)
            raw, attrition = apply_to_activities(raw, sel)
            cohort_report[name] = {**sel.as_dict(), "activity_attrition": attrition}
        else:
            # No assay metadata: the cohort CANNOT be applied. Say so rather than silently pooling --
            # an unfiltered run is a different and weaker claim than a domain-specific one.
            cohort_report[name] = {
                "cohort": cohort_name,
                "applied": False,
                "reason": (
                    f"no {name}_assays.csv in the pull, so assay descriptions are unavailable and no "
                    f"cohort can be applied. Re-pull to enable cohort selection."
                ),
            }
            if cohort_name != "target_associated":
                raise ValueError(
                    f"config requests cohort {cohort_name!r} but the pull for {name} carries no assay "
                    f"metadata. Refusing to run: the result would be target-associated data labelled "
                    f"as a filtered cohort."
                )
        frames.append(curate_activities(
            raw, name, accept_types=accept_types, temporal_cutoff_year=cutoff))

    cols = ["target", "molecule_chembl_id", "canonical_smiles", "pIC50",
            "assay_count", "document_year", "document_year_max", "source"]
    # Era-split labels ride along when curation produced them. Computing them and then dropping them
    # here is what made the leakage fix inert: the harness never saw a pre-cutoff label and kept
    # training on the all-years median.
    era_cols = ["pIC50_pre", "n_pre", "pIC50_post", "n_post", "temporal_cutoff_year"]
    if frames:
        cols += [c for c in era_cols if c in frames[0].columns]
    all_bio = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    all_bio_path = workdir / "panel_activity_curated.csv"
    all_bio.to_csv(all_bio_path, index=False)

    selectivity = (
        build_selectivity(all_bio, primary=primary or None, comparators=comparators or None)
        if not all_bio.empty else pd.DataFrame()
    )
    sel_path = workdir / "panel_selectivity_matrix.csv"
    selectivity.to_csv(sel_path, index=False)

    # FULL-range training set for the PRIMARY target (already one row per canonical_smiles).
    train_cols = ["canonical_smiles", "pIC50", "document_year"]
    train_cols += [c for c in ("pIC50_pre", "n_pre", "pIC50_post", "n_post") if c in all_bio.columns]
    primary_train = (
        all_bio[all_bio["target"] == primary][train_cols].reset_index(drop=True)
    )
    POTENCY_TRAINING_SCHEMA.validate(primary_train)
    primary_path = workdir / "potency_training.csv"
    primary_train.to_csv(primary_path, index=False)

    # Drug-like SUBSET — emitted for downstream candidate triage, NOT for QSAR training.
    druglike = primary_train[_druglike_mask(primary_train)] if not primary_train.empty else primary_train
    druglike_path = workdir / "potency_training_druglike.csv"
    druglike.to_csv(druglike_path, index=False)

    metrics = {
        "curated_rows": int(len(all_bio)),
        "primary_target": primary,
        # The cohort report is not decoration: without it a reader cannot tell whether this is a
        # domain-specific model or a target-associated one, and those support different claims.
        "assay_cohort": {
            "name": getattr(cohort_cfg, "name", "target_associated"),
            "label": (getattr(cohort_cfg, "label", "")
                      or getattr(cohort_cfg, "name", "target_associated")),
            "spec_version": COHORT_SPEC_VERSION,
            "per_target": cohort_report,
        },
        "temporal_labels": {
            "cutoff_year": cutoff,
            "note": (
                "when a cutoff is set, pIC50_pre is the median of PRE-cutoff measurements only; "
                "training on the all-years median would let post-cutoff data inform labels of "
                "compounds assigned to train by their earliest year"
            ) if cutoff else "no cutoff configured: labels span all years",
        },
        "primary_compounds": int(len(primary_train)),
        "primary_druglike": int(len(druglike)),
        "per_target": {t: int((all_bio["target"] == t).sum()) for t in all_bio["target"].unique()},
    }
    (workdir / "curate_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return StageResult(
        name="curate",
        outputs={
            "all_bio": str(all_bio_path),
            "selectivity": str(sel_path),
            "potency_training": str(primary_path),
            "potency_training_druglike": str(druglike_path),
        },
        metrics=metrics,
    )
