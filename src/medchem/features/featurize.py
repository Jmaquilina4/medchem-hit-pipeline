"""Featurization stage: SMILES -> ECFP4 (chiral) fingerprints + RDKit descriptors.

Reads the curated JAK1 training set and emits a feature matrix (``features.npz``)
plus a ``meta.csv`` carrying the SMILES, label, Bemis-Murcko scaffold **group key**,
and ``document_year`` (the temporal-split key).

Config-driven (``configs/<target>.yaml`` -> ``features``): fingerprint size/radius,
chirality, and the descriptor list all come from the config so a new target is a config
change, not a code edit. Chirality is ON by default (stereochemistry matters here).

Empty-scaffold handling: RDKit returns an empty Murcko scaffold for acyclic molecules
(and on failure). Those would all collapse into one GroupKFold group and corrupt
scaffold-CV, so an empty scaffold falls back to the molecule's own canonical SMILES —
each such compound becomes its own group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from medchem.pipeline.stage import StageContext, StageResult, stage


def _descriptor_funcs() -> dict:
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    return {
        "MolWt": lambda m: float(Descriptors.MolWt(m)),  # pyright: ignore[reportAttributeAccessIssue]
        "MolLogP": lambda m: float(Crippen.MolLogP(m)),  # pyright: ignore[reportAttributeAccessIssue]
        "TPSA": lambda m: float(rdMolDescriptors.CalcTPSA(m)),
        "NumHDonors": lambda m: float(Lipinski.NumHDonors(m)),  # pyright: ignore[reportAttributeAccessIssue]
        "NumHAcceptors": lambda m: float(Lipinski.NumHAcceptors(m)),  # pyright: ignore[reportAttributeAccessIssue]
        "QED": lambda m: float(QED.qed(m)),
        "NumRotatableBonds": lambda m: float(Lipinski.NumRotatableBonds(m)),  # pyright: ignore[reportAttributeAccessIssue]
        "NumAromaticRings": lambda m: float(rdMolDescriptors.CalcNumAromaticRings(m)),
        "FractionCSP3": lambda m: float(rdMolDescriptors.CalcFractionCSP3(m)),
    }


# Default descriptor set (used when the config doesn't specify one, and by callers
# such as the selectivity stage that share this featurizer).
_DEFAULT_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "QED", "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
]


def _murcko_group(smiles: str) -> str:
    """Bemis-Murcko scaffold, or the molecule's own SMILES if it has none (acyclic)."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles)
    except Exception:
        scaf = ""
    return scaf if scaf else smiles


def compute_features(
    smiles: list[str],
    *,
    n_bits: int = 2048,
    radius: int = 2,
    use_chirality: bool = True,
    descriptors: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[bool]]:
    """Return (fingerprints, descriptors, scaffold-group-keys, keep-mask) for a SMILES list.

    Shared by the featurize and selectivity stages so both use identical
    ECFP4(chiral) + descriptor featurization and the same scaffold grouping.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    names = descriptors if descriptors is not None else _DEFAULT_DESCRIPTORS
    funcs = _descriptor_funcs()
    unknown = [n for n in names if n not in funcs]
    if unknown:
        raise ValueError(f"unknown descriptor(s) in config: {unknown}; known: {sorted(funcs)}")

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=use_chirality
    )
    fps: list[np.ndarray] = []
    descs: list[list[float]] = []
    scaffolds: list[str] = []
    keep: list[bool] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            keep.append(False)
            continue
        fps.append(gen.GetFingerprintAsNumPy(mol))
        descs.append([funcs[n](mol) for n in names])
        scaffolds.append(_murcko_group(smi))
        keep.append(True)
    x_fp = np.asarray(fps, dtype=np.float32) if fps else np.zeros((0, n_bits), dtype=np.float32)
    x_desc = (
        np.asarray(descs, dtype=np.float32)
        if descs
        else np.zeros((0, len(names)), dtype=np.float32)
    )
    return x_fp, x_desc, scaffolds, keep


@stage("discovery", "featurize", deps=("curate",), config_keys=("features",))
def featurize(ctx: StageContext) -> StageResult:
    """Compute ECFP4(chiral) + descriptor features for the JAK1 training set."""
    feat = ctx.config.features
    n_bits, radius, use_chirality = feat.n_bits, feat.radius, feat.use_chirality
    # `is None`, not `or`: None means "the standard block", [] means "no descriptor columns". Read
    # with `or`, an explicit `descriptors: []` produced the nine standard columns and the ablation
    # measured nothing. See the field's comment in medchem.config.
    descriptors = list(_DEFAULT_DESCRIPTORS if feat.descriptors is None else feat.descriptors)

    df = pd.read_csv(ctx.upstream["curate"].outputs["potency_training"])
    x_fp, x_desc, scaffolds, keep = compute_features(
        df["canonical_smiles"].tolist(),
        n_bits=n_bits, radius=radius, use_chirality=use_chirality, descriptors=descriptors,
    )
    df = df[keep].reset_index(drop=True)
    x = np.hstack([x_fp, x_desc])
    y = df["pIC50"].to_numpy(dtype=np.float32)
    year = (
        df["document_year"].to_numpy()
        if "document_year" in df.columns
        else np.full(len(df), np.nan)
    )

    out = Path(ctx.workdir)
    np.savez_compressed(out / "features.npz", X=x, y=y)
    meta = {
        "canonical_smiles": df["canonical_smiles"].to_numpy(),
        "pIC50": y,
        "scaffold": scaffolds,
        "document_year": year,
    }
    # Era-split labels must survive THIS hop too. They are produced by curation and consumed by the
    # evaluation harness, and meta.csv is the only channel between them -- dropping them here left the
    # temporal-leakage fix inert for two runs while the harness reported it as applied. A three-hop
    # artifact contract needs each hop to carry the column, and only the consumer can tell you it did
    # not, which is why the harness records its label source.
    for col in ("pIC50_pre", "n_pre", "pIC50_post", "n_post"):
        if col in df.columns:
            meta[col] = df[col].to_numpy()
    pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)

    return StageResult(
        name="featurize",
        outputs={"features": str(out / "features.npz"), "meta": str(out / "meta.csv")},
        metrics={
            "n_compounds": int(len(df)),
            "n_features": int(x.shape[1]),
            "n_bits": n_bits,
            "n_descriptors": len(descriptors),
            "use_chirality": use_chirality,
        },
    )
