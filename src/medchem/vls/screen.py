"""VLS Tier-1: ligand **annotation + stratification** — deliberately NOT selection.

Design reversal (2026-07-24, evidence in ADR 0005 §Tier-2-selection-retired): this tier used
to *choose* which compounds reached docking, under a ``docking_budget`` and a
"far-novel is never docked" rule. Both were written on the premise that docking is the
expensive step. It isn't — Uni-Dock fast mode puts a ~10⁶-compound library at roughly one
GPU-day, so capping the deck buys nothing and costs expected best-score and scaffold count
(docking-score-vs-library-size scaling is log-linear with no measured saturation from 10⁵ to
>10⁹). Worse, the retired cap had **zero enrichment**: a random 10.7% sample captured almost
exactly 10.7% of the in-domain compounds that existed.

So Tier-1 now **deletes nothing**. It annotates every prepared compound and emits strata that
are *reported labels*, used downstream to allocate the genuinely scarce budgets (Boltz-2
co-folding, and especially free energy at ~6–12 GPU-hours **per ligand**) and to stratify the
enrichment measurement. Docking gets the whole library.

Three similarity terms are computed separately, because conflating them is a real error — the
previous single ``nn_tanimoto`` answered two different questions at once:

- ``sim_to_train`` — max Tanimoto to the **full curated training set**. This is the
  **applicability domain**: an epistemic claim about where the QSAR's ±conformal interval is
  valid. It defines the strata.
- ``sim_to_actives`` — max Tanimoto to the **potent** actives. A **priority/interest** signal.
  It does *not* predict docking hit rate (prospective campaign hits sit at Tc < 0.6 to known
  actives), so it must never act as a gate.
- ``sim_to_known_reference`` — max Tanimoto to marketed reference drugs. The **known-reference similarity
  ceiling**, and the fast-follower's central trap: novelty relative to marketed compounds turns on 2D structure, so
  maximizing similarity to the originator maximizes the chance of landing inside its
  composition-of-matter claims. Flagged for review, never silently dropped. **Triage only —
  Tanimoto has no legal standing** (Markush claims are defined by substitution patterns; a
  compound at Tc 0.4 can sit inside a claim while one at 0.7 sits outside).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medchem.features.featurize import _DEFAULT_DESCRIPTORS, compute_features

# Applicability-domain strata. Reported labels — every stratum is docked.
STRATA = ("S1_in_domain", "S2_near_domain", "S3_far_novel")

# Advisory QSAR labels, meaningful ONLY inside S1 (elsewhere the model abstains).
PRIORITY_LABELS = ("fep_first", "front_of_queue", "none", "qsar_below_hit_floor", "abstain_out_of_ad")


@dataclass
class Thresholds:
    """Tier-1 thresholds. Config-driven (``vls.tier1``); every value is fingerprint-specific
    and non-transferable across featurizations, so the fp spec is recorded alongside."""

    # Applicability domain — derived empirically from the measured out-of-fold residual /
    # coverage curve (scripts/derive_ad_threshold.py), not from folklore.
    in_domain: float = 0.5
    borderline: float = 0.35
    high_conf: float = 0.6
    # QSAR advisory labels
    conformal_halfwidth: float = 0.955
    hit_floor: float = 6.0        # 1 µM
    prio_point: float = 7.0       # ~100 nM
    prio_delta: float = 1.0
    fep_lower: float = 7.4        # upadacitinib-class, on the conformal LOWER bound
    fep_delta: float = 1.5
    # Known-reference similarity
    known_reference_flag: float = 0.65


@dataclass
class ScreenResult:
    n_screened: int
    n_unparseable: int
    strata: dict[str, int]
    priority: dict[str, int]
    known_reference_flagged: int
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    similarity_summary: dict = field(default_factory=dict)

    @property
    def n_docking_queue(self) -> int:
        """Everything parseable is docked — Tier-2 performs no selection."""
        return self.n_screened


def _fp_only(
    smiles: list[str], *, n_bits: int, radius: int, use_chirality: bool
) -> tuple[np.ndarray, list[bool]]:
    """Morgan fingerprints only (no descriptors/scaffolds) for speed at ~10⁶ scale.

    Mirrors ``compute_features``' fingerprint exactly (same generator parameters) — asserted
    by a test, since every similarity term must live in the space the models were trained in.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]  # present at runtime; rdkit stubs omit it
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=use_chirality
    )
    fps: list[np.ndarray] = []
    keep: list[bool] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            keep.append(False)
            continue
        fps.append(gen.GetFingerprintAsNumPy(mol))
        keep.append(True)
    mat = np.asarray(fps, dtype=np.float32) if fps else np.zeros((0, n_bits), np.float32)
    return mat, keep


def _tanimoto(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Full Tanimoto matrix between two bit-vector sets (query x ref)."""
    inter = query @ ref.T
    union = query.sum(1, keepdims=True) + ref.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def assign_stratum(sim_to_train: float, t: Thresholds) -> str:
    """Applicability-domain stratum. A label, not a verdict — all strata are docked."""
    if sim_to_train >= t.in_domain:
        return "S1_in_domain"
    if sim_to_train >= t.borderline:
        return "S2_near_domain"
    return "S3_far_novel"


def assign_priority(point: float, delta: float, sim_to_train: float, t: Thresholds) -> str:
    """Advisory QSAR label used to allocate the scarce downstream budgets.

    Outside S1 the model **abstains** — a prediction there is not trustworthy (temporal
    R² ≈ 0 off-manifold), so it is neither a promotion nor a demotion.
    """
    if sim_to_train < t.in_domain:
        return "abstain_out_of_ad"
    if point + t.conformal_halfwidth < t.hit_floor:
        return "qsar_below_hit_floor"   # confidently weak AND in-domain: labelled, still docked
    if sim_to_train >= t.high_conf and point - t.conformal_halfwidth >= t.fep_lower and delta >= t.fep_delta:
        return "fep_first"
    if sim_to_train >= t.high_conf and point >= t.prio_point and delta >= t.prio_delta:
        return "front_of_queue"
    return "none"


def screen_library(
    records: list[dict],
    *,
    potency_predict,
    selectivity_predict,
    train_fp: np.ndarray,
    active_fp: np.ndarray | None = None,
    known_ref_fp: np.ndarray | None = None,
    known_ref_names: list[str] | None = None,
    thresholds: Thresholds | None = None,
    n_bits: int = 2048,
    radius: int = 2,
    use_chirality: bool = True,
    descriptors: list[str] | None = None,
    batch_size: int = 20000,
) -> ScreenResult:
    """Annotate + stratify a prepared library in one streaming pass. Selects nothing.

    Returns column arrays (memory-efficient at ~10⁶ scale — the caller assembles the table)
    plus the strata census, which is the headline scientific output of this tier.
    """
    t = thresholds or Thresholds()
    # `is None`, not `or`. This was the last consumer still collapsing the two requests, and it is the
    # one that matters most: the screen feeds its matrix to a model TRAINED by `featurize`, so if the
    # config asks for `descriptors: []` and featurize honours it while the screen substitutes nine
    # columns, the loaded model is handed a matrix of the wrong width. That surfaces as a shape error
    # from inside sklearn, thousands of compounds into a screen, with nothing pointing at the config.
    names = list(_DEFAULT_DESCRIPTORS if descriptors is None else descriptors)
    train_fp = np.asarray(train_fp, dtype=np.float32)
    n = len(records)

    sim_train = np.full(n, -1.0, np.float32)
    sim_act = np.full(n, -1.0, np.float32)
    sim_known_ref = np.full(n, -1.0, np.float32)
    pat_near = np.full(n, -1, np.int16)
    pot = np.full(n, np.nan, np.float32)
    sel = np.full(n, np.nan, np.float32)
    parsed = np.zeros(n, bool)

    for start in range(0, n, batch_size):
        block = records[start : start + batch_size]
        smiles = [r["smiles"] for r in block]
        x_fp, x_desc, _scaf, keep = compute_features(
            smiles, n_bits=n_bits, radius=radius, use_chirality=use_chirality, descriptors=names
        )
        pos = np.array([start + i for i, k in enumerate(keep) if k], dtype=int)
        if not pos.size:
            continue
        parsed[pos] = True
        x = np.hstack([x_fp, x_desc])
        pot[pos] = np.asarray(potency_predict(x), dtype=float)
        if selectivity_predict is not None:
            sel[pos] = np.asarray(selectivity_predict(x), dtype=float)
        fp32 = x_fp.astype(np.float32)
        sim_train[pos] = _tanimoto(fp32, train_fp).max(1)
        if active_fp is not None and len(active_fp):
            sim_act[pos] = _tanimoto(fp32, np.asarray(active_fp, np.float32)).max(1)
        if known_ref_fp is not None and len(known_ref_fp):
            pm = _tanimoto(fp32, np.asarray(known_ref_fp, np.float32))
            sim_known_ref[pos] = pm.max(1)
            pat_near[pos] = pm.argmax(1).astype(np.int16)

    # Vectorized stratification + advisory labels.
    strat = np.where(
        sim_train >= t.in_domain, "S1_in_domain",
        np.where(sim_train >= t.borderline, "S2_near_domain", "S3_far_novel"),
    ).astype(object)
    strat[~parsed] = "unparseable"
    prio = np.array(
        [
            assign_priority(float(pot[i]), float(sel[i]), float(sim_train[i]), t)
            if parsed[i] else "unparseable"
            for i in range(n)
        ],
        dtype=object,
    )

    ok = parsed
    strata_counts = {s: int((strat == s).sum()) for s in STRATA}
    priority_counts = {p: int((prio == p).sum()) for p in PRIORITY_LABELS}
    names_list = list(known_ref_names or [])
    nearest = [
        names_list[i] if (0 <= i < len(names_list)) else "" for i in pat_near.tolist()
    ]

    def _summary(a: np.ndarray) -> dict:
        v = a[ok & (a >= 0)]
        if not v.size:
            return {}
        return {
            "median": round(float(np.median(v)), 3),
            "p90": round(float(np.quantile(v, 0.9)), 3),
            "p99": round(float(np.quantile(v, 0.99)), 3),
            "max": round(float(v.max()), 3),
        }

    return ScreenResult(
        n_screened=int(ok.sum()),
        n_unparseable=int((~ok).sum()),
        strata=strata_counts,
        priority=priority_counts,
        known_reference_flagged=int((ok & (sim_known_ref >= t.known_reference_flag)).sum()),
        columns={
            "id": np.array([r["id"] for r in records], dtype=object),
            "smiles": np.array([r["smiles"] for r in records], dtype=object),
            "pred_pic50": pot,
            "pred_pic50_lower": pot - t.conformal_halfwidth,
            "pred_pic50_upper": pot + t.conformal_halfwidth,
            "pred_selectivity_delta": sel,
            "sim_to_train": sim_train,
            "sim_to_actives": sim_act,
            "sim_to_known_reference": sim_known_ref,
            "nearest_known_reference": np.array(nearest, dtype=object),
            "stratum": strat,
            "qsar_label": prio,
            "parsed": parsed,
        },
        similarity_summary={
            "sim_to_train": _summary(sim_train),
            "sim_to_actives": _summary(sim_act),
            "sim_to_known_reference": _summary(sim_known_ref),
            "fingerprint": f"morgan r={radius} nbits={n_bits} chirality={use_chirality}",
        },
    )
