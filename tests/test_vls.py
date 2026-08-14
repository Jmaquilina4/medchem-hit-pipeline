"""VLS Tier-0 (library prep) + Tier-1 (annotate + stratify, NO selection) — CPU, deterministic."""

from __future__ import annotations

import numpy as np

from medchem.features.featurize import compute_features
from medchem.vls.library import LeadLikeBounds, prepare_library
from medchem.vls.screen import (
    STRATA,
    Thresholds,
    _fp_only,
    assign_priority,
    assign_stratum,
    screen_library,
)

T = Thresholds()  # JAK1 defaults


# --- the invariant everything else rests on -------------------------------------------------
def test_lean_fingerprint_matches_the_canonical_featurizer():
    # Every similarity term MUST live in the space the models were trained in, so the
    # speed-optimized fingerprint-only path has to agree bit-for-bit with compute_features.
    smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "CC(=O)Nc1ccccc1", "CCO"]
    canonical, _desc, _scaf, keep_c = compute_features(smiles, n_bits=2048, radius=2)
    lean, keep_l = _fp_only(smiles, n_bits=2048, radius=2, use_chirality=True)
    assert keep_c == keep_l
    assert np.array_equal(canonical, lean)


# --- stratification: labels, never verdicts -------------------------------------------------
def test_strata_boundaries():
    assert assign_stratum(0.80, T) == "S1_in_domain"
    assert assign_stratum(0.50, T) == "S1_in_domain"   # inclusive at the floor
    assert assign_stratum(0.40, T) == "S2_near_domain"
    assert assign_stratum(0.20, T) == "S3_far_novel"


def test_qsar_abstains_outside_the_domain_however_potent_the_prediction():
    # off-manifold the model has no skill (temporal R2 ~ 0) -> abstain, neither promote nor demote
    assert assign_priority(point=9.5, delta=2.0, sim_to_train=0.3, t=T) == "abstain_out_of_ad"


def test_below_hit_floor_is_a_label_not_a_drop():
    # in-domain AND conformal upper bound (4.0 + 0.955) < 6.0: confidently weak, still docked.
    assert assign_priority(point=4.0, delta=0.0, sim_to_train=0.8, t=T) == "qsar_below_hit_floor"


def test_priority_labels_require_the_high_confidence_band():
    assert assign_priority(point=7.5, delta=1.2, sim_to_train=0.8, t=T) == "front_of_queue"
    # lower bound 8.5 - 0.955 = 7.545 >= 7.4 and delta >= 1.5 -> FEP-first
    assert assign_priority(point=8.5, delta=1.6, sim_to_train=0.8, t=T) == "fep_first"
    # in-domain but below the high-confidence band -> no priority
    assert assign_priority(point=7.5, delta=1.2, sim_to_train=0.55, t=T) == "none"


# --- Tier 0 ---------------------------------------------------------------------------------
def test_tier0_dedup_and_unparseable():
    records = [("CCO", "a"), ("CCO", "b"), ("not_a_smiles", "c"), ("c1ccccc1", "d")]
    res = prepare_library(records, bounds=LeadLikeBounds(mw=(0.0, 1e4), logp_max=1e3),
                          apply_pains=False, workers=1)
    assert res.dropped["unparseable"] == 1
    assert res.dropped["duplicate"] == 1
    assert res.n_prepared == 2
    counts = [n for _, n in res.attrition]
    assert counts == sorted(counts, reverse=True)  # monotonically shrinking funnel


def test_tier0_physchem_drops_tiny_molecule():
    res = prepare_library([("CCO", "a")], bounds=LeadLikeBounds(mw=(300.0, 460.0)),
                          apply_pains=False, workers=1)
    assert res.dropped["physchem"] == 1
    assert res.n_prepared == 0


# --- Tier 1: annotates everything, discards nothing -----------------------------------------
def _screen(records, **kw):
    train_fp, *_ = compute_features(["c1ccccc1", "c1ccncc1", "CC(=O)Nc1ccccc1"])
    return screen_library(
        records,
        potency_predict=lambda x: np.full(len(x), 7.5),
        selectivity_predict=lambda x: np.full(len(x), 1.2),
        train_fp=train_fp, thresholds=T, **kw,
    )


def test_tier1_selects_nothing_every_compound_reaches_docking():
    records = [{"smiles": s, "id": f"c{i}"}
               for i, s in enumerate(["c1ccccc1", "CCO", "CCCCCCCCCCCC"])]
    res = _screen(records)
    assert res.n_screened == 3
    # THE reversal: the docking queue is the whole screened set — no budget, no truncation.
    assert res.n_docking_queue == res.n_screened
    assert sum(res.strata.values()) == 3
    assert set(res.strata) == set(STRATA)


def test_tier1_reports_all_three_similarity_terms_separately():
    # benzene is in the training set; the "patent" ref here is pyridine.
    act_fp, _ = _fp_only(["CC(=O)Nc1ccccc1"], n_bits=2048, radius=2, use_chirality=True)
    pat_fp, _ = _fp_only(["c1ccncc1"], n_bits=2048, radius=2, use_chirality=True)
    res = _screen(
        [{"smiles": "c1ccccc1", "id": "benzene"}],
        active_fp=act_fp, known_ref_fp=pat_fp, known_ref_names=["pyridine"],
    )
    cols = res.columns
    assert cols["sim_to_train"][0] == 1.0        # identical to a training compound
    assert 0.0 <= cols["sim_to_actives"][0] < 1.0  # a different question, different value
    assert 0.0 <= cols["sim_to_known_reference"][0] <= 1.0
    assert cols["nearest_known_reference"][0] == "pyridine"


def test_tier1_flags_fto_without_dropping():
    # query IS the patent reference -> sim_to_known_reference 1.0 -> flagged, but still in the queue.
    pat_fp, _ = _fp_only(["c1ccccc1"], n_bits=2048, radius=2, use_chirality=True)
    res = _screen(
        [{"smiles": "c1ccccc1", "id": "benzene"}],
        known_ref_fp=pat_fp, known_ref_names=["ref"],
    )
    assert res.known_reference_flagged == 1
    assert res.n_docking_queue == 1  # flagged for review, NOT removed


def test_tier1_unparseable_is_counted_not_silently_skipped():
    res = _screen([{"smiles": "c1ccccc1", "id": "ok"}, {"smiles": "$$bad$$", "id": "bad"}])
    assert res.n_screened == 1
    assert res.n_unparseable == 1
    assert res.columns["stratum"][1] == "unparseable"
