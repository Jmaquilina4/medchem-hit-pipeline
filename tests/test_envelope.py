"""Deriving a screening window from a target's own actives, and the guard that makes it safe.

The window was tuned on one target and rejected 5 of 7 of a second target's reference compounds — a
screen structurally unable to find its own known chemistry while reporting success. Hand-widening the
numbers was the first repair and is barely better: those bounds came from seven compounds a human
picked. These tests cover the derivation, the two constraints that keep it honest, and the guard.
"""

from __future__ import annotations

import pytest

from medchem.vls.envelope import check_admits, derive_envelope, resolve_potency_cut


def test_potency_cut_prefers_a_quantile_over_an_absolute():
    """'Potent' is target-relative. An absolute 8.0 discards JQ1 (~7.1) on a bromodomain."""
    values = [5.0, 6.0, 7.0, 8.0, 9.0]
    cut, how = resolve_potency_cut(values, quantile=0.75, absolute=8.0)
    assert cut == pytest.approx(8.0)
    assert "quantile" in how
    cut2, how2 = resolve_potency_cut(values, quantile=0.5, absolute=8.0)
    assert cut2 == pytest.approx(7.0), "the quantile must win when supplied"
    assert "quantile" in how2


def test_absolute_is_used_only_when_no_quantile_is_given():
    cut, how = resolve_potency_cut([1.0, 2.0], quantile=None, absolute=7.5)
    assert cut == 7.5 and "explicit absolute" in how


def test_neither_threshold_is_an_error():
    with pytest.raises(ValueError, match="neither a potency quantile nor an absolute"):
        resolve_potency_cut([1.0], quantile=None, absolute=None)


def test_quantile_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="must be in"):
        resolve_potency_cut([1.0, 2.0], quantile=1.5, absolute=None)


def test_envelope_is_wider_than_the_observed_actives():
    """The central constraint: a window fitted tightly to known actives DEFINES the chemotype, so the
    screen can only rediscover it. Bounds must sit outside the quantile range."""
    mws = list(range(300, 500, 5))
    env = derive_envelope({"MolWt": [float(x) for x in mws]}, quantiles=(0.05, 0.95), margin=0.15)
    import numpy as np
    lo, hi = np.quantile(mws, 0.05), np.quantile(mws, 0.95)
    assert env.bounds["mw_min"] < lo
    assert env.bounds["mw_max"] > hi


def test_zero_margin_reproduces_the_quantile_range():
    mws = [float(x) for x in range(300, 500, 5)]
    env = derive_envelope({"MolWt": mws}, quantiles=(0.05, 0.95), margin=0.0)
    import numpy as np
    assert env.bounds["mw_max"] == pytest.approx(float(np.quantile(mws, 0.95)), abs=1e-3)


def test_quantiles_resist_a_single_outlier():
    """A lone unusual compound must not set a screening bound; the outer 5% is where salt forms and
    mis-annotations live."""
    mws = [400.0] * 100 + [5000.0]
    env = derive_envelope({"MolWt": mws}, quantiles=(0.05, 0.95), margin=0.0)
    assert env.bounds["mw_max"] < 1000.0


def test_lower_bound_never_goes_negative():
    env = derive_envelope({"MolWt": [10.0, 12.0, 14.0]}, quantiles=(0.05, 0.95), margin=5.0)
    assert env.bounds["mw_min"] >= 0.0


def test_provenance_records_what_it_was_derived_from():
    env = derive_envelope({"MolWt": [400.0, 450.0]}, n_actives=2, potency_cut=7.2,
                          potency_cut_how="quantile 0.75")
    p = env.provenance
    assert p["n_actives"] == 2 and p["potency_cut"] == 7.2
    assert "quantile" in p["potency_cut_how"]
    assert p["observed"]["MolWt"]["n"] == 2


def test_malformed_quantiles_and_margin_are_rejected():
    with pytest.raises(ValueError, match="0 <= lo < hi <= 1"):
        derive_envelope({"MolWt": [1.0]}, quantiles=(0.9, 0.1))
    with pytest.raises(ValueError, match="margin must be"):
        derive_envelope({"MolWt": [1.0]}, margin=-0.1)


def test_check_admits_reproduces_the_inherited_window_failure():
    """The measured failure: the flagship window rejects 5 of 7 bromodomain references, mostly on
    lipophilicity, because an acetyl-lysine pocket is hydrophobic."""
    inherited = {"mw_min": 300.0, "mw_max": 460.0, "logp_max": 3.5, "tpsa_max": 140.0}
    refs = {
        "jq1":         {"MolWt": 457.0, "MolLogP": 5.53, "TPSA": 69.4},
        "birabresib":  {"MolWt": 492.0, "MolLogP": 5.53, "TPSA": 92.4},
        "mivebresib":  {"MolWt": 459.5, "MolLogP": 4.37, "TPSA": 93.2},
        "pelabresib":  {"MolWt": 365.8, "MolLogP": 4.07, "TPSA": 81.5},
        "molibresib":  {"MolWt": 423.9, "MolLogP": 3.66, "TPSA": 81.4},
        "apabetalone": {"MolWt": 370.4, "MolLogP": 2.60, "TPSA": 93.7},
        "pfi_1":       {"MolWt": 347.4, "MolLogP": 2.47, "TPSA": 87.7},
    }
    r = check_admits(inherited, refs)
    assert not r["passed"]
    assert r["n_admitted"] == 2, "exactly the measured 2 of 7"
    assert "birabresib" in r["rejected"]


def test_check_admits_passes_a_window_fitted_to_those_references():
    wide = {"mw_min": 320.0, "mw_max": 620.0, "logp_max": 6.8, "tpsa_max": 145.0}
    refs = {"jq1": {"MolWt": 457.0, "MolLogP": 5.53, "TPSA": 69.4}}
    assert check_admits(wide, refs)["passed"]


# --- the lead-likeness ceiling must SLIDE with the target -------------------------------------------

_JAK_REFS = {
    "upadacitinib": {"MolWt": 380.4, "MolLogP": 2.10, "TPSA": 90.0},
    "tofacitinib":  {"MolWt": 312.4, "MolLogP": 1.10, "TPSA": 88.9},
    "baricitinib":  {"MolWt": 371.4, "MolLogP": 1.20, "TPSA": 128.9},
    "filgotinib":   {"MolWt": 425.5, "MolLogP": 2.91, "TPSA": 121.0},
    "abrocitinib":  {"MolWt": 323.4, "MolLogP": 1.30, "TPSA": 78.0},
}
_BET_REFS = {
    "jq1":         {"MolWt": 457.0, "MolLogP": 5.53, "TPSA": 69.4},
    "birabresib":  {"MolWt": 492.0, "MolLogP": 5.53, "TPSA": 92.4},
    "mivebresib":  {"MolWt": 459.5, "MolLogP": 4.37, "TPSA": 93.2},
    "pelabresib":  {"MolWt": 365.8, "MolLogP": 4.07, "TPSA": 81.5},
    "molibresib":  {"MolWt": 423.9, "MolLogP": 3.66, "TPSA": 81.4},
    "apabetalone": {"MolWt": 370.4, "MolLogP": 2.60, "TPSA": 93.7},
    "pfi_1":       {"MolWt": 347.4, "MolLogP": 2.47, "TPSA": 87.7},
}


def test_ceiling_slides_with_the_representative_compounds():
    """The point of the whole exercise: a fixed 520 Da ceiling is itself a borrowed constant. BET
    chemotypes are larger than kinase inhibitors, so the ceiling must move on its own."""
    from medchem.vls.envelope import derive_reference_ceiling

    jak = derive_reference_ceiling(_JAK_REFS)["bounds"]
    bet = derive_reference_ceiling(_BET_REFS)["bounds"]
    assert bet["mw_max"] > jak["mw_max"] + 50, "the BET ceiling must sit well above the kinase one"
    assert bet["logp_max"] > jak["logp_max"] + 2, "BET chemotypes are markedly more lipophilic"


def test_ceiling_reproduces_the_hand_tuned_windows():
    """Validation that the derivation recovers human judgement on BOTH targets rather than inventing
    new numbers. Hand-written was MW 300-460 for the kinase and 330-520 for the bromodomain.

    The assertion is AGREEMENT WITHIN 10%, not equality, and deliberately so: the margin inflates for
    small reference sets, so the derived window sits slightly wider than the hand-tuned one. Pinning
    exact numbers would make this test a snapshot of one margin setting -- it broke exactly that way
    when the inflation was added, which is the wrong kind of failure: the code was right and the test
    was encoding a superseded constant.
    """
    from medchem.vls.envelope import derive_reference_ceiling

    for refs, hand in ((_JAK_REFS, (300.0, 460.0)), (_BET_REFS, (330.0, 520.0))):
        b = derive_reference_ceiling(refs)["bounds"]
        for derived, expected in ((b["mw_min"], hand[0]), (b["mw_max"], hand[1])):
            assert abs(derived - expected) / expected < 0.10, (
                f"derived {derived:.0f} disagrees with human judgement {expected:.0f} by more than 10%"
            )


def test_inflated_margin_widens_rather_than_narrows_the_window():
    """The loosening must go outward on both sides -- more headroom, not a shifted window."""
    from medchem.vls.envelope import derive_reference_ceiling

    tight = derive_reference_ceiling(_BET_REFS, inflation=0.0)["bounds"]
    loose = derive_reference_ceiling(_BET_REFS, inflation=2.0)["bounds"]
    assert loose["mw_min"] < tight["mw_min"]
    assert loose["mw_max"] > tight["mw_max"]
    assert loose["logp_max"] > tight["logp_max"]


def test_ceiling_can_never_reject_its_own_references():
    """It is their range widened, so this holds by construction — and a regression here would mean the
    guard and the ceiling had come apart."""
    from medchem.vls.envelope import check_admits, derive_reference_ceiling

    for refs in (_JAK_REFS, _BET_REFS):
        bounds = derive_reference_ceiling(refs)["bounds"]
        assert check_admits(bounds, refs)["passed"]


def test_ceiling_covers_size_and_lipophilicity_only():
    """TPSA is a permeability constraint, not a lead-likeness one, and 7 references span it by accident:
    including it capped BRD4 at 97 against a defensible 140."""
    from medchem.vls.envelope import derive_reference_ceiling

    bounds = derive_reference_ceiling(_BET_REFS)["bounds"]
    assert "tpsa_max" not in bounds
    assert {"mw_min", "mw_max", "logp_max"} <= set(bounds)


def test_small_reference_sets_are_flagged_as_weakly_determined():
    from medchem.vls.envelope import derive_reference_ceiling

    out = derive_reference_ceiling({"only": {"MolWt": 400.0, "MolLogP": 3.0}})
    assert "caveat" in out and "weakly determined" in out["caveat"]
    assert "caveat" not in derive_reference_ceiling(_BET_REFS)


def test_intersection_takes_the_tighter_bound_from_each_source():
    """Three sources with distinct jobs: actives give chemotype plausibility (wide), references give
    lead-likeness (tighter), explicit config gives any hard cap the data cannot express."""
    from medchem.vls.envelope import intersect_bounds

    actives = {"mw_min": 300.0, "mw_max": 626.0, "logp_max": 6.8, "tpsa_max": 142.0}
    ceiling = {"mw_min": 326.0, "mw_max": 514.0, "logp_max": 5.99}
    out = intersect_bounds(actives, ceiling)
    assert out["mw_max"] == 514.0     # ceiling is tighter
    assert out["mw_min"] == 326.0     # minima take the HIGHER
    assert out["logp_max"] == 5.99
    assert out["tpsa_max"] == 142.0   # untouched by the ceiling


def test_empty_reference_set_yields_no_ceiling_rather_than_a_bogus_one():
    from medchem.vls.envelope import derive_reference_ceiling

    out = derive_reference_ceiling({})
    assert out["bounds"] == {} and "no sliding ceiling" in out["note"]
