"""Molecular scorer wiring: real component computation + the applicability-domain penalty."""

from __future__ import annotations

import numpy as np

from medchem.features.featurize import compute_features
from medchem.generative.scorer import score_molecules

_SPEC = [
    {"name": "qsar_pic50", "transform": "sigmoid", "center": 7.5},
    {"name": "selectivity_delta", "transform": "sigmoid", "center": 1.0},
    {"name": "applicability_domain", "transform": "reverse_sigmoid", "center": 0.4, "k": 10.0},
]


def test_scorer_penalizes_out_of_domain_molecules():
    # "training" chemistry: a few aromatic, drug-like molecules
    train = ["c1ccccc1", "c1ccncc1", "CC(=O)Nc1ccccc1", "c1ccc(cc1)C(=O)O"]
    tfp, *_ = compute_features(train)

    # constant, favourable potency + selectivity so the ONLY differentiator is the AD term
    pot = lambda x: np.full(len(x), 8.0)  # noqa: E731
    sel = lambda x: np.full(len(x), 1.5)  # noqa: E731

    # in-domain (a training member, NN-Tanimoto 1.0) vs out-of-domain (a long aliphatic chain)
    res = score_molecules(
        [train[0], "CCCCCCCC"],
        potency_predict=pot, selectivity_predict=sel, train_fp=tfp, spec=_SPEC,
    )
    by = {r["smiles"]: r for r in res}

    in_dom, ood = by[train[0]], by["CCCCCCCC"]
    # AD distance: ~0 in-domain, large out-of-domain
    assert in_dom["components"]["applicability_domain"] < 0.1
    assert ood["components"]["applicability_domain"] > 0.6
    # under the geometric mean, the out-of-domain molecule is tanked despite good potency/selectivity
    assert in_dom["score"] > ood["score"]
    assert ood["score"] < 0.35
    # real physchem was computed from RDKit, not mocked (benzene 78.1, octane 114.2 Da)
    assert abs(in_dom["components"]["mw"] - 78.1) < 1.0
    assert abs(ood["components"]["mw"] - 114.2) < 1.0
    assert all(0.0 <= r["score"] <= 1.0 for r in res)


def test_scorer_drops_unparseable_and_keeps_order():
    train = ["c1ccccc1"]
    tfp, *_ = compute_features(train)
    res = score_molecules(
        ["c1ccccc1", "not_a_smiles", "CCO"],
        potency_predict=lambda x: np.full(len(x), 7.0),
        selectivity_predict=lambda x: np.full(len(x), 1.0),
        train_fp=tfp, spec=_SPEC,
    )
    assert [r["smiles"] for r in res] == ["c1ccccc1", "CCO"]  # bad SMILES dropped, order kept


def test_scoring_without_a_selectivity_model_omits_the_component_rather_than_crashing():
    """The regression for a real crash. ``disable_stages: [selectivity]`` composes at the graph
    level, but the generative stage then passes ``selectivity_predict=None`` into the scorer, which
    called it unconditionally -- a TypeError once the stage actually ran. Graph-level composition
    tests could not see it because they never execute the stage body.

    The correct behaviour is to emit NO selectivity component. A neutral stand-in value would be a
    fabricated measurement that silently shifts every reward.
    """
    train = ["c1ccccc1", "c1ccncc1", "CC(=O)Nc1ccccc1"]
    tfp, *_ = compute_features(train)
    spec_no_sel = [c for c in _SPEC if c["name"] != "selectivity_delta"]

    res = score_molecules(
        [train[0], "CCCCCCCC"],
        potency_predict=lambda x: np.full(len(x), 8.0),
        selectivity_predict=None,
        train_fp=tfp, spec=spec_no_sel,
    )

    assert len(res) == 2
    for r in res:
        assert "selectivity_delta" not in r["components"]
        assert "selectivity_delta" not in r["transformed"]
        assert "qsar_pic50" in r["components"]          # everything else still scored
        assert 0.0 <= r["score"] <= 1.0
    # the AD term still discriminates, so the reward is intact minus one component
    by = {r["smiles"]: r["score"] for r in res}
    assert by[train[0]] > by["CCCCCCCC"]


def test_requesting_selectivity_without_a_model_is_a_loud_error():
    """A spec that still asks for the component while no model was supplied is a configuration
    mistake. It must raise rather than score the component as a constant."""
    import pytest

    tfp, *_ = compute_features(["c1ccccc1", "c1ccncc1"])
    with pytest.raises(ValueError, match="selectivity_delta"):
        score_molecules(
            ["c1ccccc1"],
            potency_predict=lambda x: np.full(len(x), 8.0),
            selectivity_predict=None,
            train_fp=tfp, spec=_SPEC,          # _SPEC still contains selectivity_delta
        )
