"""Molecular scorer: turn the reward engine into per-molecule scores on REAL molecules.

Given trained potency + selectivity predictors and the training fingerprints (for the
applicability-domain term), compute each molecule's component vector — QSAR pIC50, direct-Δ
selectivity, RDKit MW/SlogP/TPSA/QED, and an AD distance — and reduce it to a scalar reward
via ``medchem.generative.scoring``.

Models are **injected** (the generative stage supplies the pipeline's own trained models),
so this stays pure, CPU-only, and unit-testable — no GPU, no sampler. The GPU generative
stage calls ``score_molecules`` as its reward function.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from medchem.features.featurize import _DEFAULT_DESCRIPTORS, compute_features
from medchem.generative.scoring import score_components
from medchem.reward_components import DESCRIPTOR_COMPONENTS

Predict = Callable[[np.ndarray], np.ndarray]


def _nn_tanimoto_max(fp: np.ndarray, train_fp: np.ndarray) -> np.ndarray:
    """Max Tanimoto of each fingerprint to any training fingerprint (bit vectors)."""
    inter = fp @ train_fp.T
    a = fp.sum(1, keepdims=True)
    b = train_fp.sum(1)[None, :]
    union = a + b - inter
    t = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return t.max(1)


def score_molecules(
    smiles: Sequence[str],
    *,
    potency_predict: Predict,
    selectivity_predict: Predict | None,
    train_fp: np.ndarray,
    spec: Iterable[Mapping],
    aggregation: str = "geometric_mean",
    n_bits: int = 2048,
    radius: int = 2,
    use_chirality: bool = True,
    descriptors: list[str] | None = None,
) -> list[dict]:
    """Score a list of SMILES. Unparseable molecules are dropped; returns one dict per kept
    molecule: ``{smiles, score, components, transformed}``. ``applicability_domain`` is the
    distance (1 − max-NN-Tanimoto) to the training set — larger = more out-of-domain.

    ``selectivity_predict`` is ``None`` when the optional ``selectivity`` stage is disabled. In
    that case no ``selectivity_delta`` component is emitted at all — a molecule must not be
    scored on a component nothing measured, and substituting a neutral value would be a
    fabricated number that silently changes every reward. A ``spec`` that still asks for
    ``selectivity_delta`` without a predictor is a configuration error and raises.
    """
    spec = list(spec)
    if selectivity_predict is None and any(c.get("name") == "selectivity_delta" for c in spec):
        raise ValueError(
            "spec requests the 'selectivity_delta' component but selectivity_predict is None "
            "(the selectivity stage is disabled). Drop the component from the scoring spec, or "
            "re-enable the stage — scoring it as a constant would invent a measurement."
        )
    # The featurisation has to match the one the loaded models were trained on, so every parameter
    # is threaded through from `features` config rather than defaulted here. `use_chirality` was not
    # a parameter at all: candidates were fingerprinted with chirality on regardless, and a
    # `use_chirality: false` run scored them with a model trained on achiral bits -- same shape, no
    # error, different molecules.
    names = list(_DEFAULT_DESCRIPTORS if descriptors is None else descriptors)
    x_fp, x_desc, _scaf, keep = compute_features(
        list(smiles), n_bits=n_bits, radius=radius, use_chirality=use_chirality, descriptors=names
    )
    kept = [s for s, k in zip(smiles, keep, strict=True) if k]
    if not kept:
        return []
    x = np.hstack([x_fp, x_desc])
    pic50 = np.asarray(potency_predict(x), dtype=float)
    seldelta = None if selectivity_predict is None else np.asarray(selectivity_predict(x), dtype=float)
    ad_far = 1.0 - _nn_tanimoto_max(x_fp.astype(np.float32), np.asarray(train_fp, np.float32))
    di = {n: i for i, n in enumerate(names)}
    # Four components are read straight out of the descriptor block, so a descriptor list that omits
    # one cannot produce it. Say which, here, instead of raising KeyError from a column lookup --
    # `descriptors: []` is a legitimate ablation and its consequence for the reward should be legible.
    wanted = {str(c.get("name")) for c in spec} & set(DESCRIPTOR_COMPONENTS)
    if absent := sorted(c for c in wanted if DESCRIPTOR_COMPONENTS[c] not in di):
        raise ValueError(
            f"the scoring spec requests {absent}, which come from descriptor(s) "
            f"{[DESCRIPTOR_COMPONENTS[c] for c in absent]} that features.descriptors does not "
            f"include (configured: {names or 'none'}). Add the descriptor, or drop the component."
        )

    out: list[dict] = []
    for i, s in enumerate(kept):
        comps = {
            "qsar_pic50": float(pic50[i]),
            **({} if seldelta is None else {"selectivity_delta": float(seldelta[i])}),
            **{c: float(x_desc[i, di[d]]) for c, d in DESCRIPTOR_COMPONENTS.items() if d in di},
            "applicability_domain": float(ad_far[i]),
        }
        agg, per = score_components(comps, spec, aggregation=aggregation)
        out.append({"smiles": s, "score": agg, "components": comps, "transformed": per})
    return out
