"""Generative design orchestration: sample → score → select, with the two-arm
naive-vs-constrained comparison (ADR 0003) made runnable.

``generate_and_select`` scores a sampler's candidates under BOTH the naive ``sum`` reward and
the constrained ``geometric_mean`` reward, and returns each arm's top-k. The point (the
reward-hacking → recovery story): the sum arm can be gamed by an out-of-domain molecule that
maxes potency/selectivity while failing the applicability-domain term; the product arm forces
all objectives at once and pushes it out. Pure + CPU-testable via an injected Sampler + model
predictors — the GPU sampler/scorer drop in behind the interfaces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from medchem.generative.interfaces import Sampler
from medchem.generative.scorer import Predict, score_molecules


def generate_and_select(
    sampler: Sampler,
    *,
    n: int,
    potency_predict: Predict,
    selectivity_predict: Predict | None,
    train_fp,
    spec: Iterable[Mapping],
    top_k: int = 10,
    seed_smiles: str | None = None,
    n_bits: int = 2048,
    radius: int = 2,
    use_chirality: bool = True,
    descriptors: list[str] | None = None,
) -> dict:
    """Sample n candidates, score under both rewards, return each arm's top-k + AD profile.

    The featurisation parameters are NAMED here rather than collected in a ``**kwargs`` bag. They used
    to be, under a docstring one line below asserting the opposite -- and the bag is what let the
    generative stage omit ``use_chirality`` and the descriptor list for as long as it did: adding a
    parameter to ``score_molecules`` produced no signal at this boundary, and a misspelling at a call
    site would travel one frame further before failing.
    """
    candidates = sampler.sample(n, seed_smiles=seed_smiles)

    def _ranked(aggregation: str) -> list[dict]:
        """Both arms score the SAME candidates with the SAME inputs; only the reduction differs.

        Every argument is passed explicitly, so a mismatch against ``score_molecules`` is a type error
        the checker reports here rather than a TypeError raised a frame deeper at runtime."""
        return sorted(
            score_molecules(
                candidates,
                aggregation=aggregation,
                potency_predict=potency_predict,
                selectivity_predict=selectivity_predict,
                train_fp=train_fp,
                spec=spec,
                n_bits=n_bits,
                radius=radius,
                use_chirality=use_chirality,
                descriptors=descriptors,
            ),
            key=lambda r: r["score"], reverse=True,
        )

    constrained = _ranked("geometric_mean")
    naive = _ranked("sum")

    def _ad(rows: list[dict]) -> list[float]:
        return [r["components"]["applicability_domain"] for r in rows[:top_k]]

    return {
        "n_candidates": len(candidates),
        "constrained_top": constrained[:top_k],  # geometric-mean selection (what we ship)
        "naive_top": naive[:top_k],              # sum-reward selection (the gameable baseline)
        "constrained_top_ad": _ad(constrained),  # AD distances of the two selections — the
        "naive_top_ad": _ad(naive),              #   constrained arm should stay far more in-domain
    }
