"""Assay-driven outer loop — INTERFACE ONLY, deliberately not enabled.

Each round would: generate candidates -> score them -> pick a batch by an acquisition rule (exploit by
reward, or explore by applicability-domain uncertainty) -> obtain MEASURED activity for that batch ->
retrain the surrogate on those measurements -> repeat.

WHY IT IS NOT ENABLED, and why the earlier framing was wrong.

A structural or ligand-based model does not produce labels. It produces hypotheses. An outer loop that
retrains a surrogate on another model's predictions is not active learning -- it is a model learning to
agree with itself. The reported metric improves while the shared bias compounds, which is the most
flattering possible failure mode and the hardest to notice from inside the loop.

``retrain`` is therefore the hook where *measured assay data* refits the surrogate. Until such data
exists, the pipeline stops at candidate nomination and hands off. See configs/generative/jak1/README.md for the
scope decision.

The mock Sampler and Scorer below make the control flow testable on CPU without implying that a
computational score is evidence.
"""


from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from medchem.generative.interfaces import ComputationalScorer, Sampler
from medchem.generative.scorer import Predict, score_molecules


def _acquire(scored: list[dict], k: int, mode: str) -> list[dict]:
    """Pick k candidates: 'score' exploits (highest reward); 'uncertainty' explores (farthest
    out-of-domain, i.e. highest AD distance — the conformal/AD acquisition signal)."""
    if mode == "uncertainty":
        key = lambda r: r["components"]["applicability_domain"]  # noqa: E731
    else:
        key = lambda r: r["score"]  # noqa: E731
    return sorted(scored, key=key, reverse=True)[:k]


def active_learning_loop(
    sampler: Sampler,
    scorer: ComputationalScorer,
    *,
    rounds: int,
    potency_predict: Predict,
    selectivity_predict: Predict | None,
    train_fp,
    spec: Iterable[Mapping],
    n_per_round: int = 100,
    batch_size: int = 10,
    acquisition: str = "score",
    retrain: Callable | None = None,
    seed_smiles: str | None = None,
    **score_kw,
) -> dict:
    """Run ``rounds`` of generate → score → acquire → scorer-validate → (retrain). Returns per-round
    records + a convergence trace (mean scorer score of the validated batch per round)."""
    pot, sel = potency_predict, selectivity_predict
    history: list[dict] = []
    for r in range(rounds):
        cands = sampler.sample(n_per_round, seed_smiles=seed_smiles)
        scored = score_molecules(
            cands, potency_predict=pot, selectivity_predict=sel,
            train_fp=train_fp, spec=spec, aggregation="geometric_mean", **score_kw,
        )
        batch = _acquire(scored, batch_size, acquisition)
        validated = [{**b, "scorer": scorer.score(b["smiles"])} for b in batch]
        mean_scorer = (
            sum(v["scorer"].get("structure_score", 0.0) for v in validated) / len(validated)
            if validated else 0.0
        )
        history.append({
            "round": r,
            "n_generated": len(cands),
            "n_validated": len(validated),
            "mean_reward": (sum(b["score"] for b in batch) / len(batch)) if batch else 0.0,
            "mean_scorer_score": mean_scorer,
            "batch": [
                {"smiles": v["smiles"], "reward": v["score"], "scorer": v["scorer"]} for v in validated
            ],
        })
        if retrain is not None:  # refit the surrogate on the newly-validated labels (real = GPU scorer)
            pot, sel = retrain(validated, pot, sel)
    return {"rounds": rounds, "history": history, "trace": [h["mean_scorer_score"] for h in history]}
