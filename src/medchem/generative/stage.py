"""Generative design stage: sample → score (naive vs constrained) → select.

Back-half stage wired into the DAG. The MOCK sampler (default) replays curated JAK1 SMILES as a
candidate pool so the stage runs on CPU and proves the wiring; the real REINVENT4 sampler
(ADR 0003, GPU) drops in behind the ``Sampler`` interface via ``generative.sampler: reinvent4``.
Loads the persisted potency + selectivity models from the qsar/selectivity stages and scores
under both the naive sum and the constrained geometric-mean rewards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from medchem.generative.design import generate_and_select
from medchem.generative.samplers import MockSampler
from medchem.pipeline.stage import StageContext, StageResult, stage

# NO MODULE-LEVEL DEFAULT SPEC. There was one -- seven components mirroring configs/jak1.yaml, with
# JAK1's centres baked in (qsar_pic50 at 7.5, an MW window of 300-450) -- and the stage resolved its
# spec as `gen.scoring.components or _SPEC`. Two ways that went wrong, and the second is the reason it
# is deleted rather than merely bypassed:
#
#   * a config that OMITTED components got a kinase's reward function. On a bromodomain that is a
#     different scientific objective, applied silently: JQ1 sits near 7.1, so a centre of 7.5 rewards
#     the wrong end of the range, and BET chemotypes routinely exceed 450 Da.
#   * a config that asked for `components: []` -- no reward -- got the same seven. An explicit request
#     for nothing became the most target-specific default in the file.
#
# A default reward is not a safe fallback the way a default thread count is. It is a scientific claim
# about what "better" means for this target, so there is no value the code can supply on the config's
# behalf. The stage fails closed instead, and this comment stays where the constant used to be so the
# next person does not helpfully add it back.


@stage("discovery", "generative", deps=("curate", "featurize", "qsar"), optional_deps=("selectivity",),
       config_keys=("generative", "features", "seed"))
def generative(ctx: StageContext) -> StageResult:
    """Score a sampler's candidates under both rewards and record each arm's selection."""
    import joblib

    # These sections are typed models, so read them by attribute. Dict-punning a modeled section
    # (`cfg.get("scoring", {}).get(...)`) breaks the moment a subsection becomes a submodel, and it
    # breaks at runtime rather than under the type checker -- which is how it broke once already.
    feat = ctx.config.features
    n_bits, radius = feat.n_bits, feat.radius
    # Chirality and the descriptor block were NOT passed on, so candidates were featurized with
    # `use_chirality=True` and the nine standard descriptors whatever the config said, and then
    # scored by a model trained on the configured featurization. Two ways for that to go wrong:
    # a different descriptor LIST gives a different column count and the model raises somewhere
    # inside sklearn, and `use_chirality: false` gives the same shape with different bits -- no
    # error at all, just predictions from a model fed a matrix it was not trained on.
    chir = feat.use_chirality
    descriptors = None if feat.descriptors is None else [str(d) for d in feat.descriptors]
    gen = ctx.config.generative
    top_k, n_cand, sampler_name = gen.top_k, gen.n_candidates, gen.sampler

    # The scoring spec is REQUIRED at runtime, and the two absences are reported differently because
    # they are different mistakes: one config forgot to say, the other said "nothing".
    if not gen.scoring.components:
        omitted = gen.scoring.components is None
        raise ValueError(
            f"the generative stage was invoked for target {ctx.config.target!r} with "
            + ("no generative.scoring.components at all"
               if omitted else "generative.scoring.components: [] -- an empty specification")
            + ". A reward specification is a scientific claim about what counts as a better molecule "
              "for THIS target, so there is no default this stage can supply: it previously injected a "
              "seven-component spec carrying JAK1's potency centre and molecular-weight window, which "
              "for another target silently optimises the wrong objective. Declare the components in "
              "the config, or disable the stage with disable_stages: [generative]."
        )
    spec = [dict(c) for c in gen.scoring.components]

    potency = joblib.load(ctx.upstream["qsar"].outputs["model"])
    train_fp = np.load(ctx.upstream["featurize"].outputs["features"])["X"][:, :n_bits]
    # Selectivity is OPTIONAL. When absent, its scoring component is dropped from the spec rather
    # than scored as zero -- a missing objective must not look like a failed one, which under a
    # geometric mean would drive every candidate's score to nothing.
    sel_up = ctx.upstream.get("selectivity")
    selectivity_predict = None
    if sel_up is not None and "model" in sel_up.outputs:
        selectivity_predict = joblib.load(sel_up.outputs["model"]).predict

    primary_df = pd.read_csv(ctx.upstream["curate"].outputs["potency_training"])
    pool = primary_df["canonical_smiles"].dropna().tolist()[:n_cand]

    if sampler_name == "mock":
        sampler = MockSampler(pool)
    else:
        from medchem.generative.samplers import Reinvent4Sampler

        sampler = Reinvent4Sampler(seed_core=gen.seed_core)

    if selectivity_predict is None:
        spec = [c for c in spec if c.get("name") != "selectivity_delta"]

    res = generate_and_select(
        sampler, n=n_cand,
        potency_predict=potency.predict, selectivity_predict=selectivity_predict,
        train_fp=train_fp, spec=spec, top_k=top_k, n_bits=n_bits, radius=radius,
        use_chirality=chir, descriptors=descriptors,
    )

    metrics = {
        "sampler": sampler_name,
        "n_candidates": res["n_candidates"],
        "top_k": top_k,
        "constrained_mean_ad": float(np.mean(res["constrained_top_ad"])) if res["constrained_top_ad"] else None,
        "naive_mean_ad": float(np.mean(res["naive_top_ad"])) if res["naive_top_ad"] else None,
        "note": "MOCK sampler replays curated compounds (wiring proof). Real reward-hacking "
                "evidence needs the REINVENT4 GPU sampler generating novel molecules.",
    }
    out = Path(ctx.workdir)
    (out / "generative_selection.json").write_text(
        json.dumps(
            {**metrics,
             "constrained_top": [r["smiles"] for r in res["constrained_top"]],
             "naive_top": [r["smiles"] for r in res["naive_top"]]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return StageResult(
        name="generative",
        outputs={"selection": str(out / "generative_selection.json")},
        metrics=metrics,
    )
