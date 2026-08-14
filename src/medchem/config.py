"""Typed configuration for medchem pipelines.

A run is fully described by one YAML config (``configs/<target>.yaml``). Pointing the pipeline at a
new target is a config change, not a code change — that is the whole reusability thesis, so config
sections must never be hardcoded in stages.

**Why some sections forbid unknown keys and others allow them.**

``extra="allow"`` is the right default for a section whose stage is not written yet: a
forward-looking key validates today and gains structure later. It is the *wrong* default for a
section a stage actually reads, because there the failure is silent. ``primary`` misspelled as
``primry`` under ``extra="allow"`` validates, is preserved in ``model_dump()``, contributes to the
cache key, and then the stage falls back to a default — producing a complete, plausible, wrong run
with no error anywhere. This project has already been bitten by exactly that shape of failure in
other guises (see docs/PITFALLS.md).

The split below is not a guess about what is "implemented"; it follows an AST audit of every config
key that ``src/medchem`` actually reads.

**Strict (``extra="forbid"``) — keys read by package code, where a typo would silently fall back to
a default:** ``data``, ``features``, ``eval``, ``model`` (both ``potency`` and ``selectivity``),
``generative``, and inside ``vls`` the three subsections the stage consumes: ``library`` (with
``lead_like``), ``tier1``, and ``actives``. ``vls.tier1`` matters most — those are the applicability-
domain and prioritisation thresholds, so a mistyped key there changes which compounds are called
in-domain while the run still reports success.

**Permissive, deliberately, and why each one. The list is now short, and that is the point:**

* ``vls.tiers``, ``vls.budget_allocation``, ``vls.validation``, ``vls.prospective`` — consumed by the
  out-of-package execution scripts, not by the DAG. Their shape is still moving (the free-energy tier
  was retired outright), and forbidding keys here would reject a legitimate campaign edit. Permissive
  in their CONTENTS only: an unknown key at the ``vls`` level itself is rejected, because ``tier_1:``
  beside ``tier1:`` left the real applicability-domain thresholds at their defaults.
* ``vls.known_reference`` — a free-form ``name -> SMILES`` mapping. The keys are compound names, so
  there is no fixed key set to enforce.

**The ROOT is strict too, as of the fail-closed pass.** It was the last permissive surface and the worst
one to leave open: every top-level section is modeled, so an unrecognised name there does not extend
anything — it sits beside a real section, and that section then runs entirely on its defaults.
``featuers:`` validated, reached the cache key, and the run featurised with 2048 chiral bits and nine
descriptors while the config on screen said otherwise.

(``loop`` and ``structure`` were the reason the root stayed open. ``structure`` has been strict since the
receptor stage was written; ``loop`` was **deleted from the configs** rather than left as decoration —
an AST audit found zero of its four keys read anywhere. ``generative/active_learning.py`` keeps the
interface it describes, documented as interface-only; a config section implying a feature runs is worse
than no section. A genuinely new section now costs one line here, which is cheaper than a silent typo.)

Cross-section invariants are checked too, because a per-field type cannot catch them: the primary
target must exist in ``data.targets``, and every ``model.selectivity.pairs`` entry must name two
configured targets. Both are the kind of mistake a second target invites, and both would otherwise
surface as an empty result set rather than as an error.

**Removed rather than modeled.** The same audit found six keys that looked like knobs and were read
nowhere: ``model.potency.algorithms``, ``model.potency.calibration``,
``generative.scoring.aggregation``, ``vls.library.lead_like.derive.potency_quantile``,
``generative.stages`` and ``generative.scoring.diversity_filter``. RF and XGBoost both always train,
conformal calibration always runs, both aggregations are always computed for the reward-hacking
comparison, and the envelope's actives are selected by ``vls.actives``. Keeping them would mean
accepting ``algorithms: [random_forest]`` and training XGBoost anyway.

The last two were defended as *records* of the external generative campaign rather than as knobs, which
is a real category — but they were not accurate records. ``diversity_filter`` said
``scaffold_similarity`` while the campaign config it described,
``configs/generative/jak1/armB_v2_rl.toml``, specifies ``IdenticalMurckoScaffold``. Those TOMLs ship and
are what the external scripts read, so they are the record; a hand-maintained second copy is a thing
that drifts, and it had.

**Omitted is not the same request as empty.** Three fields distinguish ``None`` from an explicit
empty value, because for each of them the empty form is a real experiment somebody would want to run:
``features.descriptors: []`` (fingerprint bits only), ``model.selectivity.pairs: []`` (no selectivity
modelling), and ``vls.tier1.conformal_halfwidth: 0.0`` (no uncertainty band). Each had a falsy
default and a consumer written as ``configured or <fallback>``, so each explicit empty request
silently produced the fallback: the full descriptor block, the whole derived pair panel, and this
run's measured half-width. The ablation would have reported the unablated number, which is worse than
refusing to run it.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

STRICT = ConfigDict(extra="forbid")


class DemoConfig(BaseModel):
    """Parameters for the built-in ``demo`` pipeline used by tests/CI smoke."""

    model_config = STRICT

    n: int = 5


class AssayCohortConfig(BaseModel):
    """Which assays constitute "the target", by regex over the assay description.

    A single-protein ChEMBL target is not one assay. BRD4's IC50 records are 47.7% BD1-explicit,
    25.0% biochemical with no domain stated, 15.8% cell-based and 6.3% BD2-explicit, and the domains
    measure different things — apabetalone is 5.85 on BD1 and 6.88 on BD2, so a median across both
    represents neither. (Derived by scripts/derive_composition.py from the published inputs.)

    Without an explicit cohort the resulting model is a "target-associated IC50" model, which is a
    weaker and different claim from a domain-specific one, and any structural arm built on a
    domain-specific crystal form is then a separate hypothesis rather than the same one.

    ``label`` is required when filtering: it names the cohort in every artifact, so a plot cannot be
    captioned with a domain the data does not support.
    """

    model_config = STRICT

    # A cohort is NAMED from the frozen spec in medchem.data.cohort, not spelled out per config.
    # Per-config regexes would make every run's cohort a slightly different thing and unreproducible
    # from the manifest; the spec carries a version so a rule change is visible as a version change.
    name: str = "target_associated"
    label: str = ""                       # human label for artifacts; defaults to the cohort name

    @model_validator(mode="after")
    def _cohort_must_be_frozen(self) -> AssayCohortConfig:
        from medchem.cohorts import COHORT_ALIASES, FROZEN_COHORTS, resolve_cohort

        # Accept deprecated spec-1.0 names so configs and manifests written before the 1.1 rename still
        # load. The name is NOT rewritten here: the cohort string feeds this stage's cache key, so
        # silently canonicalising it would change the key of an already-executed run.
        if resolve_cohort(self.name) not in FROZEN_COHORTS:
            raise ValueError(
                f"data.assay_cohort.name={self.name!r} is not a frozen cohort. Known: "
                f"{sorted(FROZEN_COHORTS)} (deprecated aliases: {sorted(COHORT_ALIASES)}). Cohorts are "
                f"versioned so a run is reproducible from its manifest rather than from a regex that "
                f"may since have changed."
            )
        return self


class CurationConfig(BaseModel):
    """How raw measurements become labels. SEPARATE from acquisition, deliberately.

    ``data`` is hashed into the ``data_pull`` cache key, so anything living there changes what gets
    *fetched*. Cohort selection and the temporal cutoff change only how already-fetched rows are
    interpreted — putting them in ``data`` made a headline and a sensitivity config produce different
    pull keys, which would have triggered four live ChEMBL pulls instead of two and left the two
    analyses consuming different snapshots. The whole point of a sensitivity run is that it shares the
    headline's raw bytes.
    """

    model_config = STRICT

    assay_cohort: AssayCohortConfig | None = None
    # Measurements at or after this year are held out of TRAINING labels. Curation must know the
    # cutoff: it deduplicates by median, so a label built from all years is informed by post-cutoff
    # data even when the compound is assigned to train by its earliest year.
    # REQUIRED, and required to equal eval.temporal_cutoff_year (see Config._temporal_cutoffs_agree).
    # Temporal evaluation is not optional in this pipeline: the harness always computes a temporal split
    # and every published panel reports one, so a curation cutoff of None would build no era-split label
    # for a split that runs regardless -- the exact mismatch that lets a post-cutoff measurement inform a
    # training label.
    temporal_cutoff_year: int = 2022


class DataConfig(BaseModel):
    """Data ACQUISITION parameters only — hashed into the pull's cache key.

    Nothing here may depend on how the data is later interpreted, or two analyses of one snapshot
    become two snapshots.
    """

    model_config = STRICT


    chembl_release: str = "34"
    targets: dict[str, str] = Field(default_factory=dict)  # name -> ChEMBL target id
    primary: str | None = None  # which target is modeled; others become comparators
    # Omission keeps the documented default. An EMPTY list is rejected by the validator below rather
    # than defaulted, because unlike the other empty-request fields there is no experiment it expresses:
    # accepting no activity type at all curates zero measurements, and the consumer answered it with
    # `configured or ["IC50"]` -- so a config asking for nothing silently got IC50.
    activity_types: list[str] = Field(default_factory=lambda: ["IC50"])
    # Literal, not str: the conversion to pIC50 is fixed in medchem.data.curate, so any other value was
    # accepted and then ignored. A knob that silently does nothing is worse than an absent one -- narrowing
    # the type turns the field from undocumented decoration into an enforced statement of what happens.
    standardize_units_to: Literal["pIC50"] = "pIC50"   # from standard_units; never trusted from a label
    # Only "median" is implemented: curate aggregates replicate measurements by median. The other three
    # were declared, validated, and then never read.
    dedup: Literal["median"] = "median"
    # NOTE: QSAR trains on the FULL quality-filtered pIC50 range for coverage; there is
    # deliberately no pchembl_min gate. Drug-likeness is applied downstream at candidate
    # triage, not to the training set.

    @model_validator(mode="after")
    def _activity_types_must_not_be_empty(self) -> DataConfig:
        """``activity_types: []`` selects no measurement at all, so it cannot be honoured or defaulted.

        Rejected here rather than in the consumer, and rejected rather than filled in: curate read it as
        ``configured or ["IC50"]``, so the one request that cannot work quietly became the default. It is
        also not a meaningful ablation -- an empty accept-list curates an empty training set, which the
        downstream stages would then report on.
        """
        if not self.activity_types:
            raise ValueError(
                "data.activity_types is empty. It selects which measured endpoints curation accepts, so "
                "an empty list admits nothing and produces an empty training set. Omit the key to take "
                "the documented default ['IC50'], or name the endpoints you want."
            )
        return self

    @model_validator(mode="after")
    def _primary_must_be_a_configured_target(self) -> DataConfig:
        if self.primary is not None and self.targets and self.primary not in self.targets:
            raise ValueError(
                f"data.primary={self.primary!r} is not a key of data.targets "
                f"({sorted(self.targets)}). The primary target must be one of the targets pulled, "
                f"or curation would train on an empty selection."
            )
        return self

    @property
    def comparators(self) -> list[str]:
        """Targets other than the primary, in config order — the selectivity denominators."""
        return [t for t in self.targets if t != self.primary]


class FeaturesConfig(BaseModel):
    """Featurisation — read by ``featurize`` and every stage that must match it exactly."""

    model_config = STRICT

    # Morgan/ECFP4 is fixed in medchem.features.featurize. The GEOMETRY keys below (radius, n_bits,
    # use_chirality, descriptors) ARE honoured; only the family selector was inert.
    fingerprint: Literal["ecfp4"] = "ecfp4"
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True
    # None and [] are DIFFERENT REQUESTS, and the default must be None to keep them apart.
    #
    # None means "the project's standard descriptor block" (medchem.features.featurize's
    # _DEFAULT_DESCRIPTORS, nine of them). ``descriptors: []`` means "fingerprint bits only, no
    # descriptor columns" -- a real ablation, and the cheapest way to ask whether the descriptor
    # block earns its place.
    #
    # With a ``[]`` default the two were indistinguishable, and every consumer wrote
    # ``configured or DEFAULTS``, so an explicit empty list silently trained on nine descriptor
    # columns the config asked to remove. The ablation would have reported the unablated number.
    descriptors: list[str] | None = None


class SelectivityModelConfig(BaseModel):
    """Direct-Δ selectivity model — read by the ``selectivity`` stage."""

    model_config = STRICT

    # The direct-delta formulation is fixed in medchem.models.selectivity: it predicts the isoform Δ
    # directly rather than subtracting two potency models. No alternative is implemented.
    mode: Literal["direct_delta"] = "direct_delta"
    # None and [] are DIFFERENT REQUESTS, like features.descriptors above.
    #
    # None means "derive the pairs from data.targets": every primary-comparator combination. ``[]``
    # means "model no pairs" -- the stage runs, produces the potency-only outputs, and reports no
    # selectivity, which is what a single-target config wants.
    #
    # The default was [], the stage read ``pairs or <derived>``, and so an explicit empty list was
    # overwritten by the full derived panel. A config asking for no selectivity got all of it.
    pairs: list[str] | None = None  # "PRIMARY-COMPARATOR"
    delta_threshold: float = 1.0
    rf_n_estimators: int = 400


class XgbConfig(BaseModel):
    """The XGBoost potency model's hyperparameters — exactly the ones ``qsar`` forwards.

    Strict, and strict about a narrower thing than it looks: these are not "XGBRegressor's
    parameters". ``medchem.models.qsar`` forwards five named values and nothing else, so a key that
    XGBoost itself accepts -- ``min_child_weight``, ``reg_lambda``, ``gamma`` -- is still dropped on
    the floor here. As a free-form dict this section accepted all of them, contributed them to the
    cache key, and trained a model that had never seen them; the run then reported a tuned model it
    had not fitted. Modelling the five that are real turns "quietly ignored" into "rejected at load".

    Adding a parameter means adding a field here AND forwarding it in ``qsar`` -- the strictness is
    what keeps those two in step.
    """

    model_config = STRICT

    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class PotencyModelConfig(BaseModel):
    """Potency estimators — read by the ``qsar`` stage.

    These are the POTENCY model's settings. ``medchem.models.selectivity`` fits its own XGBoost as a
    fixed comparison baseline (see ``_XGB_BASELINE`` there) and deliberately does not read this
    section: the point of that baseline is to be the same estimator every time so the direct-Δ model
    is compared against a constant, and a config able to tune it could tune its own comparator.
    """

    model_config = STRICT

    rf_n_estimators: int = 400
    xgb: XgbConfig = Field(default_factory=XgbConfig)


class ModelConfig(BaseModel):
    """Model hyperparameters. Strict: every key here is read by the qsar or selectivity stage."""

    model_config = STRICT

    potency: PotencyModelConfig = Field(default_factory=PotencyModelConfig)
    selectivity: SelectivityModelConfig = Field(default_factory=SelectivityModelConfig)


class EvalGates(BaseModel):
    """Hard pass/fail thresholds for the evaluation report."""

    model_config = STRICT

    scaffold_cv_r2_min: float = 0.55
    y_scramble_r2_max: float = 0.10
    # Optional hard gate on exact duplicate pairs. The harness implemented it as `if
    # "leakage_max_exact_dupes" in gates`, testing membership against what it assumed was a dict --
    # but this section is strict and had no such field, so no config could set it and the branch was
    # unreachable. Declared here so the capability the harness offers is one a config can request.
    # None means "report leakage without gating on it", which is what the frozen panels do: on
    # analog-dense data random-split near-duplicates are unavoidable and gating on them is noise.
    leakage_max_exact_dupes: int | None = None


class EvalConfig(BaseModel):
    """Evaluation protocol — read by the ``evaluate`` stage."""

    model_config = STRICT

    # The harness computes exactly these three splits unconditionally; the field could not select a
    # subset. Constrained so a config asking for something else fails instead of being ignored.
    # EXACTLY these three, in this order. The harness computes all three unconditionally and reports
    # them all, so this field cannot select a subset, reorder the work, or ask for one twice -- it
    # records what happens. Constrained rather than removed because the configs state it and a reader
    # should be able to see the split set without reading the harness.
    splits: list[Literal["random", "scaffold", "temporal"]] = Field(
        default_factory=lambda: ["random", "scaffold", "temporal"]
    )
    temporal_cutoff_year: int = 2022
    gates: EvalGates = Field(default_factory=EvalGates)


class GenerativeScoringConfig(BaseModel):
    """Reward specification for the in-package generative stage.

    ``components`` is a list of free-form mappings by design — each entry names a component and its
    transform, and the transform's own parameters (``center``, ``low``, ``high``, ``k``) vary by
    transform type.

    ``diversity_filter`` was here, read by nothing, and documented as "a record of the campaign's
    shape". It was not an accurate record: it said ``scaffold_similarity`` while the campaign config it
    claimed to describe -- ``configs/generative/jak1/armB_v2_rl.toml``, which SHIPS -- specifies
    ``IdenticalMurckoScaffold`` with a bucket size and three thresholds. A hand-maintained second copy
    of a setting whose real home is a tracked file is a thing that drifts, and it had. Removed: the
    TOMLs are the record, and being STRICT this section now rejects a config that still sets it.
    """

    model_config = STRICT

    # None and [] are DIFFERENT REQUESTS, like features.descriptors and model.selectivity.pairs.
    #
    # None means "no scoring specification given". [] means "a specification with no components", which
    # is not a reward at all. NEITHER may be answered with a default: the only default that ever existed
    # was a JAK1-shaped seven-component spec hard-coded in the generative stage, and injecting it for a
    # bromodomain -- or for any target whose config asked for nothing -- silently scored candidates
    # against a kinase's centres (qsar_pic50 at 7.5, MW window 300-450). The stage now refuses to run
    # without an explicit non-empty spec, and this field records which of the two absences it was.
    components: list[dict] | None = None

    @model_validator(mode="after")
    def _components_must_be_executable(self) -> GenerativeScoringConfig:
        """Validate each component against the real transform signature.

        The shipped config declared ``transform: sigmoid`` with no ``center`` for three components.
        ``sigmoid`` has no default centre, so that spec could never run — and nobody found out,
        because the stage used a hard-coded default spec and ignored the configured one. The moment
        the stage started honouring config, the run died with
        ``sigmoid() missing 1 required positional argument: 'center'``.

        Signatures are introspected rather than duplicated here, so adding a required parameter to a
        transform automatically invalidates configs that omit it.

        Component NAMES are checked the same way, against the set the scorer emits. A name the scorer
        does not produce is the quietest failure available here. ``score_components`` scores a missing
        component 0.0 and ``aggregate`` floors it at 1e-9, so the reward does not collapse -- instead
        that objective becomes a constant factor shared by every candidate, the ranking is decided by
        the remaining components, and the report still lists the objective as scored. Measured: the
        ranking under a misspelled name is identical to the ranking with the component deleted.
        """
        import inspect

        from medchem.reward_components import SCORED_COMPONENTS
        from medchem.transforms import _TRANSFORMS

        meta = {"name", "transform", "weight"}
        for i, comp in enumerate(self.components or []):
            where = f"generative.scoring.components[{i}]"
            name, tname = comp.get("name"), comp.get("transform")
            if not name:
                raise ValueError(f"{where} has no 'name'")
            if name not in SCORED_COMPONENTS:
                raise ValueError(
                    f"{where} scores {name!r}, which the scorer does not compute; it emits "
                    f"{sorted(SCORED_COMPONENTS)}. A component nothing measures is scored 0.0, then "
                    f"floored to 1e-9 by the aggregator, so it becomes the SAME constant factor for "
                    f"every candidate: the objective is silently dropped from the ranking rather "
                    f"than failing."
                )
            if tname not in _TRANSFORMS:
                raise ValueError(
                    f"{where} ({name}) uses unknown transform {tname!r}; "
                    f"known: {sorted(_TRANSFORMS)}"
                )
            sig = inspect.signature(_TRANSFORMS[tname])
            params = list(sig.parameters.values())[1:]  # the first parameter is the value itself
            required = [
                p.name for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            supplied = set(comp) - meta
            if missing := [r for r in required if r not in supplied]:
                raise ValueError(
                    f"{where} ({name}) uses transform {tname!r} which requires {missing}, "
                    f"but the component supplies {sorted(supplied) or 'nothing'}. This spec would "
                    f"raise at scoring time, not here."
                )
            if unknown := sorted(supplied - {p.name for p in params}):
                raise ValueError(
                    f"{where} ({name}) passes {unknown} to transform {tname!r}, which accepts "
                    f"{[p.name for p in params]}. A misspelled parameter would otherwise be dropped."
                )
        return self


class GenerativeConfig(BaseModel):
    """Candidate generation — read by the ``generative`` stage.

    ``stages`` was here too, alongside ``scoring.diversity_filter``, on the same "documents the
    campaign's shape" rationale and with the same problem: the generator sequence is defined by the
    REINVENT TOMLs under ``configs/generative/``, which ship and which the external scripts actually
    read. Two records of one decision, one of them inert and neither checked against the other, is how
    the diversity-filter value came to disagree with the file it described. Removed; the TOMLs are the
    record.
    """

    model_config = STRICT

    # The two implemented samplers, named. As a free string this selected by NEGATION: the stage
    # tests ``== "mock"`` and sends everything else to the REINVENT4 sampler, so ``reinvent`` or
    # ``reinvent-4`` or ``mockk`` all asked for the GPU sampler -- and a typo meant to say "mock"
    # would try to load REINVENT4 and fail somewhere far from the config that caused it.
    sampler: Literal["mock", "reinvent4"] = "mock"
    n_candidates: int = 500
    top_k: int = 20
    seed_core: str = ""
    scoring: GenerativeScoringConfig = Field(default_factory=GenerativeScoringConfig)


class EnvelopeDeriveConfig(BaseModel):
    """Derive the physchem window from THIS target's own actives instead of borrowing one.

    Set this and the explicit bounds below become a CEILING rather than the window: the derived
    envelope is intersected with them. That is deliberate. Deriving from known actives captures what
    has already been *made* against a target, which drifts larger than lead-like starting material —
    on the bromodomain case 30% of potent actives exceed 500 Da, so an unconstrained derivation puts the
    ceiling near 626 Da. Intersecting keeps the derivation honest about chemotype while letting the
    config state a lead-likeness intent the data cannot supply.
    """

    model_config = STRICT

    # ``potency_quantile`` was here and was read by nothing. The compounds the envelope is derived
    # from are selected by ``vls.actives`` (``pic50_quantile``, else ``pic50_min``), whose resolved
    # cut is recorded in the envelope as ``potency_cut``/``potency_cut_how``. Two config keys for one
    # decision, one of them inert, is worse than one: a config setting it got the OTHER key's cut and
    # a record that did not mention the disagreement. Removed rather than wired, because a second
    # potency cut for the same selection has nothing to express. Being STRICT, this section now
    # rejects a config that still sets it instead of ignoring it again.
    envelope_quantiles: list[float] = Field(default_factory=lambda: [0.05, 0.95])
    margin: float = 0.15                    # widen outward, so the filter excludes rather than defines
    must_admit_references: bool = True      # a filter rejecting your benchmarks is wrong
    intersect_with_explicit: bool = True     # treat the bounds below as a lead-likeness ceiling


class LeadLikeConfig(BaseModel):
    """Tier-0 physchem bounds. Strict: a mistyped bound would silently revert to a default and
    change how large the prepared deck is.

    When ``derive`` is set the window comes from the target's own actives, intersected with these
    numbers as a ceiling. Both the derived and the configured values are recorded, so a reader can see
    which bound actually applied.
    """

    model_config = STRICT

    derive: EnvelopeDeriveConfig | None = None

    mw_min: float = 300
    mw_max: float = 460
    logp_max: float = 3.5
    tpsa_max: float = 140
    hbd_max: int = 5
    hba_max: int = 10
    rotb_max: int = 10


class LibraryConfig(BaseModel):
    """Tier-0 purchasable deck.

    ``manifest`` is the pin: when it is set, ``medchem.vls.stage`` verifies the library file against the
    SHA-256 recorded in it BEFORE screening, and raises on a mismatch. ``path`` alone identifies nothing --
    a file at a path can be replaced.

    Field status, because three of these are records rather than controls:
      * ``path``, ``manifest``  -- LIVE. The path is opened; the manifest is verified against it.
      * ``lead_like``           -- LIVE. Supplies the physicochemical window applied to the library.
      * ``prefilters``          -- LIVE, and now honest about its scope: the ONLY selectable filter is
                                   ``pains``, so this is an optional list whose sole permitted member is
                                   that. The physicochemical window and de-duplication are
                                   UNCONDITIONAL -- ``lead_like`` drives the window and the library
                                   loader always de-duplicates -- so naming them here suggested a choice
                                   that does not exist. Unknown and duplicate members are rejected.
      * ``source``, ``snapshot``-- RECORD-ONLY. Copied verbatim into the provenance record so a reader can
                                   see which vendor deck and which vendor snapshot were intended. Neither
                                   selects, fetches or validates anything: the bytes are pinned by
                                   ``manifest``, not by these labels.
    """

    model_config = STRICT

    source: str = "zinc22"          # RECORD-ONLY: provenance label, selects nothing
    snapshot: str = ""              # RECORD-ONLY: provenance label, selects nothing
    path: str = ""
    manifest: str = ""              # the pin: verified against `path` before screening
    lead_like: LeadLikeConfig = Field(default_factory=LeadLikeConfig)
    # `pains` is the only SELECTABLE filter. The physicochemical window (from `lead_like`) and
    # de-duplication both run unconditionally, so they are not options and must not be listed as if
    # they were: the previous default read [lead_like_physchem, pains, dedup], which described two
    # unconditional behaviours as switchable. Empty means "no PAINS screen"; ["pains"] enables it.
    prefilters: list[Literal["pains"]] = Field(default_factory=lambda: ["pains"])


class Tier1Config(BaseModel):
    """Applicability-domain and prioritisation thresholds — the highest-value strict section.

    These decide which compounds are labelled in-domain, so a typo would substitute a default and
    relabel the deck while the run still reported success.

    ``derived_for`` is the target whose out-of-fold coverage curve produced these numbers. An
    applicability-domain threshold measured on one target is **not** transferable to another — the
    fingerprint-similarity distribution differs — and nothing previously stopped a screen from running
    on another target's thresholds. Leaving it unset is allowed but recorded as unprovenanced; setting
    it to a different target than the run is an error.

    The potency thresholds accept quantile forms for the same reason ``actives`` does: an absolute
    ``hit_floor`` of 6.0 or ``prio_point`` of 7.0 means different things across target classes.
    """

    model_config = STRICT

    derived_for: str | None = None
    in_domain: float = 0.5
    borderline: float = 0.35
    high_conf: float = 0.6
    # NO default. 0.955 was JAK1's measured conformal half-width; inheriting it silently applied one
    # target's uncertainty to another. Unset means "take this run's own qsar half-width".
    #
    # 0.0 is a LEGITIMATE VALUE, not a synonym for unset: it asks for point predictions with no
    # uncertainty band, which is how you measure what the band is doing to the strata. The consumer
    # resolved it with ``configured or measured``, so 0.0 fell through to the measured half-width and
    # the ablation silently ran the unablated screen. ``ge=0`` because a negative band would invert
    # every interval comparison in the tier logic and still produce a full census.
    conformal_halfwidth: float | None = Field(default=None, ge=0.0)
    hit_floor: float = 6.0
    hit_floor_quantile: float | None = None
    prio_point: float = 7.0
    prio_point_quantile: float | None = None
    prio_delta: float = 1.0
    fep_lower: float = 7.4
    fep_delta: float = 1.5
    known_reference_flag: float = 0.65


class ActivesConfig(BaseModel):
    """Known-active anchor set for the priority signal (not the AD term, not a gate).

    ``pic50_quantile`` takes precedence when set: "potent" is target-relative, and an absolute 8.0
    discards JQ1 (~7.1), the bromodomain field's benchmark compound. The resolved absolute value is
    recorded in the run metrics so a reader still sees the number that applied.
    """

    model_config = STRICT

    pic50_min: float = 8.0
    pic50_quantile: float | None = None
    max_anchors: int = 2500


class VlsConfig(BaseModel):
    """Virtual library screening.

    Permissive at *this* level on purpose: ``tiers``, ``budget_allocation``, ``validation`` and
    ``prospective`` are consumed by the out-of-package execution scripts and their shape is still
    moving. The three subsections the package stage actually reads are strict submodels, so the keys
    that can silently change a result are checked while campaign structure stays editable.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    library: LibraryConfig = Field(default_factory=LibraryConfig)
    tier1: Tier1Config = Field(default_factory=Tier1Config)
    actives: ActivesConfig = Field(default_factory=ActivesConfig)
    # Free-form compound-name -> SMILES; the keys are names, so there is no key set to enforce.
    known_reference: dict[str, str] = Field(default_factory=dict)

    # The out-of-package campaign sections, enumerated. `extra="allow"` is still right for their
    # CONTENTS -- the execution scripts own that shape and it is still moving -- but it was also
    # letting unknown keys in at THIS level, where the package's own live sections live. A config
    # writing ``tier_1:`` or ``active:`` or ``librrary:`` validated, reached the cache key, and left
    # the real ``tier1``/``actives``/``library`` at their defaults: the deck screened against
    # in_domain=0.5 and a 460 Da ceiling the config believed it had replaced.
    CAMPAIGN_SECTIONS: ClassVar[frozenset[str]] = frozenset(
        {"tiers", "budget_allocation", "validation", "prospective"}
    )

    @model_validator(mode="after")
    def _extra_keys_must_be_campaign_sections(self) -> VlsConfig:
        unknown = sorted(set(self.__pydantic_extra__ or {}) - self.CAMPAIGN_SECTIONS)
        if unknown:
            live = sorted(type(self).model_fields)
            raise ValueError(
                f"unknown vls key(s) {unknown}. This level is permissive only for the campaign "
                f"sections consumed outside the package ({sorted(self.CAMPAIGN_SECTIONS)}); the "
                f"package's own sections are {live}. An unrecognised key here does not extend a "
                f"section -- it sits beside one, leaving the real section at its defaults."
            )
        return self


class StructureConfig(BaseModel):
    """Receptor preparation — read by the ``receptor`` stage.

    Strict now that a stage consumes it. Before the receptor stage existed, ``reference_pdb`` was
    read by nothing: the receptor was a hand-prepared PDBQT committed under ``assets/``, so the
    structural arm was never config-driven and a second target simply had no receptor.

    ``hinge_residue`` and ``anchor_residue`` name the same thing in different target vocabularies — a
    kinase has a hinge, a bromodomain has a conserved anchoring asparagine. Both are LOAD-BEARING: the
    receptor stage asserts that the named residue exists, is the type named, and sits inside the docking
    box. They were documented as descriptive-only when nothing consumed them; that is no longer true.

    **Exactly one, or neither.** The two are alternative vocabularies for one slot, not two slots, and the
    receptor stage resolved them as ``anchor_residue or hinge_residue`` -- so a config supplying both had
    its hinge silently ignored. That is the worst available outcome for this particular field: the site
    residue is what proves the box is centred on the intended pocket, and for a multi-domain protein the
    difference between two candidate sites is the difference between two real answers. A config naming
    both is not expressing a preference, it is confused, and the honest response is to say so rather than
    to pick. Read the resolved value through :attr:`site_residue`, never by ``or``.
    """

    model_config = STRICT

    # RECORD-ONLY. No code reads this: `medchem.structure.receptor` prepares an engine-agnostic
    # receptor PDBQT and a docking box, and nothing in the package dispatches on an engine name. It
    # records which co-folding/docking backend the campaign intended, for a reader and for the
    # provenance record -- it does not select an implementation, and setting it changes nothing.
    engine: str = "boltz2"
    reference_pdb: str = ""
    reference_ligand: str | None = None  # HET resname defining the box; largest non-solvent if unset
    box_size: float = 22.0               # Å, cubic
    box_center: list[float] | None = None  # explicit [x,y,z]; overrides the ligand centroid
    ph: float = 7.4                      # protonation pH -- changes scores, so it is provenance
    # Crystallographic waters within this many Å of the box-defining ligand are RETAINED. None drops
    # every water, which is defensible for an ATP site and wrong for a bromodomain: acetyl-lysine
    # recognition is mediated by a conserved ordered-water network, so docking a dry BET pocket scores
    # a site that does not exist. Target-class dependent, therefore configured rather than assumed.
    keep_waters_within: float | None = None
    # A configured receptor that fails to build is an ERROR, not a partial success. Return codes were
    # recorded but not enforced, so a pdb2pqr or receptor-prep failure still produced a "box_only"
    # stage that a caller could mistake for a prepared receptor.
    allow_box_only: bool = False
    # These are no longer decorative. Either one, when set, is ASSERTED by the receptor stage: the
    # residue must exist, be the residue type named, and lie inside the docking box. That turns a
    # target-class annotation into a check that the box is centred on the intended pocket -- which for
    # a multi-domain protein is the difference between two real sites.
    hinge_residue: str | None = None     # kinase vocabulary
    anchor_residue: str | None = None    # bromodomain equivalent; checked identically
    max_protein_fraction_in_box: float = 0.60  # above this the box is the fold, not a site

    @model_validator(mode="after")
    def _at_most_one_site_residue_vocabulary(self) -> StructureConfig:
        """Exactly one of the two, or neither, and neither may be blank when supplied."""
        blank = [n for n in ("hinge_residue", "anchor_residue")
                 if getattr(self, n) is not None and not str(getattr(self, n)).strip()]
        if blank:
            raise ValueError(
                f"structure.{blank[0]} is set to an empty value. A blank residue name asserts nothing, "
                f"and the receptor stage would treat it as unset -- so it reads as a site check that is "
                f"silently not happening. Omit the key, or name the residue."
            )
        if self.hinge_residue is not None and self.anchor_residue is not None:
            raise ValueError(
                f"structure names BOTH a hinge residue ({self.hinge_residue!r}) and an anchor residue "
                f"({self.anchor_residue!r}). They are two vocabularies for ONE slot -- a kinase hinge and "
                f"a bromodomain's conserved anchoring asparagine -- not two independent checks, and the "
                f"receptor stage can assert only one site residue. Supplying both previously resolved by "
                f"truthiness precedence and silently dropped the hinge. Name whichever one applies to "
                f"this target."
            )
        return self

    @property
    def site_residue(self) -> str | None:
        """The configured site residue, resolved WITHOUT truthiness precedence, or None if neither.

        The validator above guarantees at most one is set, so this cannot silently prefer either; the
        explicit ``is not None`` tests are what make that guarantee visible at the point of use rather
        than only at the point of validation.
        """
        if self.anchor_residue is not None:
            return self.anchor_residue
        if self.hinge_residue is not None:
            return self.hinge_residue
        return None


class Config(BaseModel):
    """Root config. Unknown top-level sections are REJECTED.

    This was the last permissive surface, and it was the worst place to leave one. Every section named
    here is now modeled, so an unrecognised key at this level does not extend anything -- it sits beside
    a real section and leaves that section entirely at its defaults. ``featuers:`` validated, reached the
    cache key, and the run featurised with 2048 chiral bits and nine descriptors while the config on
    screen said otherwise. That is the exact failure this whole module is organised against, and it was
    reachable through the one door still open.

    The reason it stayed open was ``loop`` and ``structure`` -- sections a stage did not yet read.
    ``structure`` has been modeled since the receptor stage was written, and ``loop`` was deleted from
    every config rather than left as decoration. Nothing needs the door now, and a new section costs one
    line here, which is a better price than a silent typo.

    ``extra="allow"`` is retained so a rejected key can be NAMED in the error; the validator below is
    what makes it fail closed.
    """

    model_config = ConfigDict(extra="allow")

    # The project slug labels a RUN; it does not select code paths. Stages register under the
    # generic "discovery" pipeline so a new target is data plus config, never a new graph.
    target: str = "discovery"
    seed: int = 42
    demo: DemoConfig = Field(default_factory=DemoConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    generative: GenerativeConfig = Field(default_factory=GenerativeConfig)
    vls: VlsConfig = Field(default_factory=VlsConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    # Stages to leave out of the graph (ADR 0006). Typed HERE rather than picked up as an untyped
    # root extra, which is how the CLI read it: `getattr(cfg, "disable_stages", [])` on a permissive
    # root returns whatever YAML produced. ``disable_stages: selectivity`` -- a string, and the
    # obvious way to write one name -- iterated into the characters of "selectivity" and disabled
    # nothing, because no stage is called "s". The runner then checked that no kept stage required a
    # dropped one, found none, and ran the full graph.
    disable_stages: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unknown_top_level_sections_are_rejected(self) -> Config:
        """A misspelled section name leaves the real section at its defaults. See the class docstring."""
        unknown = sorted(self.__pydantic_extra__ or {})
        if unknown:
            known = sorted(type(self).model_fields)
            raise ValueError(
                f"unknown top-level config section(s) {unknown}. The sections this pipeline reads are "
                f"{known}. An unrecognised name here does not extend a section -- it sits beside one, "
                f"and the real section then runs entirely on its defaults while the config appears to "
                f"configure it. Add a model for a genuinely new section rather than relying on this "
                f"level being permissive."
            )
        return self

    @model_validator(mode="after")
    def _disable_stages_must_be_distinct_names(self) -> Config:
        """Shape only. Whether a name IS a stage is checked by the runner, which knows the graph.

        Kept here because these two cases need no graph to judge: an empty string can never name a
        stage, and a repeated name is a request that cannot be honoured twice.
        """
        seen = list(self.disable_stages)
        if any(not str(s).strip() for s in seen):
            raise ValueError("disable_stages contains an empty name")
        if len(set(seen)) != len(seen):
            dupes = sorted({s for s in seen if seen.count(s) > 1})
            raise ValueError(
                f"disable_stages repeats {dupes}. A stage is disabled or it is not; listing it twice "
                f"means the config was assembled from two sources that disagree."
            )
        return self

    @model_validator(mode="after")
    def _box_center_must_be_three_numbers(self) -> Config:
        bc = self.structure.box_center
        if bc is not None and len(bc) != 3:
            raise ValueError(
                f"structure.box_center must be [x, y, z]; got {len(bc)} value(s). An explicit centre "
                f"is the documented escape hatch for apo structures, so a malformed one must not "
                f"silently fall back to a ligand centroid that does not exist."
            )
        return self

    @model_validator(mode="after")
    def _library_prefilters_must_be_unique(self) -> Config:
        """``pains`` at most once. The member type rejects an unknown name; a repeat is still a request
        that cannot be honoured, and silently collapsing it would hide a config error."""
        pf = list(self.vls.library.prefilters)
        if len(set(pf)) != len(pf):
            raise ValueError(
                f"vls.library.prefilters contains duplicates ({pf}). The only selectable filter is "
                f"'pains'; the physicochemical window and de-duplication run unconditionally."
            )
        return self

    @model_validator(mode="after")
    def _eval_splits_must_be_the_implemented_set(self) -> Config:
        """Exactly ``["random", "scaffold", "temporal"]`` -- same members, same order, no duplicates.

        The member type already rejects an unknown name, which leaves three ways to write something the
        harness will not honour: a subset, a reordering, and a repeat. All three were accepted, and all
        three are silently ignored, because the harness computes and reports all three splits
        unconditionally. A field that cannot change behaviour must not be able to express a request.
        """
        want = ["random", "scaffold", "temporal"]
        got = list(self.eval.splits)
        if got != want:
            why = ("duplicate entries" if len(set(got)) != len(got)
                   else "a different order" if sorted(got) == sorted(want)
                   else "a subset" if set(got) < set(want)
                   else "a different set")
            raise ValueError(
                f"eval.splits must be exactly {want} -- got {got} ({why}). The harness computes and "
                f"reports all three splits unconditionally, so this field records the split set rather "
                f"than selecting it, and a request it cannot honour is rejected instead of ignored."
            )
        return self

    @model_validator(mode="after")
    def _temporal_cutoffs_must_agree(self) -> Config:
        """``curation`` and ``eval`` each carry a cutoff year, and they must be the same year.

        They are consumed at different points for different purposes, which is why two fields exist:
        ``curation.temporal_cutoff_year`` decides where the era-split LABEL is cut (``pIC50_pre`` is the
        median over pre-cutoff measurements only), and ``eval.temporal_cutoff_year`` decides where the
        evaluation splits ROWS into train and test.

        If they diverge, the split is scored against labels cut at a different boundary, and the direction
        that matters is silent: with a curation cutoff LATER than the evaluation cutoff, a training row
        from before the evaluation boundary carries a label informed by measurements published after it --
        exactly the post-cutoff leakage the era-split design exists to remove, reintroduced by
        configuration rather than by code.

        Nothing detected this. Both fields validated independently, both accepted any year, and a run with
        divergent cutoffs produced a plausible temporal R2 with no warning. So the constraint is enforced
        here rather than documented.

        Only checked when curation sets a cutoff at all: it is optional, and a config that omits it is
        asking for no era-split labels rather than for a mismatched pair.
        """
        cur = self.curation.temporal_cutoff_year
        if cur is None:
            raise ValueError(
                "curation.temporal_cutoff_year is required: the evaluation harness always computes a "
                "temporal split, so omitting the curation cutoff would score that split against labels "
                "built over all years."
            )
        if cur != self.eval.temporal_cutoff_year:
            raise ValueError(
                f"curation.temporal_cutoff_year ({cur}) must equal eval.temporal_cutoff_year "
                f"({self.eval.temporal_cutoff_year}). The first cuts the era-split LABEL, the second "
                f"splits train/test ROWS; different years mean rows are scored against labels cut at "
                f"another boundary, which can reintroduce post-cutoff leakage silently."
            )
        return self

    @model_validator(mode="after")
    def _selectivity_pairs_must_name_configured_targets(self) -> Config:
        """A pair naming an unconfigured target yields an empty Δ set, not an error — so check it
        here. Only enforced when targets are configured at all (bare Config() is used in tests)."""
        known = set(self.data.targets)
        if not known:
            return self
        pairs = self.model.selectivity.pairs
        if pairs is None:
            # Derived from data.targets, which by construction names configured targets. Written as an
            # explicit None check rather than `pairs or ()`: that form is indistinguishable from the
            # empty-request bug this module is organised against, and a test now scans for it.
            return self
        for pair in pairs:
            parts = pair.split("-")
            if len(parts) != 2:
                raise ValueError(
                    f"model.selectivity.pairs entry {pair!r} must have the form 'PRIMARY-COMPARATOR'"
                )
            unknown = [p for p in parts if p not in known]
            if unknown:
                raise ValueError(
                    f"model.selectivity.pairs entry {pair!r} names {unknown} which is not in "
                    f"data.targets ({sorted(known)}). Selectivity would be computed over no rows."
                )
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a :class:`Config`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    return Config.model_validate(raw)
