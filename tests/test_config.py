"""Configuration is the retargeting interface, so its failure modes must be loud.

Every test here encodes a mistake that ``extra="allow"`` used to accept silently. That is the point:
a config error that validates produces a complete, plausible, wrong run — the worst failure shape
this project has, and the one docs/PITFALLS.md is mostly about.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from medchem.config import Config, DataConfig, load_config

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------------------------
# Shared helpers for the "config value replaced by a default" guard. Module level and used by BOTH
# the repository scan and the synthetic mutation test, because the two previously had separate
# implementations that disagreed: the synthetic one recognised `st.` as a config read and the
# production one did not, so the scan skipped the very expression the test claimed to prove.
# ---------------------------------------------------------------------------------------------

# Names through which a stage reads configuration. `st.` is here because `medchem.structure.receptor`
# binds `ctx.config.structure` to `st` -- omitting it made the guard blind to that whole module.
_CFG_READ = re.compile(
    r"\b(ctx\.config|cfg_data|cfg\.|gen\.|feat\.|sel_cfg|pot_cfg|t1\.|st\.|vls_cfg|lib_cfg|acfg"
    r"|self\.(?:data|features|model|eval|vls|generative|curation|structure))"
)


def _reads_config(segment: str) -> bool:
    """Does this source segment read a configuration value?"""
    return bool(_CFG_READ.search(segment)) or ".get(" in segment


def _substantive(node, *, segment: str) -> bool:
    """Does the right-hand side OVERRIDE the left, rather than normalise an absent value?

    An empty literal container, an empty string or a zero normalises ``None`` and overrides nothing.
    Anything else standing in for a config value is a default -- or, for an Attribute / Subscript / Call,
    a SECOND CONFIG VALUE, which is the same failure in a different shape.
    """
    import ast

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.keys)
    if isinstance(node, ast.Constant):
        return node.value not in ("", 0, 0.0, False, None)
    if isinstance(node, ast.Name):                      # e.g. `or _DEFAULT_DESCRIPTORS`
        return node.id.isupper() or node.id.startswith("_DEFAULT")
    if isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        return _reads_config(segment)
    return False


def _or_fallback_findings(text: str, where: str) -> list[str]:
    """Every `config-value or <substantive>` expression in one module's source."""
    import ast

    out: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)):
            continue
        segment = ast.get_source_segment(text, node) or ""
        if not _reads_config(segment):
            continue
        if _substantive(node.values[-1], segment=segment):
            out.append(f"{where}:{node.lineno}: {segment[:90]}")
    return out



def test_flagship_config_validates_under_strict_sections():
    """The shipped config must survive the strict models — otherwise the strictness is theatre."""
    cfg = load_config(REPO / "configs" / "jak1.yaml")
    assert cfg.data.primary == "JAK1"
    assert cfg.data.comparators == ["JAK2", "JAK3", "TYK2"]
    assert cfg.features.n_bits == 2048 and cfg.features.use_chirality is True
    assert cfg.model.selectivity.pairs == ["JAK1-JAK2", "JAK1-JAK3", "JAK1-TYK2"]
    assert cfg.eval.gates.scaffold_cv_r2_min == 0.55
    # unmodeled sections still reach the cache key rather than being dropped
    dumped = cfg.model_dump()
    # `loop` is deliberately absent: it was read by nothing and was deleted rather than left as
    # decoration. Keeping it in this assertion would re-enshrine the thing that was removed.
    assert {"vls", "structure", "generative"} <= set(dumped)


def test_typo_in_a_stage_read_key_is_rejected():
    """`primry` instead of `primary` used to validate, persist into the cache key, and then let the
    stage fall back to a default. Now it fails at load."""
    with pytest.raises(ValidationError, match="primry"):
        Config.model_validate({"data": {"chembl_release": "34", "primry": "JAK1"}})


def test_primary_target_must_be_one_of_the_configured_targets():
    """Otherwise curation selects zero rows for the primary and reports an empty training set as
    though it were a legitimate result."""
    with pytest.raises(ValidationError, match="not a key of data.targets"):
        DataConfig(targets={"JAK1": "CHEMBL2835"}, primary="JAK2")


def test_selectivity_pairs_must_name_configured_targets():
    """A pair against an unpulled target yields an empty Δ set, not an error — so it is checked."""
    with pytest.raises(ValidationError, match="not in data.targets"):
        Config.model_validate({
            "data": {"targets": {"ACME1": "CHEMBL1"}, "primary": "ACME1"},
            "model": {"selectivity": {"pairs": ["ACME1-ACME9"]}},
        })


def test_malformed_selectivity_pair_is_rejected():
    with pytest.raises(ValidationError, match="PRIMARY-COMPARATOR"):
        Config.model_validate({
            "data": {"targets": {"ACME1": "CHEMBL1", "ACME2": "CHEMBL2"}, "primary": "ACME1"},
            "model": {"selectivity": {"pairs": ["ACME1_ACME2"]}},
        })


def test_dedup_strategy_is_constrained():
    """`dedup: mediam` would otherwise be accepted and then ignored."""
    with pytest.raises(ValidationError):
        DataConfig(dedup="mediam")  # type: ignore[arg-type]


def test_comparators_derive_from_config_not_from_constants():
    """The property a second target depends on: comparators are whatever else was configured."""
    cfg = DataConfig(targets={"P38A": "C1", "P38B": "C2", "ERK2": "C3"}, primary="P38A")
    assert cfg.comparators == ["P38B", "ERK2"]


def test_permissiveness_is_inside_the_campaign_sections_not_beside_them():
    """A forward-looking section must not need a model before its stage exists — but "forward
    looking" has to mean somewhere in particular.

    This test used to assert that ANY new key under ``vls`` validated, which is what let ``tier_1``
    sit beside ``tier1`` and leave the real applicability-domain thresholds at their defaults. The
    permissiveness that was actually wanted is for the CONTENTS of the four campaign sections, whose
    shape the out-of-package execution scripts own; a brand-new name at the ``vls`` level is a typo
    far more often than it is a new section, and the cost of the two is not symmetric.
    """
    cfg = Config.model_validate({
        "vls": {"tiers": [{"name": "docking_triage", "some_future_key": 1}]},
        "structure": {"engine": "boltz2"},
    })
    assert cfg.model_dump()["vls"]["tiers"][0]["some_future_key"] == 1
    with pytest.raises(ValidationError, match="unknown vls key"):
        Config.model_validate({"vls": {"some_future_section": 1}})


def test_bare_config_is_valid():
    """Tests and the demo pipeline construct Config() with no target panel at all; the
    cross-section validators must not fire on an empty panel."""
    cfg = Config()
    assert cfg.target == "discovery" and cfg.data.comparators == []


# --- the strict/permissive split is a POLICY, so pin it -------------------------------------------

def test_potency_model_section_rejects_unknown_keys():
    """`rf_n_estimatorss` would otherwise train 400 trees while looking configured."""
    with pytest.raises(ValidationError, match="rf_n_estimatorss"):
        Config.model_validate({"model": {"potency": {"rf_n_estimatorss": 900}}})


def test_removed_decorative_keys_are_now_rejected():
    """`algorithms` and `calibration` were read by nothing: RF and XGB both always train and the
    conformal interval always runs. Accepting them implied a choice that did not exist."""
    for key, value in (("algorithms", ["random_forest"]), ("calibration", "conformal")):
        with pytest.raises(ValidationError, match=key):
            Config.model_validate({"model": {"potency": {key: value}}})
    with pytest.raises(ValidationError, match="aggregation"):
        Config.model_validate({"generative": {"scoring": {"aggregation": "sum"}}})


def test_applicability_domain_thresholds_reject_unknown_keys():
    """The highest-value strict section: these thresholds decide which compounds are called
    in-domain, so a typo must not silently substitute the default."""
    with pytest.raises(ValidationError, match="in_domian"):
        Config.model_validate({"vls": {"tier1": {"in_domian": 0.5}}})


def test_lead_like_bounds_reject_unknown_keys():
    """A mistyped bound would change how large the prepared deck is, with no error."""
    with pytest.raises(ValidationError, match="mw_maxx"):
        Config.model_validate({"vls": {"library": {"lead_like": {"mw_maxx": 500}}}})


def test_externally_driven_vls_subsections_stay_permissive():
    """`tiers` / `budget_allocation` / `validation` / `prospective` are consumed by out-of-package
    execution scripts and their shape is still moving — forbidding keys would reject a legitimate
    campaign edit. This test exists so that stays a decision, not an accident."""
    cfg = Config.model_validate({"vls": {
        "tiers": [{"name": "docking_triage", "some_new_engine_flag": True}],
        "budget_allocation": {"tier9_future": {"exploit": 1}},
        "validation": {"new_benchmark": "x"},
        "prospective": {"assays": ["spr"]},
    }})
    dumped = cfg.model_dump()["vls"]
    assert dumped["tiers"][0]["some_new_engine_flag"] is True
    assert dumped["budget_allocation"]["tier9_future"] == {"exploit": 1}


def test_known_reference_keys_are_compound_names_not_a_fixed_schema():
    cfg = Config.model_validate({"vls": {"known_reference": {"some_new_drug": "CCO"}}})
    assert cfg.vls.known_reference["some_new_drug"] == "CCO"


def test_the_loop_section_is_rejected_because_nothing_reads_it():
    """`loop` is neither modeled nor accepted, and both halves of that matter.

    NOT MODELED: an AST audit of src/ found zero of its keys read anywhere. Giving it a model would
    imply the active-learning loop runs, and a config section implying a feature runs is worse than no
    section. The literal check below is the tripwire — if the loop ever starts reading config, this
    test fails and demands a strict model rather than silently allowing the section back.

    NOT ACCEPTED: this test used to assert the opposite — that `loop` VALIDATED, because the root was
    permissive. That permissiveness is what let any misspelled top-level section through, so the root
    is now strict and `loop` is rejected with the rest. A section nothing reads should not validate;
    that was the argument for deleting it from the configs, and it applies here too.
    """
    import ast

    src = Path(__file__).resolve().parent.parent / "src" / "medchem"
    literals: set[str] = set()
    for f in src.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.add(node.value)
    assert not ({"acquisition", "conformal_uncertainty"} & literals), (
        "the active-learning loop now reads config; give `loop` a strict model"
    )
    with pytest.raises(ValidationError, match="unknown top-level config section"):
        Config.model_validate({"loop": {"rounds": 0, "anything": 1}})


def test_a_misspelled_top_level_section_is_rejected():
    """The root was the last permissive surface, and the worst one: a misspelled section name does not
    extend anything, it sits beside the real section and leaves it entirely at its defaults."""
    for bad in ("featuers", "curration", "modle", "evaluation"):
        with pytest.raises(ValidationError, match="unknown top-level config section"):
            Config.model_validate({bad: {"n_bits": 512}})
    # and the real thing still validates, so the check is not simply refusing everything
    assert Config.model_validate({"features": {"n_bits": 512}}).features.n_bits == 512


def test_the_removed_generative_records_are_rejected():
    """`stages` and `scoring.diversity_filter` were read nowhere and were defended as records of the
    external campaign — but `diversity_filter` said `scaffold_similarity` while the TOML it described
    specifies IdenticalMurckoScaffold. The TOMLs ship and are the record; the duplicate drifted."""
    with pytest.raises(ValidationError, match="stages"):
        Config.model_validate({"generative": {"stages": ["mol2mol"]}})
    with pytest.raises(ValidationError, match="diversity_filter"):
        Config.model_validate({"generative": {"scoring": {"diversity_filter": "scaffold_similarity"}}})


# --- the scoring spec must be EXECUTABLE, not merely well-formed ----------------------------------

def test_shipped_scoring_spec_is_executable_and_there_is_no_module_default_to_drift_from():
    """The shipped spec declared `transform: sigmoid` with no `center` for three components, so it could
    never have run. It went unnoticed because the stage ignored config and used a hard-coded default.

    This test used to assert the config EQUALLED that default, which kept the two from drifting. The
    default is now deleted outright -- a reward function is a per-target scientific claim, so there is
    nothing for the code to supply -- and the stronger property is that no such constant exists to drift
    from. Executability is enforced by the validator, so loading the config at all is the check.
    """
    import medchem.generative.stage as gen_stage

    assert not hasattr(gen_stage, "_SPEC"), (
        "a module-level default scoring spec is back; it was JAK1-shaped and was injected silently"
    )
    cfg = load_config(REPO / "configs" / "jak1.yaml")   # raises if any component is unexecutable
    names = [c["name"] for c in cfg.generative.scoring.components or []]
    assert len(names) == len(set(names)) and len(names) == 7


def test_component_missing_a_required_transform_parameter_is_rejected():
    """`sigmoid` has no default centre, so this spec raises at scoring time — far from the cause."""
    with pytest.raises(ValidationError, match="requires \\['center'\\]"):
        Config.model_validate({"generative": {"scoring": {"components": [
            {"name": "qed", "transform": "sigmoid"},
        ]}}})


def test_component_with_a_misspelled_transform_parameter_is_rejected():
    """`centre` would otherwise be silently dropped and the centre left at whatever the default is."""
    with pytest.raises(ValidationError, match="centre"):
        Config.model_validate({"generative": {"scoring": {"components": [
            {"name": "qed", "transform": "sigmoid", "centre": 0.5},
        ]}}})


def test_component_with_an_unknown_transform_is_rejected():
    with pytest.raises(ValidationError, match="unknown transform"):
        Config.model_validate({"generative": {"scoring": {"components": [
            {"name": "qed", "transform": "sigmoidal", "center": 0.5},
        ]}}})


def test_transform_requirements_are_introspected_not_duplicated():
    """If a transform gains a required parameter, existing configs must start failing. Duplicating
    the requirement list in the validator would silently miss that."""
    import inspect

    from medchem.generative.scoring import _TRANSFORMS

    required = {
        n: [p.name for p in list(inspect.signature(f).parameters.values())[1:]
            if p.default is inspect.Parameter.empty]
        for n, f in _TRANSFORMS.items()
    }
    assert required["sigmoid"] == ["center"]
    assert required["double_sigmoid"] == ["low", "high"]
    # a spec valid for one transform must not be valid for another with different requirements
    with pytest.raises(ValidationError):
        Config.model_validate({"generative": {"scoring": {"components": [
            {"name": "mw", "transform": "double_sigmoid", "center": 7.5},
        ]}}})


def test_second_target_config_validates_under_the_same_strict_models():
    """The generalisation claim is only meaningful if the second target's config passes the SAME
    validation, unrelaxed. Also pins the facts the two-target comparison rests on."""
    cfg = load_config(REPO / "configs" / "brd4.yaml")
    assert cfg.data.primary == "BRD4"
    assert cfg.data.comparators == ["BRD2", "BRD3", "BRDT"]
    assert cfg.model.selectivity.pairs == ["BRD4-BRD2", "BRD4-BRD3", "BRD4-BRDT"]
    # identical gates: loosening them for a harder target would make "it generalises" unfalsifiable
    jak1 = load_config(REPO / "configs" / "jak1.yaml")
    assert cfg.eval.gates == jak1.eval.gates
    # featurisation and model hyperparameters are UNCHANGED -- if they had to be retuned per target,
    # the "config-only retargeting" claim would be false in a way this test would catch
    assert cfg.features == jak1.features
    assert cfg.model.potency == jak1.model.potency
    # ...while the physchem window is deliberately NOT inherited (it rejects 5 of 7 BET references)
    assert cfg.vls.library.lead_like != jak1.vls.library.lead_like


def test_second_target_physchem_window_admits_its_own_reference_compounds():
    """The guard the two-target run showed was missing: a physchem filter that rejects the compounds
    you benchmark against is wrong. JAK1's window rejects 5 of BRD4's 7 references, so inheriting it
    would have made the screen unable to find this target's known chemistry -- while reporting
    success. Checked in both directions so neither config can drift into the other's assumptions."""
    pytest.importorskip("rdkit")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors

    # rdkit stubs omit these three; all are real at runtime.
    RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]

    def admitted(cfg) -> int:
        ll = cfg.vls.library.lead_like
        n = 0
        for smi in cfg.vls.known_reference.values():
            m = Chem.MolFromSmiles(smi)
            assert m is not None, f"unparseable reference SMILES: {smi}"
            mw = Descriptors.MolWt(m)  # pyright: ignore[reportAttributeAccessIssue]
            logp = Crippen.MolLogP(m)  # pyright: ignore[reportAttributeAccessIssue]
            if ll.mw_min <= mw <= ll.mw_max and logp <= ll.logp_max:
                n += 1
        return n

    for name in ("jak1", "brd4"):
        cfg = load_config(REPO / "configs" / f"{name}.yaml")
        refs = len(cfg.vls.known_reference)
        assert admitted(cfg) == refs, (
            f"configs/{name}.yaml lead_like window rejects "
            f"{refs - admitted(cfg)}/{refs} of its own reference compounds"
        )


def test_ad_thresholds_from_another_target_are_refused():
    """An applicability-domain threshold measured on one target is not transferable: the
    fingerprint-similarity distribution differs. Nothing previously stopped a screen from relabelling a
    deck against another target's coverage curve."""
    cfg = Config.model_validate({
        "target": "brd4",
        "vls": {"tier1": {"derived_for": "jak1", "in_domain": 0.5}},
    })
    assert cfg.vls.tier1.derived_for == "jak1" and cfg.target == "brd4"
    # the refusal lives in the stage (it needs the run's target); the config records the mismatch
    assert cfg.vls.tier1.derived_for != cfg.target


def test_tier1_potency_thresholds_accept_quantile_forms():
    """`hit_floor: 6.0` and `prio_point: 7.0` mean different things across target classes, exactly as
    `actives.pic50_min: 8.0` did."""
    cfg = Config.model_validate({
        "vls": {"tier1": {"hit_floor_quantile": 0.4, "prio_point_quantile": 0.8}}
    })
    assert cfg.vls.tier1.hit_floor_quantile == 0.4
    assert cfg.vls.tier1.prio_point_quantile == 0.8
    assert cfg.vls.tier1.hit_floor == 6.0, "the absolute stays as the fallback"


def test_loop_section_is_gone_from_every_shipped_config():
    """It was read by nothing. A config section implying a feature runs is worse than no section."""
    import yaml

    for path in sorted((REPO / "configs").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert "loop" not in raw, f"{path.name} still ships a decorative loop section"


def test_every_reinvent_config_passes_the_required_oracle_config():
    """Making --config mandatory in the scoring bridge broke every existing REINVENT config, which
    invoked it without one. A required argument is only safe if every caller supplies it, and the
    callers here are TOML files no type checker reads."""
    tomls = sorted((REPO / "configs" / "generative").rglob("*.toml"))
    assert tomls, "no generative configs found; this test would pass vacuously"
    offenders = []
    for t in tomls:
        text = t.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "reinvent_oracle.py" in line and "--config" not in line:
                offenders.append(f"{t.name}: {line.strip()[:70]}")
    assert not offenders, "oracle invoked without --config:\n  " + "\n  ".join(offenders)


# --- fields that were accepted and then ignored ------------------------------------------------------
#
# Five fields were declared, type-validated, and read by no stage: data.standardize_units_to, data.dedup,
# features.fingerprint, eval.splits and model.selectivity.mode. Setting them changed nothing, so a config
# could ask for mean aggregation, a different fingerprint family, or a subset of splits, and get the
# defaults with no warning. All four frozen configs happen to set exactly the implemented values, so
# narrowing each type to what the code does keeps them loading and turns the fields from decoration into
# an enforced statement.

@pytest.mark.parametrize(
    ("section", "field", "bad"),
    [
        ("data", "standardize_units_to", "pKi"),
        ("data", "dedup", "mean"),
        ("features", "fingerprint", "fcfp4"),
        ("eval", "splits", ["random", "grouped"]),
    ],
)
def test_unimplemented_config_values_are_rejected(section: str, field: str, bad: object):
    cfg = load_config(REPO / "configs" / "jak1.yaml").model_dump()
    cfg[section][field] = bad
    with pytest.raises(ValidationError):
        Config.model_validate(cfg)


def test_unimplemented_selectivity_mode_is_rejected():
    cfg = load_config(REPO / "configs" / "jak1.yaml").model_dump()
    cfg["model"]["selectivity"]["mode"] = "subtract_potency_models"
    with pytest.raises(ValidationError):
        Config.model_validate(cfg)


@pytest.mark.parametrize("name", ["jak1", "jak1_sensitivity", "brd4", "brd4_sensitivity"])
def test_frozen_configs_still_load_under_the_narrowed_types(name: str):
    """Narrowing must not invalidate the configs that produced the frozen results."""
    cfg = load_config(REPO / "configs" / f"{name}.yaml")
    assert cfg.data.standardize_units_to == "pIC50"
    assert cfg.data.dedup == "median"
    assert cfg.features.fingerprint == "ecfp4"
    assert cfg.eval.splits == ["random", "scaffold", "temporal"]
    assert cfg.model.selectivity.mode == "direct_delta"


# --- the two temporal cutoffs must agree -------------------------------------------------------------

def test_divergent_temporal_cutoffs_are_rejected():
    """The leakage vector this closes is silent, which is why it needs a test rather than a comment.

    `curation.temporal_cutoff_year` cuts the era-split LABEL; `eval.temporal_cutoff_year` splits train/test
    ROWS. With curation LATER than eval, a training row from before the evaluation boundary carries a label
    informed by measurements published after it — post-cutoff leakage, reintroduced by configuration.
    Both fields validated independently and accepted any year.
    """
    cfg = load_config(REPO / "configs" / "jak1.yaml").model_dump()
    cfg["curation"]["temporal_cutoff_year"] = 2018
    cfg["eval"]["temporal_cutoff_year"] = 2022
    with pytest.raises(ValidationError, match="must equal eval.temporal_cutoff_year"):
        Config.model_validate(cfg)

    # and the leakage-producing direction specifically
    cfg["curation"]["temporal_cutoff_year"] = 2023
    with pytest.raises(ValidationError, match="must equal eval.temporal_cutoff_year"):
        Config.model_validate(cfg)


def test_absent_curation_cutoff_is_rejected():
    """It is REQUIRED, because the temporal split is not optional.

    The harness always computes a temporal split and every published panel reports one, so a curation
    cutoff of None would leave that split scored against labels built over all years -- the mismatch the
    era-split design exists to remove.
    """
    cfg = load_config(REPO / "configs" / "jak1.yaml").model_dump()
    cfg["curation"]["temporal_cutoff_year"] = None
    with pytest.raises(ValidationError):
        Config.model_validate(cfg)


@pytest.mark.parametrize("bad", [[], ["random"], ["random", "scaffold"],
                                 ["random", "random", "scaffold", "temporal"],
                                 ["temporal", "scaffold", "random"]])
def test_eval_splits_must_be_the_exact_implemented_set(bad: list[str]):
    """Empty, subset, duplicate and reordered are all rejected.

    The member type already rejects an unknown name; these four are the ways to write a request the
    harness cannot honour, because it computes and reports all three splits unconditionally.
    """
    cfg = load_config(REPO / "configs" / "jak1.yaml").model_dump()
    cfg["eval"]["splits"] = bad
    with pytest.raises(ValidationError, match="eval.splits must be exactly"):
        Config.model_validate(cfg)


@pytest.mark.parametrize("name", ["jak1", "jak1_sensitivity", "brd4", "brd4_sensitivity"])
def test_frozen_configs_have_agreeing_cutoffs(name: str):
    cfg = load_config(REPO / "configs" / f"{name}.yaml")
    assert cfg.curation.temporal_cutoff_year == cfg.eval.temporal_cutoff_year == 2022


# ---------------------------------------------------------------------------------------------
# Fail-closed configuration.
#
# Each test below is a config that used to validate and then not happen. They divide into three
# shapes, and the third is the one that took longest to see:
#
#   1. a key nothing reads     -- accepted, hashed into the cache key, ignored;
#   2. a value nothing honours -- a sampler name, a stage name, an XGBoost parameter;
#   3. an EMPTY request        -- `[]` or `0.0`, indistinguishable from "unset" because the field's
#                                 default was itself falsy and every consumer wrote `configured or
#                                 <fallback>`. These are the dangerous ones: each is a deliberate
#                                 ablation, and each silently ran unablated and reported a number
#                                 that looked like the ablation's answer.
# ---------------------------------------------------------------------------------------------


def test_disable_stages_must_be_a_list_not_a_bare_string():
    """`disable_stages: selectivity` is the obvious way to write one name, and it iterated into
    the characters of the string — disabling nothing, because no stage is called "s"."""
    with pytest.raises(ValidationError):
        Config.model_validate({"disable_stages": "selectivity"})


def test_disable_stages_rejects_repeats_and_empty_names():
    with pytest.raises(ValidationError, match="repeats"):
        Config.model_validate({"disable_stages": ["selectivity", "selectivity"]})
    with pytest.raises(ValidationError, match="empty name"):
        Config.model_validate({"disable_stages": [""]})


def test_disabling_a_stage_that_does_not_exist_is_rejected_by_the_runner():
    """The graph is the runner's to know, so the name check lives there. A misspelling ran the
    full pipeline and reported success."""
    import medchem.stages  # noqa: F401  (registers the stages)
    from medchem.pipeline import runner

    with pytest.raises(ValueError, match="no such stage"):
        runner.plan("discovery", ["selectivty"])
    assert [s.name for s in runner.plan("discovery", ["selectivity"])].count("selectivity") == 0


def test_an_unknown_key_beside_the_live_vls_sections_is_rejected():
    """`tier_1` sits BESIDE `tier1`; it does not extend it. The screen then ran on default
    applicability-domain thresholds while the config showed the intended ones."""
    with pytest.raises(ValidationError, match="unknown vls key"):
        Config.model_validate({"vls": {"tier_1": {"in_domain": 0.9}}})
    with pytest.raises(ValidationError, match="unknown vls key"):
        Config.model_validate({"vls": {"active": {"pic50_min": 7.0}}})


def test_xgb_parameters_are_the_ones_qsar_actually_forwards():
    """A free-form dict accepted any XGBoost parameter and forwarded five. `min_child_weight` is a
    real parameter of the estimator and is NOT one of the five, so it changed the cache key,
    changed nothing about the model, and produced a run that reported a tuning it never applied."""
    cfg = load_config(REPO / "configs" / "jak1.yaml").model_dump()
    cfg["model"]["potency"]["xgb"]["min_child_weight"] = 3
    with pytest.raises(ValidationError, match="min_child_weight"):
        Config.model_validate(cfg)


def test_misspelled_xgb_parameter_is_rejected():
    cfg = load_config(REPO / "configs" / "jak1.yaml").model_dump()
    cfg["model"]["potency"]["xgb"]["max_dept"] = 9
    with pytest.raises(ValidationError, match="max_dept"):
        Config.model_validate(cfg)


def test_the_selectivity_comparator_is_not_configurable():
    """`model.potency.xgb` is the potency model's. The potency-subtract baseline in
    medchem.models.selectivity is fixed on purpose: it is what the direct-Δ model is measured
    against, and a config able to weaken the comparator could improve the result by lowering the
    bar. The two specifications differ (600 trees vs 400), so this pins the intent."""
    from medchem.models.selectivity import _XGB_BASELINE

    assert _XGB_BASELINE["n_estimators"] == 400
    assert load_config(REPO / "configs" / "jak1.yaml").model.potency.xgb.n_estimators == 600


def test_sampler_must_name_an_implemented_sampler():
    """The stage tests `== "mock"` and sends everything else to REINVENT4, so a free string
    selected the GPU sampler by typo."""
    for bad in ("mockk", "reinvent", "reinvent-4"):
        with pytest.raises(ValidationError):
            Config.model_validate({"generative": {"sampler": bad}})
    assert Config.model_validate({"generative": {"sampler": "reinvent4"}}).generative.sampler


def test_a_scoring_component_the_scorer_does_not_compute_is_rejected():
    """The quietest failure in the file, and quieter than an earlier version of this docstring claimed.

    `score_components` scores a missing component 0.0, and `aggregate` then floors every component at
    1e-9 so that one zero cannot annihilate the geometric mean. So a misspelled name does NOT zero the
    reward: it makes that objective inert. Every candidate gets the same floored constant for it, the
    ranking is decided by the remaining components, and the report still lists the objective as scored.
    Measured on this very spec, the misspelled ranking equals the ranking with the component deleted.
    """
    cfg = load_config(REPO / "configs" / "jak1.yaml").model_dump()
    cfg["generative"]["scoring"]["components"][0]["name"] = "qsar_pic_50"
    with pytest.raises(ValidationError, match="does not compute"):
        Config.model_validate(cfg)


def test_scorer_component_names_are_introspected_not_duplicated():
    """The validator reads the scorer's own constant, so a component added there needs no edit
    here — and one removed there immediately invalidates configs that still score it."""
    from medchem.reward_components import SCORED_COMPONENTS

    components = load_config(REPO / "configs" / "jak1.yaml").generative.scoring.components
    assert components, "the shipped config must declare its scoring spec; there is no default to fall back on"
    shipped = {c["name"] for c in components}
    assert shipped <= set(SCORED_COMPONENTS)
    assert "applicability_domain" in SCORED_COMPONENTS


def test_the_dead_envelope_potency_quantile_is_now_rejected():
    """It duplicated `vls.actives.pic50_quantile`, which is the key the derivation actually uses.
    A config setting it got the other key's cut, and the envelope record named that one."""
    with pytest.raises(ValidationError, match="potency_quantile"):
        Config.model_validate({"vls": {"library": {"lead_like": {
            "derive": {"potency_quantile": 0.9}}}}})


def test_explicit_empty_descriptors_survives_as_a_request():
    """`descriptors: []` asks for fingerprint bits only. It must not come back as the nine
    standard descriptors, which is what a `[]` default plus `configured or DEFAULTS` produced."""
    cfg = Config.model_validate({"features": {"descriptors": []}})
    assert cfg.features.descriptors == []
    assert Config.model_validate({"features": {}}).features.descriptors is None


def test_no_consumer_substitutes_a_default_for_a_config_derived_value():
    """The regression guard for every "empty request answered with a default" defect in this project.

    All of them had one shape: a consumer writing ``configured or <default>``, which cannot distinguish
    ``[]`` from ``None``, ``0.0`` from unset, or one config value from another. Six were found this way --
    features.descriptors, model.selectivity.pairs, vls.tier1.conformal_halfwidth,
    generative.scoring.components, data.activity_types, and structure's two site-residue vocabularies.

    The logic lives in module-level helpers (``_reads_config``, ``_substantive``, ``_or_fallback_findings``)
    so that the repository scan below and the synthetic mutation test share EXACTLY one implementation. It
    did not: the synthetic proof defined its own reader regex including ``st.`` while the production regex
    omitted it, so the scan would have skipped ``st.anchor_residue or st.hinge_residue`` and the test that
    was supposed to prove otherwise proved nothing about the code that runs.
    """
    src_root = Path(__file__).resolve().parent.parent / "src" / "medchem"
    findings: list[str] = []
    for f in sorted(src_root.rglob("*.py")):
        findings += _or_fallback_findings(f.read_text(encoding="utf-8"),
                                         str(f.relative_to(src_root.parent.parent)))
    assert not findings, (
        "a config-derived value is being replaced by a substantive default (or by another config value), "
        "so an explicitly empty or zero request would run as though it had asked for that default:\n  "
        + "\n  ".join(findings)
    )


def test_the_guard_catches_the_former_receptor_expression_using_the_production_helpers():
    """Proof against the ACTUAL guard, not a re-implementation of it.

    ``st.anchor_residue or st.hinge_residue`` is the expression the receptor stage used, and it resolved
    two alternative vocabularies for one slot by truthiness precedence -- silently dropping the hinge when
    a config named both. The previous version of this test built its own regex, which happened to include
    ``st.`` while the production one did not, so it passed while the real scan skipped the defect.
    """
    src = "def f(st):\n    return st.anchor_residue or st.hinge_residue\n"
    assert _or_fallback_findings(src, "synthetic.py"), (
        "the production helpers do not catch the config-vs-config shape they were widened for"
    )

    # And the normalising idiom must still be allowed: `or ""` / `or {}` overrides no request.
    for ok in ('def f(st):\n    return str(st.primary or "")\n',
               "def f(ctx):\n    return dict(ctx.config.vls.known_reference or {})\n"):
        assert not _or_fallback_findings(ok, "synthetic.py"), f"false positive on: {ok.strip()!r}"


def test_the_two_known_default_substitutions_are_gone_from_their_consumers():
    """Names the two the independent review found, so the guard above cannot silently stop covering them.

    Asserted against the CODE, with comments stripped by round-tripping through the AST. Both files
    deliberately describe the removed defaults in prose -- ``stage.py`` keeps a comment where the
    constant used to be, so the next person does not helpfully restore it -- and a check that read the
    raw text would be tripped by the very documentation that prevents the regression.
    """
    import ast

    src_root = Path(__file__).resolve().parent.parent / "src" / "medchem"

    def code_only(rel: str) -> str:
        return ast.unparse(ast.parse((src_root / rel).read_text(encoding="utf-8")))

    stage = code_only("generative/stage.py")
    assert "_SPEC" not in stage, (
        "the JAK1-shaped default scoring spec is back in the generative stage; injecting it for another "
        "target silently optimises a kinase's objective"
    )
    # Exact expression, not the bare string: this module's docstring legitimately says "IC50-only is
    # enforced", and ast.unparse keeps docstrings (they are AST nodes) while dropping comments.
    curate = code_only("data/curate.py")
    assert "or ['IC50']" not in curate, (
        "curate is defaulting activity_types again with `... or ['IC50']`; config.py owns that default "
        "and now rejects the empty list rather than silently substituting it"
    )
    assert "activity_types" in curate, "curate no longer reads the field at all"
    # And the fail-closed branch is present rather than merely the default being absent.
    assert "generative.scoring.components" in stage and "raise ValueError" in stage


def test_the_vls_screen_honours_an_empty_descriptor_list():
    """The consumer that was missed, pinned by behaviour rather than by source.

    `screen_library` feeds its matrix to the model `featurize` trained. If featurize honours
    `descriptors: []` (2048 columns) and the screen substitutes nine, the model is handed 2057 columns
    and raises from inside sklearn, thousands of compounds into a screen, with nothing naming the config.
    """
    import numpy as np

    from medchem.vls.screen import screen_library

    recs = [{"smiles": "CCO", "id": "c0"}, {"smiles": "c1ccccc1O", "id": "c1"}]
    seen: list[int] = []

    def predict(x):
        seen.append(x.shape[1])
        return np.zeros(len(x))

    train_fp = np.zeros((2, 64), dtype=np.float32)
    screen_library(recs, potency_predict=predict, selectivity_predict=None, train_fp=train_fp,
                   n_bits=64, descriptors=[])
    assert seen and all(w == 64 for w in seen), (
        f"descriptors=[] must give fingerprint bits only; the screen featurised {seen} columns"
    )
    seen.clear()
    screen_library(recs, potency_predict=predict, selectivity_predict=None, train_fp=train_fp,
                   n_bits=64, descriptors=None)
    assert seen and all(w == 64 + 9 for w in seen), (
        f"descriptors=None must give the standard nine; the screen featurised {seen} columns"
    )


def test_explicit_empty_selectivity_pairs_is_not_the_derived_panel():
    """`pairs: []` asks for no selectivity modelling. Read with `or`, it became every
    primary-comparator combination — the opposite request."""
    cfg = Config.model_validate({
        "data": {"primary": "JAK1", "targets": {"JAK1": "CHEMBL2835", "JAK2": "CHEMBL2971"}},
        "model": {"selectivity": {"pairs": []}},
    })
    assert cfg.model.selectivity.pairs == []
    assert Config.model_validate({}).model.selectivity.pairs is None


def test_a_zero_conformal_halfwidth_is_a_value_not_an_absence():
    """0.0 asks for point predictions with no uncertainty band. `configured or measured` replaced
    it with this run's measured half-width and the ablation ran unablated."""
    cfg = Config.model_validate({"vls": {"tier1": {"conformal_halfwidth": 0.0}}})
    assert cfg.vls.tier1.conformal_halfwidth == 0.0
    assert Config.model_validate({}).vls.tier1.conformal_halfwidth is None


def test_a_negative_conformal_halfwidth_is_rejected():
    """A negative band inverts every interval comparison in the tier logic and still produces a
    full, plausible strata census."""
    with pytest.raises(ValidationError):
        Config.model_validate({"vls": {"tier1": {"conformal_halfwidth": -0.1}}})


def test_the_evaluation_harness_evaluates_the_configured_xgb():
    """`qsar` trained the configured estimator and `evaluate` trained a literal copy of its
    defaults, so a tuned config produced two different XGBs and one name for both.

    Checked by reading the source rather than by running a panel: the point is that the five values
    are not written twice, and a test that only compared outputs would pass while the duplication
    sat there waiting for the first config to change one of them.
    """
    import ast
    import inspect

    from medchem.eval import harness

    src = inspect.getsource(getattr(harness.evaluate, "fn", harness.evaluate))
    call = next(n for n in ast.walk(ast.parse(inspect.cleandoc(src)))
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "XGBRegressor")
    tuned = {k.arg: k.value for k in call.keywords if k.arg in
             {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"}}
    assert len(tuned) == 5
    for name, node in tuned.items():
        assert isinstance(node, ast.Attribute) and node.attr == name, (
            f"{name} is a literal in the evaluation harness; it must come from model.potency.xgb"
        )


def test_the_optional_leakage_gate_is_reachable():
    """The harness offered `leakage_max_exact_dupes` as an optional gate and tested for it with a
    dict membership check — against a strict model that had no such field. No config could set it,
    so the branch never ran. Declaring it is what makes the offer real."""
    assert Config().eval.gates.leakage_max_exact_dupes is None
    cfg = Config.model_validate({"eval": {"gates": {"leakage_max_exact_dupes": 0}}})
    assert cfg.eval.gates.leakage_max_exact_dupes == 0


# ---------------------------------------------------------------------------------------------
# The two fail-open paths independent review found after three earlier passes. Both had the same
# shape as the ones already fixed, and both were results-determining: one chooses what "better"
# means for a target, the other chooses which measurements exist at all.
# ---------------------------------------------------------------------------------------------

def _generative_ctx(tmp_path, cfg):
    """A StageContext with the generative stage's three required upstreams satisfied.

    The upstream artifacts are real but tiny; the point is to reach the spec check with a config, not
    to score anything, so the stage must fail before any model is loaded.
    """
    from medchem.pipeline.stage import StageContext, StageResult

    return StageContext(
        config=cfg, workdir=str(tmp_path),
        upstream={
            "curate": StageResult(name="curate", outputs={"potency_training": "x.csv"}, metrics={}),
            "featurize": StageResult(name="featurize", outputs={"features": "f.npz"}, metrics={}),
            "qsar": StageResult(name="qsar", outputs={"model": "m.joblib"}, metrics={}),
        },
    )


def test_generative_refuses_to_run_with_omitted_components(tmp_path):
    """Omitted used to mean "use the module's JAK1-shaped seven-component spec"."""
    from medchem.generative.stage import generative

    cfg = Config.model_validate({"target": "brd4"})
    assert cfg.generative.scoring.components is None
    fn = getattr(generative, "fn", generative)
    with pytest.raises(ValueError, match="no generative.scoring.components at all"):
        fn(_generative_ctx(tmp_path, cfg))


def test_generative_refuses_to_run_with_explicit_empty_components(tmp_path):
    """`components: []` asked for no reward and received the most target-specific default in the file."""
    from medchem.generative.stage import generative

    cfg = Config.model_validate({"target": "brd4", "generative": {"scoring": {"components": []}}})
    assert cfg.generative.scoring.components == []
    fn = getattr(generative, "fn", generative)
    with pytest.raises(ValueError, match=r"components: \[\] -- an empty specification"):
        fn(_generative_ctx(tmp_path, cfg))


def test_a_new_target_cannot_inherit_another_targets_reward(tmp_path):
    """The scientific form of the defect: a bromodomain scored on a kinase's objective.

    The deleted default centred `qsar_pic50` at 7.5 and windowed MW to 300-450. JQ1, the BET
    benchmark, sits near 7.1 and BET chemotypes routinely exceed 450 Da, so the default rewarded the
    wrong end of both ranges — silently, because nothing in the output recorded which spec was used.
    """
    from medchem.generative import stage as gen_stage

    cfg = Config.model_validate({
        "target": "some_new_target",
        "data": {"targets": {"ACME1": "CHEMBL1"}, "primary": "ACME1"},
    })
    fn = getattr(gen_stage.generative, "fn", gen_stage.generative)
    with pytest.raises(ValueError, match="some_new_target"):
        fn(_generative_ctx(tmp_path, cfg))


def test_a_valid_nonempty_component_spec_passes_the_check(tmp_path):
    """The check must not simply refuse everything: a real spec reaches the model load and fails there.

    Reaching a missing-artifact error rather than the spec error is the proof that the spec was accepted.
    """
    from medchem.generative.stage import generative

    cfg = load_config(REPO / "configs" / "jak1.yaml")
    assert cfg.generative.scoring.components
    fn = getattr(generative, "fn", generative)
    with pytest.raises(Exception) as exc:
        fn(_generative_ctx(tmp_path, cfg))
    assert "generative.scoring.components" not in str(exc.value), (
        f"a valid spec was rejected by the spec check: {exc.value}"
    )


def test_empty_activity_types_is_rejected_at_validation():
    """`activity_types: []` admits no measurement, so it cannot be honoured — and must not be defaulted.

    curate read it as `configured or ["IC50"]`, so the one value that cannot work silently became the
    documented default and the run curated IC50 while the config asked for nothing.
    """
    with pytest.raises(ValidationError, match="data.activity_types is empty"):
        Config.model_validate({"data": {"activity_types": []}})
    with pytest.raises(ValidationError, match="data.activity_types is empty"):
        DataConfig(activity_types=[])


def test_omitted_activity_types_keeps_the_documented_default():
    assert Config().data.activity_types == ["IC50"]
    assert load_config(REPO / "configs" / "jak1.yaml").data.activity_types == ["IC50"]


def test_curate_consumes_the_validated_activity_types_verbatim(tmp_path, monkeypatch):
    """Consumer-level: whatever validation produced is what curation accepts, with no second default."""
    import ast

    src = Path(__file__).resolve().parent.parent / "src" / "medchem" / "data" / "curate.py"
    code = ast.unparse(ast.parse(src.read_text(encoding="utf-8")))
    assert "accept_types = list(cfg_data.activity_types)" in code, (
        "curate no longer reads the validated value directly; a transformation here is where a second "
        "default would reappear"
    )
    cfg = Config.model_validate({"data": {"activity_types": ["IC50", "Ki"]}})
    assert cfg.data.activity_types == ["IC50", "Ki"]
