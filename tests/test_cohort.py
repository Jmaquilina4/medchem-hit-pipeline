"""The frozen cohort spec: precedence, exclusions, and attrition accounting.

Two axes were measured on this project's data and both must be removed together. Domain: one phase-3
reference compound reads 5.85 on one bromodomain and 6.88 on the other. Format: JAK1's biochemical and
cell-based medians sit 0.45 log units apart. A cohort that fixes one and pools the other has traded one
ambiguity for another, which is why the headline cohorts require both conditions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from medchem.cohorts import (
    COHORT_SPEC_VERSION,
    FROZEN_COHORTS,
    NO_DESCRIPTION,
    UNMATCHED,
    apply_to_activities,
    classify_description,
    select_assays,
)


def test_cell_based_wins_over_a_domain_mention():
    """PRECEDENCE, and the most consequential rule. A cell assay against a BD1 construct is still a
    cell measurement -- its potency is governed by permeability, not domain affinity. Letting the BD1
    mention claim it would reintroduce the 0.45-log-unit format heterogeneity the spec removes."""
    assert classify_description("Inhibition of BRD4 BD1 in MV4-11 cell proliferation") == "cell_based"
    assert classify_description("Antiproliferative activity against HeLa cells") == "cell_based"


def test_domain_is_assigned_on_biochemical_assays():
    assert classify_description("Inhibition of BRD4 BD1 by TR-FRET") == "domain_1"
    assert classify_description("Binding affinity at the second bromodomain of BRD4") == "domain_2"


def test_biochemical_is_the_catch_all_for_enzymatic_assays():
    assert classify_description("Inhibition of recombinant JAK1 enzyme activity") == "biochemical"
    assert classify_description("Inhibition of JAK1 using HTRF") == "biochemical"


def test_unlabelled_and_empty_descriptions_are_never_folded_in_silently():
    assert classify_description("") == NO_DESCRIPTION
    assert classify_description(None) == NO_DESCRIPTION
    assert classify_description("nan") == NO_DESCRIPTION
    assert classify_description("Activity against the target") == UNMATCHED


def test_headline_bet_cohort_requires_domain_AND_format():
    """The refinement that matters: BD1-explicit alone still pools cell-based measurements."""
    assays = {
        "A1": "Inhibition of BRD4 BD1 by TR-FRET",                  # domain_1  -> admitted
        "A2": "Inhibition of BRD4 BD1 in MV4-11 cells",             # cell_based -> excluded
        "A3": "Inhibition of BRD4 BD2 by TR-FRET",                  # domain_2  -> excluded
        "A4": "Inhibition of recombinant BRD4 enzyme",              # biochemical, no domain -> excluded
        "A5": "Activity against BRD4",                              # unmatched -> excluded
    }
    sel = select_assays(assays, "domain1_noncellular_explicit")
    assert sel.admitted == {"A1"}
    assert set(sel.excluded) == {"A2", "A3", "A4", "A5"}
    assert all("excluded by cohort" in r or "not in the cohort" in r for r in sel.excluded.values())


def test_jak1_headline_cohort_drops_cell_based_and_unmatched():
    assays = {
        "B1": "Inhibition of recombinant JAK1 enzyme activity",     # biochemical -> admitted
        "B2": "Inhibition of JAK1 in whole blood",                  # cell_based -> excluded
        "B3": "Activity at JAK1",                                   # unmatched -> excluded
        "B4": "",                                                    # no_description -> excluded
    }
    sel = select_assays(assays, "biochemical_explicit")
    assert sel.admitted == {"B1"}
    assert len(sel.excluded) == 3


def test_target_associated_cohort_admits_everything_including_unmatched():
    """The historical baseline must remain reproducible, pooling and all."""
    assays = {"C1": "Activity at target", "C2": "", "C3": "Inhibition in cells"}
    sel = select_assays(assays, "target_associated")
    assert sel.admitted == {"C1", "C2", "C3"} and not sel.excluded


def test_unknown_cohort_is_refused_rather_than_improvised():
    with pytest.raises(ValueError, match="unknown cohort"):
        select_assays({}, "bd1_ish")


def test_selection_records_exact_assay_ids_and_the_spec_version():
    """A cohort must be reproducible from the manifest without re-deriving it from regexes that may
    later change."""
    sel = select_assays({"A1": "Inhibition of BRD4 BD1 by TR-FRET"},
                        "domain1_noncellular_explicit")
    d = sel.as_dict()
    assert d["spec_version"] == COHORT_SPEC_VERSION
    assert d["admitted_assay_ids"] == ["A1"]
    assert "label_counts" in d and "exclusion_reasons" in d


def test_activity_and_compound_attrition_are_reported_separately():
    """A cohort can keep most assays while dropping most measurements; neither number implies the
    other, so both are recorded."""
    sel = select_assays({"A1": "Inhibition of BRD4 BD1 by TR-FRET",
                         "A2": "Inhibition of BRD4 BD1 in cells"}, "domain1_noncellular_explicit")
    acts = pd.DataFrame({
        "assay_chembl_id": ["A1", "A2", "A2", "A2", "UNKNOWN"],
        "canonical_smiles": ["CCO", "CCO", "c1ccccc1", "CCN", "CCC"],
    })
    kept, attr = apply_to_activities(acts, sel)
    assert attr["activities_in"] == 5 and attr["activities_out"] == 1
    assert attr["compounds_in"] == 4 and attr["compounds_out"] == 1
    assert attr["activities_with_unknown_assay"] == 1, "an activity citing an unseen assay must be counted"
    assert list(kept["assay_chembl_id"]) == ["A1"]


def test_missing_assay_identity_is_an_error_not_a_pass_through():
    """Without assay identity a cohort cannot be applied, and silently returning everything would be
    the exact failure this module exists to prevent."""
    sel = select_assays({"A1": "x"}, "target_associated")
    with pytest.raises(ValueError, match="no 'assay_chembl_id' column|has no"):
        apply_to_activities(pd.DataFrame({"canonical_smiles": ["CCO"]}), sel)


def test_all_four_frozen_cohorts_are_defined():
    """The run plan names four evaluations; three cohort definitions cover them (BRD4 and JAK1 share
    the pooled baseline)."""
    assert set(FROZEN_COHORTS) == {
        "biochemical_explicit", "domain1_bd1_explicit_structured_binding",
        "domain1_noncellular_explicit", "target_associated"}
    for name, rule in FROZEN_COHORTS.items():
        assert rule["description"], f"{name} has no human-readable description"


def test_spec_1_1_rename_is_behaviourally_identical():
    """The 1.0 -> 1.1 rename must change the NAME and nothing else.

    This is what licenses leaving the frozen results in place: if the deprecated name selected even one
    different assay, the rename would be a silent reclassification and every artifact produced under
    spec 1.0 would need re-deriving.
    """
    from medchem.cohorts import resolve_cohort

    descriptions = {
        "A1": "Inhibition of BRD4 BD1 by TR-FRET assay",
        "A2": "Inhibition of BRD4 BD2 using recombinant enzyme",
        "A3": "Inhibition of BRD4 BD1 in HeLa cells assessed as cell proliferation",
        "A4": "Binding affinity to full length BRD4 by ITC",
        "A5": "Displacement assay against first bromodomain of BRD4",
        "A6": "Some assay with no recognisable format or domain wording whatsoever",
    }
    old = select_assays(descriptions, "domain1_biochemical_explicit")
    new = select_assays(descriptions, "domain1_noncellular_explicit")

    assert resolve_cohort("domain1_biochemical_explicit") == "domain1_noncellular_explicit"
    assert old.admitted == new.admitted == {"A1", "A5"}
    assert old.assay_labels == new.assay_labels
    assert old.label_counts == new.label_counts
    # The selection records the name the caller passed, so an artifact states what actually ran.
    assert old.cohort == "domain1_biochemical_explicit"
    assert new.cohort == "domain1_noncellular_explicit"


def test_renamed_cohort_does_not_claim_biochemical_confirmation():
    """Guards the wording itself. The rules exclude cell-based patterns; they never positively confirm a
    biochemical assay, and an audit of ChEMBL's structured fields found four admitted assays that fail
    an independent check. A description that says otherwise is a claim the code cannot support."""
    from medchem.cohorts import DESCRIPTION_LABEL_KNOWN_EXCEPTIONS

    rule = FROZEN_COHORTS["domain1_noncellular_explicit"]
    text = f"{rule['description']} {rule['display_name']}".lower()
    assert "non-cellular" in text
    assert "not independently confirmed biochemical" in rule["description"].lower()
    assert len(DESCRIPTION_LABEL_KNOWN_EXCEPTIONS) == 4


# ---------------------------------------------------------------------------------------------------
# Cohort spec 1.2: the headline BET cohort is fail-closed on ChEMBL's structured fields.
# ---------------------------------------------------------------------------------------------------

_STRUCT_CLEAN = {"bao_format": "BAO_0000357", "assay_type": "B"}


def test_spec_1_2_requires_structured_fields_and_refuses_to_guess():
    """The whole point: a cohort that needs structured metadata must RAISE without it, not fall back.

    A silent fallback to the description-only decision is precisely the defect 1.2 closes, and a
    fallback would be invisible in the results -- the run would simply admit more assays.
    """
    with pytest.raises(ValueError, match="requires ChEMBL structured fields"):
        select_assays({"A1": "Inhibition of BRD4 BD1 by TR-FRET assay"},
                      "domain1_bd1_explicit_structured_binding")


def test_spec_1_2_excludes_the_four_impurity_classes_with_distinct_reasons():
    """Each exclusion class gets its OWN recorded reason, so the manifest shows the mix rather than a
    lump of 'excluded by cohort'."""
    descriptions = {
        "OK":     "Inhibition of BRD4 BD1 by TR-FRET assay",
        "DUAL":   "Inhibition of BRD4 BD1/BD2 (unknown origin) by TR-FRET assay",
        "DUAL2":  "Binding affinity to BRD4 bromodomain 1 and 2 long isoform",
        "CELL":   "Inhibition of BRD4 BD1 by dual luciferase reporter gene assay",
        "AMBIG":  "Inhibition of BRD4 BD1 by an unspecified method",
        "WRONGT": "Binding affinity to BRD4 BD1 by isothermal calorimetric titration",
        "NOMETA": "Inhibition of BRD4 BD1 by TR-FRET assay",
    }
    structured = {
        "OK": _STRUCT_CLEAN,
        "DUAL": _STRUCT_CLEAN,
        "DUAL2": {"bao_format": "BAO_0000019", "assay_type": "B"},
        "CELL": {"bao_format": "BAO_0000219", "assay_type": "B"},
        "AMBIG": {"bao_format": "BAO_0000019", "assay_type": "B"},
        "WRONGT": {"bao_format": "BAO_0000357", "assay_type": "A"},
        # NOMETA deliberately absent
    }
    sel = select_assays(descriptions, "domain1_bd1_explicit_structured_binding", structured=structured)

    assert sel.admitted == {"OK"}
    assert "tandem-domain" in sel.excluded["DUAL"]
    assert "tandem-domain" in sel.excluded["DUAL2"]
    assert "cell-based format" in sel.excluded["CELL"]
    assert "ambiguous" in sel.excluded["AMBIG"]
    assert "conflicts with a binding cohort" in sel.excluded["WRONGT"]
    assert "structured metadata absent" in sel.excluded["NOMETA"]
    # Reasons must be distinguishable, not interchangeable.
    assert len({sel.excluded[k] for k in ("DUAL", "CELL", "AMBIG", "WRONGT", "NOMETA")}) == 5


def test_spec_1_2_leaves_the_other_three_cohorts_untouched():
    """1.2 adds a cohort; it must not silently reclassify the other three.

    Asserted rather than assumed, because a global spec-version bump re-runs every panel: if the pooled
    or JAK1 cohorts changed too, three 'replayed' analyses would quietly be new results.
    """
    descriptions = {
        "A1": "Inhibition of JAK1 by recombinant enzyme assay",
        "A2": "Inhibition of JAK1 in HeLa cells assessed as proliferation",
        "A3": "Inhibition of BRD4 BD1/BD2 by TR-FRET assay",
        "A4": "Binding affinity to full length BRD4 by ITC",
        "A5": "Some assay with no recognisable format or domain wording",
    }
    # These cohorts declare no structured requirement, so they must behave identically with and
    # without structured metadata supplied.
    struct = dict.fromkeys(descriptions, _STRUCT_CLEAN)
    for name, expected in (
        ("biochemical_explicit", {"A1", "A3", "A4"}),
        ("target_associated", set(descriptions)),
    ):
        bare = select_assays(descriptions, name)
        withmeta = select_assays(descriptions, name, structured=struct)
        assert bare.admitted == withmeta.admitted == expected, name
        assert bare.assay_labels == withmeta.assay_labels


def test_dual_domain_regex_does_not_fire_on_a_bd1_only_description():
    """A guard against over-exclusion: fail-closed must not mean fail-always."""
    from medchem.cohorts import DUAL_DOMAIN_RE

    for ok in ("Inhibition of BRD4 BD1 by TR-FRET assay",
               "Binding affinity to BRD4 bromodomain 1 (unknown origin)",
               "Inhibition of first bromodomain of BRD4 at 1/2 log dilution"):
        assert not DUAL_DOMAIN_RE.search(ok), ok
    for bad in ("BRD4 BD1/BD2 Y390A mutant", "bromodomain 1 and 2 long isoform",
                "BRD4-BD1/2", "both bromodomains of BRD4", "BRD4 (BD1+BD2) bromodomain"):
        assert DUAL_DOMAIN_RE.search(bad), bad


def test_superseded_cohort_is_marked_in_the_source_not_in_its_strings():
    """The spec-1.1 cohort is still selectable and must be visibly marked superseded — but in a COMMENT.

    Putting the marker in the ``description`` value instead moved the scientific-source digest, because
    dict values are AST constants while comments are not. That would have reintroduced an
    analysis-vs-release digest delta needing explanation, to convey information that changes no result.
    So the warning lives where it is free, and this test pins it there.
    """
    src = (Path(__file__).resolve().parent.parent / "src" / "medchem" / "cohorts.py").read_text()
    i = src.index('"domain1_noncellular_explicit": {')
    # The marker sits at the TOP OF THE DEFINITION, which is what a reader opening it sees first.
    block = src[i:i + 1400]
    assert "SUPERSEDED" in block, "the superseded marker must head the cohort definition"
    assert "domain1_bd1_explicit_structured_binding" in block, "it must name its replacement"

    rule = FROZEN_COHORTS["domain1_noncellular_explicit"]
    assert "require_structured" not in rule, "the superseded cohort is text-only by definition"
    assert "require_structured" in FROZEN_COHORTS["domain1_bd1_explicit_structured_binding"]


def test_the_deprecated_cohort_key_still_resolves_and_selects_identically():
    """The old key asserted the structured fields confirmed the BD1 DOMAIN. They confirm the FORMAT.

    Renaming it is a correction to a claim, not to behaviour, so the alias must keep resolving and must
    select exactly the same assays -- otherwise a rename would silently become a scientific change.
    """
    from medchem.cohorts import COHORT_ALIASES, FROZEN_COHORTS

    old, new = "domain1_biochemical_confirmed", "domain1_bd1_explicit_structured_binding"
    assert COHORT_ALIASES[old] == new, "the deprecated key must resolve to the truthful one"
    assert old not in FROZEN_COHORTS, "the deprecated key must not be a live definition"

    spec = FROZEN_COHORTS[new]
    # The name may not claim the structured fields established the domain.
    for field in ("description", "display_name"):
        text = spec[field].lower()
        assert "structurally confirmed" not in text, f"{field} still claims structural confirmation"
        assert "confirmed biochemical" not in text, f"{field} still claims confirmed biochemistry"
    assert "format" in spec["description"].lower(), (
        "the description must say the structured fields confirm the assay FORMAT"
    )
    assert "description" in spec["description"].lower(), (
        "the description must say BD1 identity comes from the assay DESCRIPTION"
    )


def test_no_shipped_config_label_claims_structural_confirmation():
    """The correction has to reach the LABEL, not only the spec.

    ``data.assay_cohort.label`` is what captions the figures and heads the tables, and it is written
    per config rather than derived — so renaming the cohort key left ``configs/brd4.yaml`` still
    announcing a "structurally confirmed" cohort under the corrected key. A reader sees the label,
    not the key.
    """
    import yaml

    repo = Path(__file__).resolve().parent.parent
    for cfg_path in sorted((repo / "configs").glob("*.yaml")):
        cohort = ((yaml.safe_load(cfg_path.read_text()) or {}).get("curation") or {}).get(
            "assay_cohort") or {}
        label = str(cohort.get("label", "")).lower()
        assert "structurally confirmed" not in label, f"{cfg_path.name}: {label!r}"
        assert "confirmed biochemical" not in label, f"{cfg_path.name}: {label!r}"
