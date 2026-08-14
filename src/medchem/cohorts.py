"""Frozen, versioned assay-cohort definitions.

A single-protein ChEMBL target is not one assay, and treating it as one silently pools measurements
that disagree. Two independent axes were measured on this project's own data before this module
existed:

* **domain** — BRD4 IC50 *records* are 47.7% first-bromodomain-explicit, 25.0% biochemical with no
  domain stated, 15.8% cell-based, 6.3% second-bromodomain, 5.1% unmatched. One phase-3 reference
  compound reads **5.85** on the first domain against **6.88** on the second — 1.03 log units on the
  same molecule — so a median across both represents neither.
* **assay format** — JAK1 records are 70.5% biochemical and 25.0% cell-based, and their curated medians
  sit **0.45 log units** apart (7.75 vs 7.30). The DIRECTION is not systematic and this docstring used to
  claim it was: on BRD4 the cell-based median is *higher* than the biochemical one (6.51 vs 6.07), the
  reverse of JAK1. Mechanisms that weaken cell potency (permeability, ATP competition, protein binding)
  are real, but so are ones that raise apparent potency, and these are two population medians over
  largely different compound sets rather than paired measurements. What the figures support is that the
  formats DIFFER, enough that pooling them makes a label depend on the mix — which is the reason cohorts
  exist. Which way they differ is a per-target empirical question.

Both sets of figures are DERIVED, not asserted: ``scripts/derive_composition.py`` recomputes them from
the hash-verified raw inputs using ``curate_activities`` itself, and writes them to
``provenance/*/composition.json``. Earlier drafts of this docstring quoted an audit run before the
precedence below was frozen, and its numbers were wrong in both directions — 11% second-bromodomain
against a true 6.3%, and a JAK1 gap of 1.08 log units against a true 0.45. The lesson is in the
tooling, not the correction: an audit nobody re-runs is a number with no owner.

Removing one axis and leaving the other is not enough: a domain-matched cohort that still pools
biochemical and cell-based measurements has simply traded one ambiguity for another. Headline cohorts
therefore require BOTH conditions.

**Placement.** This sits at the package root rather than under ``data/`` because two different layers
need it: ``config`` validates that a requested cohort exists, and ``curate`` applies it. Living under
``data/`` made ``config`` import from an outer layer — the same violation the score transforms had, and
the architecture test catches it immediately.

**This file is the frozen specification.** ``COHORT_SPEC_VERSION`` is recorded in every artifact that
depends on it, so a change to these rules is visible as a version change rather than as numbers that
quietly moved. Precedence is explicit and ordered, because a description mentioning both a domain and a
cell line must land in exactly one bucket and which one it is cannot be left to dictionary order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

COHORT_SPEC_VERSION = "1.2"

# ---------------------------------------------------------------------------------------------------
# 1.1 -> 1.2: the headline BET cohort becomes FAIL-CLOSED on ChEMBL's own structured fields.
#
# Spec 1.1 renamed the cohort honestly but left it defined by DESCRIPTION TEXT alone, and an audit of
# the structured metadata showed why that is not enough:
#
#   * 23 of 704 admitted assays name BOTH bromodomains. The precedence below evaluates `domain_1`
#     before `domain_2`, and "BD1/BD2" contains "BD1", so a tandem-domain construct was silently
#     labelled first-bromodomain. FIFTEEN of those 23 also carry clean structured metadata, so no
#     structured check alone would have caught them -- the description rule is load-bearing.
#   * one assay carried a CELL-BASED BAO format ("Inhibition of BRD4 BD1 in HUVEC ... dual luciferase
#     reporter gene assay"): a cell measurement the text rules admitted.
#   * three carried `assay_type` A against a binding cohort -- conflicting metadata.
#   * 178 carried BAO_0000019, the ROOT "assay format" term, which asserts nothing about whether the
#     measurement is biochemical or cellular. Uninformative, not benign.
#
# 1.2 therefore requires POSITIVE confirmation rather than absence of a disqualifier, and excludes
# ambiguous or conflicting metadata with a recorded reason instead of admitting it. The cost is real
# and measured, not waved away: it is recorded per run in the manifest's attrition block.
#
# The three other frozen cohorts are UNCHANGED by 1.2. Their classification is identical, which is
# verified rather than asserted: tests/test_cohort.py checks that the pooled and JAK1 biochemical
# cohorts select the same assays under 1.1 and 1.2.
# ---------------------------------------------------------------------------------------------------

# BAO format terms that POSITIVELY identify a non-cellular, single-protein measurement.
# Deliberately a tiny allowlist: anything not named here is excluded, including the root term.
BIOCHEMICAL_BAO_FORMATS: frozenset[str] = frozenset({
    "BAO_0000357",      # single protein format
})
# Non-single-protein BAO formats, each NAMED FOR WHAT IT IS. They were previously lumped together and
# all reported as "cell-based format", which is wrong for three of the four: it does not change any
# admission decision (that is driven by the allowlist above) but it would mislabel a future exclusion
# reason, and an exclusion reason nobody can trust is not an audit trail.
# Terms per the BioAssay Ontology: https://www.ebi.ac.uk/ols4/ontologies/bao
NON_SINGLE_PROTEIN_BAO_FORMATS: dict[str, str] = {
    "BAO_0000219": "cell-based format",
    "BAO_0000218": "organism-based format",
    "BAO_0000220": "subcellular format",
    "BAO_0000221": "tissue-based format",
}
# Retained name, for callers and tests that ask "is this a non-protein format".
CELL_BASED_BAO_FORMATS: frozenset[str] = frozenset(NON_SINGLE_PROTEIN_BAO_FORMATS)
# ChEMBL assay_type codes acceptable for a binding cohort. 'A' (ADMET) and 'F' (functional) are not,
# even when the description reads as biochemical -- a conflict between fields is a reason to exclude,
# not a reason to pick the field that agrees with us.
BINDING_ASSAY_TYPES: frozenset[str] = frozenset({"B"})

# A construct spanning both bromodomains measures neither one alone. Matched on the description
# because the structured fields do not encode domain identity at all.
DUAL_DOMAIN_RE = re.compile(
    r"bd1\s*[/&+]\s*bd2|bd\s*1\s*(?:and|&|/|\+)\s*(?:bd\s*)?2|bromodomain\s*1\s*(?:and|&|/|\+)\s*2|"
    r"both\s+bromodomains?|\bbd1/2\b|\bbd\s*1/2\b|tandem\s+bromodomain|dual\s+bromodomain",
    re.I,
)

# 1.0 -> 1.1 is a RENAME ONLY. `domain1_biochemical_explicit` claimed more than the rules deliver: the
# precedence below evaluates `cell_based` FIRST, so an assay labelled `domain_1` is only known to be
# *non-cellular by description*, never independently confirmed biochemical. The canonical name is now
# `domain1_noncellular_explicit`; the old name resolves to the identical definition so artifacts
# produced under spec 1.0 remain valid and addressable.
#
# `tests/test_cohort.py` asserts the two names classify byte-identically, which is what makes this a
# rename rather than a change.
#
# The alias exists for ARTIFACTS, not for the current runs. This comment used to say "the frozen runs
# recorded spec 1.0 and executed under the former name", which stopped being true two spec versions ago:
# every published manifest records spec 1.2 and the current key, and the panels have been re-run from
# this source since. What the alias still buys is that an older artifact or a third-party config naming
# the old key resolves rather than failing.
COHORT_ALIASES: dict[str, str] = {
    "domain1_biochemical_explicit": "domain1_noncellular_explicit",
    # DEPRECATED. The old key asserted that ChEMBL's structured fields confirmed the BD1 DOMAIN; they
    # confirm the assay FORMAT. Kept so an existing config or record still resolves, and so a reader
    # meeting the old name in an older artifact can find what it became. Inclusion semantics identical.
    "domain1_biochemical_confirmed": "domain1_bd1_explicit_structured_binding",
}

# Assays that a DESCRIPTION-ONLY BD1 rule admits but an independent structured-field check rejects, found
# by auditing ChEMBL's own `assay_type` and `bao_format` against the description-based label.
#
# STATUS UNDER SPEC 1.2: all four are EXCLUDED. The headline cohort now requires a single-protein BAO
# format AND a binding assay type AND no tandem-domain construct, and each of these fails at least one --
# verified against the published manifest's admitted-assay list, where none of the four appears. This
# block used to say "the frozen results were produced with them included", which was true of spec 1.1 and
# has been false since; it is kept as the audit trail for WHY the structured requirement exists, not as a
# description of what currently ships:
#
#   CHEMBL4357128  bao_format = BAO_0000219 (cell-based format) — "Inhibition of BRD4 BD1 in HUVEC ...
#                  dual luciferase reporter gene assay". A genuine cell assay the regex admitted.
#   CHEMBL4668969  assay_type = A; construct is "BRD4 BD1/BD2 Y390A mutant (1 to 477 residue)" — spans
#   CHEMBL4809906  BOTH domains, so neither is BD1-pure.
#   CHEMBL4668935  assay_type = A but scientifically biochemical (ITC titration); a ChEMBL typing quirk.
#
# Exposure under the spec-1.1 rule that admitted them, measured not assumed: 76 of 19,601 BRD4 IC50
# records (0.39%), shifting the label of 49 of 3,753 compounds (1.3%) by a median 1.2 and up to 1.7 log
# units while leaving the cohort median at 6.26. That was the case FOR the structured requirement, and
# spec 1.2 acted on it: the published headline cohort excludes all four, so no re-run is outstanding.
# (This block previously said the effect "has NOT been measured, because measuring it requires a re-run".
# The re-runs happened; the current 2,794-compound cohort is the measurement.)
DESCRIPTION_LABEL_KNOWN_EXCEPTIONS: tuple[str, ...] = (
    "CHEMBL4357128", "CHEMBL4668935", "CHEMBL4668969", "CHEMBL4809906",
)

# ---------------------------------------------------------------------------------------------------
# PRECEDENCE. Evaluated top to bottom; the FIRST match wins. Order is a scientific decision:
#
# 1. cell_based first. A cell-based assay against a domain construct is still a cell measurement, and
#    its potency is governed by permeability and ATP competition rather than by domain affinity. Letting
#    a "BD1" mention in a cell-assay description claim it for the biochemical domain cohort would
#    reintroduce exactly the 0.45-log-unit format heterogeneity this spec exists to remove.
# 2. domain assignment next, on biochemical assays.
# 3. construct notes (full length / pseudokinase) after domain, since a domain mention is more specific.
# 4. biochemical last as a catch-all for enzymatic assays with no domain stated.
#
# Anything unmatched is labelled `unmatched` and is NEVER silently folded into a cohort.
# ---------------------------------------------------------------------------------------------------
CLASSIFICATION_RULES: tuple[tuple[str, str], ...] = (
    ("cell_based",   r"\bcell|prolifer|viabil|whole.?blood|pbmc|hela|ba/?f3|tf-?1|u937|hek|mv4|leukem|"
                     r"\bin\s+vivo\b|xenograft"),
    ("domain_1",     r"\bbd1\b|bromodomain\s*1\b|first\s+bromodomain|\bbrd4[- ]?bd1\b"),
    ("domain_2",     r"\bbd2\b|bromodomain\s*2\b|second\s+bromodomain"),
    ("full_length",  r"full.?length"),
    ("pseudokinase", r"\bjh2\b|pseudokinase"),
    ("biochemical",  r"enzym|kinase assay|recombinant|\bhtrf\b|lanthascreen|\badp\b|\batp\b|"
                     r"peptide substrate|\bfret\b|fluorescence polari|\btr-?fret\b|inhibition of "
                     r"(the )?(activity|enzyme)|\bbinding\b|\bitc\b|\bspr\b|thermal shift"),
)
_COMPILED = tuple((label, re.compile(pattern, re.I)) for label, pattern in CLASSIFICATION_RULES)

UNMATCHED = "unmatched"
NO_DESCRIPTION = "no_description"

# The four frozen cohorts. `require_all` names labels a *single* assay cannot simultaneously carry, so
# multi-condition cohorts are expressed as a set of ACCEPTABLE labels plus a required format axis. See
# `select_assays` for how the two axes are combined.
FROZEN_COHORTS: dict[str, dict[str, Any]] = {
    "biochemical_explicit": {
        "description": "enzymatic/binding assays only; cell-based and unmatched excluded",
        "accept_labels": ("biochemical", "domain_1", "domain_2", "full_length"),
        "require_domain": None,
        "exclude_labels": ("cell_based", "pseudokinase", UNMATCHED, NO_DESCRIPTION),
    },
    "domain1_bd1_explicit_structured_binding": {
        # Spec 1.2. The headline BET cohort.
        #
        # THE TWO HALVES REST ON DIFFERENT EVIDENCE, and the name now says so:
        #   * DOMAIN identity (BD1) comes from the assay DESCRIPTION. ChEMBL's structured fields do not
        #     encode which bromodomain was measured, so nothing here confirms BD1 independently.
        #   * ASSAY FORMAT is confirmed by the structured fields: a single-protein BAO format AND a
        #     binding assay type.
        #
        # The previous key and display name read "biochemical_confirmed" / "structurally confirmed
        # biochemical", which asserts the structured fields established the DOMAIN. They do not. That
        # wording was left in place because these are AST constants and changing them moves the
        # scientific-source digest -- but a name that misstates what the evidence supports is not a
        # cosmetic defect, and the panels are re-run at this source digest anyway. Inclusion semantics are
        # UNCHANGED: the accept/exclude labels and structured requirements below are byte-identical to
        # the ones that produced the frozen results.
        "description": ("first bromodomain by assay DESCRIPTION, with the assay FORMAT confirmed by "
                        "ChEMBL's structured fields: single-protein BAO format AND binding assay type "
                        "AND no tandem-domain construct. The structured fields confirm the format, not "
                        "the domain. Ambiguous or conflicting metadata is EXCLUDED with a reason."),
        "display_name": "BRD4 BD1-explicit, structured single-protein binding IC50 cohort",
        "accept_labels": ("domain_1",),
        "require_domain": "domain_1",
        "exclude_labels": ("cell_based", "domain_2", "full_length", UNMATCHED, NO_DESCRIPTION),
        "require_structured": {
            "bao_format_allow": tuple(sorted(BIOCHEMICAL_BAO_FORMATS)),
            "assay_type_allow": tuple(sorted(BINDING_ASSAY_TYPES)),
            "exclude_dual_domain_description": True,
        },
    },
    "domain1_noncellular_explicit": {
        # SUPERSEDED by `domain1_bd1_explicit_structured_binding` in spec 1.2, and retained only so artifacts
        # produced under spec 1.1 remain reproducible. Do not select it for new work: it decides on
        # description text alone, which admitted 23 tandem-domain constructs, one cell-based reporter
        # assay, and 172 assays whose BAO format asserts nothing. The measured consequence was not
        # cosmetic -- it moved BRD4 from 3,753 compounds to 2,794 and took the BRD4-BRD2 selectivity
        # pair from "supported" to an interval spanning zero.
        #
        # Renamed in spec 1.1. The former name, `domain1_biochemical_explicit`, asserted independent
        # biochemical confirmation that these rules do not provide: `domain_1` means the description
        # named the first bromodomain and did NOT match a cell-based pattern. That is exclusion, not
        # positive identification -- and an audit of ChEMBL's own structured fields found four admitted
        # assays that fail an independent check (see DESCRIPTION_LABEL_KNOWN_EXCEPTIONS).
        "description": ("first bromodomain named in the description AND no cell-based pattern matched; "
                        "non-cellular by exclusion, NOT independently confirmed biochemical"),
        "display_name": "BRD4 BD1-explicit, non-cellular IC50 cohort (description-based)",
        "accept_labels": ("domain_1",),
        "require_domain": "domain_1",
        "exclude_labels": ("cell_based", "domain_2", "full_length", UNMATCHED, NO_DESCRIPTION),
    },
    "target_associated": {
        "description": "every IC50 assigned to the target; the historical baseline, pooled",
        "display_name": "pooled target-associated IC50 cohort",
        "accept_labels": None,          # accept everything, including unmatched
        "require_domain": None,
        "exclude_labels": (),
    },
}


def resolve_cohort(name: str) -> str:
    """Map a cohort name to its CURRENT canonical key, accepting deprecated names from earlier specs.

    The target is the CURRENT canonical key, whatever the current spec version is -- this docstring used
    to name a specific earlier version, which stopped being true at the next rename. Deprecated names
    from every earlier spec resolve here.

    Kept deliberately small and total: an unknown name is returned unchanged so the caller's own
    validation produces the error, rather than this function inventing one.
    """
    return COHORT_ALIASES.get(name, name)


@dataclass
class CohortSelection:
    """Which assays a cohort admits, and a full account of what it dropped and why."""

    cohort: str
    spec_version: str = COHORT_SPEC_VERSION
    assay_labels: dict[str, str] = field(default_factory=dict)      # assay_chembl_id -> label
    admitted: set[str] = field(default_factory=set)
    excluded: dict[str, str] = field(default_factory=dict)          # assay_chembl_id -> reason
    label_counts: dict[str, int] = field(default_factory=dict)
    attrition: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "cohort": self.cohort,
            "spec_version": self.spec_version,
            "n_assays_total": len(self.assay_labels),
            "n_assays_admitted": len(self.admitted),
            "n_assays_excluded": len(self.excluded),
            "label_counts": self.label_counts,
            "exclusion_reasons": _tally(self.excluded.values()),
            "attrition": self.attrition,
            # Exact IDs, so a cohort is reproducible without re-deriving it from regexes that may
            # change. Sorted for a stable diff.
            "admitted_assay_ids": sorted(self.admitted),
            "excluded_assay_ids": {k: v for k, v in sorted(self.excluded.items())},
        }


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def classify_description(description: str | None) -> str:
    """Label one assay description under the frozen precedence. Never returns silently-empty."""
    text = (description or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return NO_DESCRIPTION
    for label, rx in _COMPILED:
        if rx.search(text):
            return label
    return UNMATCHED


def label_assays(assays: Mapping[str, str | None]) -> dict[str, str]:
    """``{assay_chembl_id: description}`` -> ``{assay_chembl_id: label}``."""
    return {aid: classify_description(desc) for aid, desc in assays.items()}


def select_assays(
    assays: Mapping[str, str | None],
    cohort: str,
    *,
    structured: Mapping[str, Mapping[str, Any]] | None = None,
) -> CohortSelection:
    """Apply a frozen cohort to a target's assays, recording every exclusion with its reason.

    Deprecated spec-1.0 names are accepted and resolved. The name recorded in ``CohortSelection`` is the
    one the CALLER passed, not the canonical one, so an artifact says which name actually ran.

    ``structured`` maps assay id -> ChEMBL's structured fields (``bao_format``, ``assay_type``). A cohort
    declaring ``require_structured`` cannot be applied without it and RAISES rather than silently
    degrading to a text-only decision -- which is exactly the failure spec 1.2 exists to close.
    """
    canonical = resolve_cohort(cohort)
    if canonical not in FROZEN_COHORTS:
        raise ValueError(
            f"unknown cohort {cohort!r}; frozen cohorts are {sorted(FROZEN_COHORTS)} and deprecated "
            f"aliases are {sorted(COHORT_ALIASES)}. Cohorts are versioned deliberately: an ad-hoc "
            f"cohort would not be reproducible from the manifest."
        )
    rule = FROZEN_COHORTS[canonical]
    req = rule.get("require_structured")
    if req is not None and structured is None:
        raise ValueError(
            f"cohort {cohort!r} requires ChEMBL structured fields (bao_format, assay_type) and none "
            f"were supplied. Refusing to fall back to a description-only decision: that fallback is the "
            f"defect this cohort was written to remove -- it admitted a cell-based reporter assay, three "
            f"assays with conflicting assay_type, and 23 tandem-domain constructs."
        )

    labels = label_assays(assays)
    sel = CohortSelection(cohort=cohort, assay_labels=labels, label_counts=_tally(labels.values()))

    accept = rule["accept_labels"]
    exclude = set(rule["exclude_labels"])
    for aid, label in labels.items():
        if label in exclude:
            sel.excluded[aid] = f"label={label} is excluded by cohort {cohort!r}"
        elif accept is not None and label not in accept:
            sel.excluded[aid] = f"label={label} is not in the cohort's accepted set"
        elif req is not None:
            reason = _structured_exclusion(aid, assays.get(aid), structured or {}, req)
            if reason:
                sel.excluded[aid] = reason
            else:
                sel.admitted.add(aid)
        else:
            sel.admitted.add(aid)
    sel.attrition = {
        "assays_in": len(labels),
        "assays_out": len(sel.admitted),
        "fraction_assays_kept": round(len(sel.admitted) / len(labels), 4) if labels else 0.0,
    }
    return sel


def _structured_exclusion(
    aid: str,
    description: str | None,
    structured: Mapping[str, Mapping[str, Any]],
    req: Mapping[str, Any],
) -> str | None:
    """Why this assay fails a structured requirement, or None if it passes.

    Every branch returns a DISTINCT reason. An exclusion count is not auditable if every reason reads
    "excluded by cohort"; the manifest records these strings verbatim so a reader can see the mix of
    cell-based, conflicting and merely-uninformative metadata rather than one lump.
    """
    meta = structured.get(aid)
    if meta is None:
        # Fail closed. An assay whose metadata is absent has not been confirmed, and a cohort that
        # admits the unconfirmed is the cohort this one replaced.
        return "structured metadata absent — cannot confirm; excluded fail-closed"

    if req.get("exclude_dual_domain_description") and description:
        m = DUAL_DOMAIN_RE.search(str(description))
        if m:
            return (f"description names a tandem-domain construct ({m.group(0)!r}) — measures neither "
                    f"domain alone")

    bao = str(meta.get("bao_format") or "").strip()
    allowed_bao = set(req.get("bao_format_allow", ()))
    if bao not in allowed_bao:
        if bao in NON_SINGLE_PROTEIN_BAO_FORMATS:
            return (f"bao_format={bao} is a {NON_SINGLE_PROTEIN_BAO_FORMATS[bao]}, not a single-protein "
                    f"measurement — conflicts with the description")
        if not bao:
            return "bao_format absent — cannot confirm; excluded fail-closed"
        return (f"bao_format={bao} does not positively confirm a single-protein measurement "
                f"(ambiguous; the root 'assay format' term asserts nothing)")

    atype = str(meta.get("assay_type") or "").strip().upper()
    allowed_types = {t.upper() for t in req.get("assay_type_allow", ())}
    if atype not in allowed_types:
        if not atype:
            return "assay_type absent — cannot confirm; excluded fail-closed"
        return (f"assay_type={atype} conflicts with a binding cohort (expected one of "
                f"{sorted(allowed_types)})")
    return None


def apply_to_activities(activities, selection: CohortSelection, *, id_column: str = "assay_chembl_id"):
    """Filter an activity frame to a cohort's assays, returning ``(frame, attrition)``.

    Activity-level and compound-level attrition are both reported: a cohort can keep most assays while
    dropping most measurements, and a reader cannot infer one from the other.
    """
    if id_column not in activities.columns:
        raise ValueError(
            f"activity frame has no {id_column!r} column, so a cohort cannot be applied. The pull must "
            f"retain assay identity -- without it a single-protein target collapses into one label."
        )
    before_rows = int(len(activities))
    before_cmpds = int(activities["canonical_smiles"].nunique()) if "canonical_smiles" in activities else 0
    kept = activities[activities[id_column].isin(selection.admitted)]
    after_cmpds = int(kept["canonical_smiles"].nunique()) if "canonical_smiles" in kept else 0
    attrition = {
        "activities_in": before_rows,
        "activities_out": int(len(kept)),
        "fraction_activities_kept": round(len(kept) / before_rows, 4) if before_rows else 0.0,
        "compounds_in": before_cmpds,
        "compounds_out": after_cmpds,
        "fraction_compounds_kept": round(after_cmpds / before_cmpds, 4) if before_cmpds else 0.0,
        "activities_with_unknown_assay": int(
            (~activities[id_column].isin(selection.assay_labels)).sum()
        ),
    }
    return kept, attrition
