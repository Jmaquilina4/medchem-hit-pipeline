"""Verify that every headline number in the documentation comes from a run manifest.

Why this exists
---------------
Documentation drifts from results silently. A number gets hand-copied, a run is repeated, a
sentence survives a correction it should not have — and nothing fails. This script makes the docs
a *derived* artifact: it resolves each panel's authoritative metrics from the frozen manifests, then
asserts that the strings in the docs match. If a metric changes, the docs fail until they are updated.

The disambiguation problem this had to solve
--------------------------------------------
A run directory accumulates one ``eval_report.json`` per cache key, and ``run_manifest.json`` records
the *gate values* but not which report produced them. That is not enough to identify the report,
because **scaffold-CV R² and y-scramble R² are identical across the era-label fix** — those splits
never touch the era labels. So each panel has three gate-matching reports, two of which are stale
pre-fix runs whose temporal split trained on all-years medians (i.e. leaked post-cutoff data).

Selecting by gate value alone silently picks a leaked temporal number. The authoritative report is
therefore identified by ``temporal_split.train_label_source``: exactly one report per panel reports
``pre-cutoff median``, and that is the only one whose temporal figure may be quoted. This script
asserts the "exactly one" part rather than assuming it — if a future run produces two, the ambiguity
is real and must be resolved by a human, not by taking the first match.

Usage:
    python scripts/verify_docs_against_manifests.py            # check
    python scripts/verify_docs_against_manifests.py --print    # dump resolved truth, check nothing
Exit code is 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
PROV = REPO / "provenance"          # tracked; present in a fresh clone, unlike runs/


def _docs() -> list[Path]:
    """Every prose document that may quote a result, DISCOVERED rather than listed.

    A literal path list broke silently in the sanitized export: the public tree has ``README.md`` and
    ``docs/RESULTS.md``, while the working tree has ``README.md``, ``docs/RESULTS.md`` and the
    pre-export originals under ``docs/public/``. The list named the working-tree paths, so in the export
    it found one document, skipped the one holding most of the numbers, and then reported those numbers
    as undocumented. A checker that quietly narrows its own scope is worse than one that fails.

    So: glob, then assert the scope is plausible. Finding almost nothing is a bug, not a pass.
    """
    found = [p for p in [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")),
                         *sorted((REPO / "docs" / "public").glob("*.md"))] if p.is_file()]
    if len(found) < 2:
        raise SystemExit(
            f"only {len(found)} prose document(s) found under {REPO}; refusing to verify documentation "
            f"against manifests on a scope this small — the discovery is broken, not the docs."
        )
    return found


PANELS = {
    "jak1": "JAK1 headline",
    "jak1_sensitivity": "JAK1 sensitivity",
    "brd4": "BRD4 headline",
    "brd4_sensitivity": "BRD4 sensitivity",
}
# The single revision at which all four analyses executed, on a clean tree, against the published
# snapshot. Updated when a correction re-runs the panels -- which is the only legitimate reason.
# No revision literal. The property the docs may rely on is that all four panels ran at ONE clean
# revision, which publish_provenance.py records as code.single_clean_revision -- and which is checked
# below. A hardcoded SHA here would reintroduce the identifier the published records omit.
FROZEN_SHA = None
FROZEN_COHORT_SPEC = "1.2"
PAIRED_PANELS = (("jak1", "jak1_sensitivity"), ("brd4", "brd4_sensitivity"))

# Values that WERE published and are now wrong. Maintained by hand on each correction, because the
# verifier has no access to history -- and needed because "the right number appears somewhere" was
# shown to be compatible with "the wrong number is still in a table".
SUPERSEDED_METRIC_VALUES: tuple[tuple[str, str], ...] = (
    ("0.7502", "BRD4 headline scaffold-CV under the spec-1.1 text-only cohort; now 0.7283"),
    ("0.0094", "BRD4 headline y-scramble under spec 1.1; now -0.1251"),
    ("0.0127", "BRD4 headline temporal R2 under spec 1.1; now +0.0190"),
    ("3,753",  "BRD4 headline compound count under spec 1.1; now 2,794"),
    ("70.2%",  "BRD4 headline scaffold overlap under spec 1.1; now 66.0%"),
)
# A block may legitimately quote a superseded value when it is EXPLICITLY historical: a spec-to-spec
# table, a withdrawal, a before/after row, or an audit finding.
HISTORICAL_CONTEXT = re.compile(
    r"spec 1\.1|spec 1\.0|superseded|withdraw|earlier|previous|replaced|corrected|"
    r"text-only|->|→|admitted in error|it changed a conclusion",
    re.I,
)


def _deep(d: object, *keys: str) -> object:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _resolve_eval(panel: str, gate_scaffold: float) -> tuple[Path, dict]:
    """The one eval report whose gates match the manifest AND whose labels are era-split.

    Raises rather than guessing. A panel with two era-split candidates is a genuine ambiguity.
    """
    era, leaky = [], []
    for f in sorted((RUNS / panel).rglob("eval_report.json")):
        e = json.loads(f.read_text())
        r2 = _deep(e, "scaffold_cv", "r2")
        if r2 is None or abs(float(r2) - gate_scaffold) > 1e-12:
            continue
        src = str(_deep(e, "temporal_split", "train_label_source") or "")
        (era if src.startswith("pre-cutoff") else leaky).append((f, e))
    if len(era) != 1:
        raise SystemExit(
            f"{panel}: expected exactly 1 era-split eval report, found {len(era)} "
            f"({len(leaky)} leaky). Cannot pick one automatically — resolve by hand.\n"
            + "\n".join(f"    {f.parent.name[:12]}" for f, _ in era)
        )
    return era[0]


def _resolve_selectivity(panel: str) -> dict | None:
    """Latest support-gated selectivity metrics; pre-gating formats are ignored on purpose."""
    best = None
    for f in sorted((RUNS / panel).rglob("selectivity_metrics.json")):
        e = json.loads(f.read_text())
        if e.get("production_model") is not None:
            best = e
    return best


def resolve() -> dict:
    """Prefer the tracked ``provenance/`` copies over the gitignored run tree.

    This is what lets the check run in CI and from a fresh clone. ``provenance/`` is flat and already
    disambiguated by ``publish_provenance.py``, so no resolution is needed there; falling back to
    ``runs/`` re-runs the era-split resolution rule.
    """
    truth: dict[str, dict] = {}
    for panel, label in PANELS.items():
        tracked = PROV / panel
        if (tracked / "run_manifest.json").exists():
            m = json.loads((tracked / "run_manifest.json").read_text())
            gate_scaffold = float(_deep(m, "gates", "scaffold_cv_r2_min", "value"))
            path = tracked
            ev = json.loads((tracked / "eval_report.json").read_text())
            sel_f = tracked / "selectivity_metrics.json"
            sel = json.loads(sel_f.read_text()) if sel_f.exists() else None
            if not str(_deep(ev, "temporal_split", "train_label_source") or "").startswith(
                "pre-cutoff"
            ):
                raise SystemExit(f"{panel}: published eval report is not era-split — republish")
        elif (RUNS / panel / "run_manifest.json").exists():
            m = json.loads((RUNS / panel / "run_manifest.json").read_text())
            gate_scaffold = float(_deep(m, "gates", "scaffold_cv_r2_min", "value"))
            path, ev = _resolve_eval(panel, gate_scaffold)
            sel = _resolve_selectivity(panel)
        else:
            raise SystemExit(
                f"{panel}: no provenance record in provenance/ or runs/. A documentation check "
                f"that finds no manifests must not report success — see check_binary_metadata.py."
            )
        pairs = (sel or {}).get("direct_delta_scaffold_cv", {})
        truth[panel] = {
            "label": label,
            "git_sha": _deep(m, "code", "git_sha"),
            "dirty": _deep(m, "code", "dirty"),
            "cohort": _deep(m, "cohort", "as_run", "name"),
            "spec_version": str(_deep(m, "cohort", "spec_version")),
            "raw_input_hashes": {k: v["sha256"] for k, v in (m.get("raw_inputs") or {}).items()},
            "compounds": _deep(m, "cohort", "attrition", "primary_compounds"),
            "gates_pass": all(g.get("pass") for g in (m.get("gates") or {}).values()),
            "scaffold_cv_r2": gate_scaffold,
            "y_scramble_r2": float(_deep(m, "gates", "y_scramble_r2_max", "value")),
            "temporal_r2": float(_deep(ev, "temporal_split", "r2")),
            "temporal_cutoff": _deep(ev, "temporal_split", "cutoff_year"),
            "scaffold_overlap": float(_deep(ev, "leakage", "test_scaffold_overlap_frac")),
            "single_clean_revision": _deep(m, "code", "single_clean_revision"),
            "eval_report": path.name if path.is_dir() else path.parent.name,
            "source": "provenance/ (tracked)" if path.is_dir() else "runs/ (resolved)",
            "supported_pairs": sorted(p for p, v in pairs.items()
                                      if _deep(v, "support", "supported")),
            "n_pairs": len(pairs),
            "production_basis": _deep(sel, "production_model", "basis_column"),
            "production_comparators": _deep(sel, "production_model", "supported_comparators"),
            "pair_detail": {p: {"n": v["n"], "r2": round(v["r2"], 4),
                                "ci": [round(x, 3) for x in (_deep(v, "support", "r2_ci95") or [])],
                                "supported": _deep(v, "support", "supported")}
                            for p, v in pairs.items()},
        }
    return truth


def check(truth: dict) -> list[str]:
    """Every assertion the documentation depends on. Failures are returned, not raised, so one run
    reports all of them."""
    bad: list[str] = []
    docs = _docs()
    text = {str(d.relative_to(REPO)): d.read_text(encoding="utf-8") for d in docs}

    # --- any git revision the DOCS cite must be the frozen one -----------------------------------
    # This checker compared every manifest SHA to FROZEN_SHA and never looked at the SHA the prose
    # claimed, so a revision that appears nowhere in the manifests once sat in RESULTS.md as the one
    # that produced all four panels. A provenance claim nobody checks is the one that goes stale.
    #
    # Scoped to backticked SHORT hex tokens: 40+ hex is an action pin or a SHA-256 digest, and neither
    # is a claim about which revision ran the analysis.
    for rel, body in text.items():
        for m in re.finditer(r"`([0-9a-f]{7,12})`", body):
            line = body[: m.start()].count("\n") + 1
            bad.append(f"{rel}:{line}: cites a short-hex revision identifier. The analysis workspace is "
                       f"not published, so no revision of it may be quoted; cite the "
                       f"scientific-source digest instead")

    # --- invariants the user specified: one clean SHA, shared snapshots, no unsupported production model
    for panel, t in truth.items():
        if t.get("single_clean_revision") is not True:
            bad.append(f"{panel}: published manifest does not record "
                       f"code.single_clean_revision=true — the panels no longer share one clean "
                       f"revision, or the record was not produced by publish_provenance.py")
        if t["dirty"] is not False:
            bad.append(f"{panel}: manifest records dirty={t['dirty']}; must be False")
        if not t["gates_pass"]:
            bad.append(f"{panel}: a gate did not pass")
        if t["spec_version"] != FROZEN_COHORT_SPEC:
            bad.append(f"{panel}: cohort spec {t['spec_version']} != {FROZEN_COHORT_SPEC}")
        if t["supported_pairs"] and t["production_comparators"] is not None:
            if sorted(c for c in t["production_comparators"]) != [
                p.split("-", 1)[1] for p in t["supported_pairs"]
            ]:
                bad.append(f"{panel}: production model comparators "
                           f"{t['production_comparators']} != supported pairs {t['supported_pairs']}")
        if not t["supported_pairs"] and t["production_basis"] is not None:
            bad.append(f"{panel}: no supported pair, yet a production model basis exists "
                       f"({t['production_basis']}) — an unsupported model must not ship")

    for a, b in PAIRED_PANELS:
        if truth[a]["raw_input_hashes"] != truth[b]["raw_input_hashes"]:
            bad.append(f"{a} and {b} do not share identical raw input hashes — the cohort "
                       f"comparison is confounded by different source data")
        if len(truth[a]["raw_input_hashes"]) != 8:
            bad.append(f"{a}: {len(truth[a]['raw_input_hashes'])} raw inputs hashed, expected 8")

    # --- every number quoted in the docs must exist in the resolved truth
    for panel, t in truth.items():
        for label, val, places in (("scaffold-CV", t["scaffold_cv_r2"], 3),
                                   ("y-scramble", t["y_scramble_r2"], 3),
                                   ("temporal", t["temporal_r2"], 3)):
            # Accept either sign glyph: the docs use a typographic minus.
            s = f"{val:.{places}f}"
            variants = {s, s.replace("-", "−"), f"+{s}" if val > 0 else s}
            if not any(v in body for body in text.values() for v in variants):
                bad.append(f"{panel}: {label} R² {s} appears in no doc — docs may be stale")
        if str(t["compounds"]) not in "".join(text.values()) and \
           f"{t['compounds']:,}" not in "".join(text.values()):
            bad.append(f"{panel}: compound count {t['compounds']:,} appears in no doc")

    # --- and no SUPERSEDED value may sit in an ACTIVE table
    #
    # The presence check above is necessary and not sufficient, and the gap it leaves is concrete: a
    # stale BRD4 gate table reading 0.7502 / -0.0094 while the correct 0.7283 / -0.1251 appears
    # elsewhere in the same document satisfies every assertion. Checking that a right number exists
    # says nothing about whether a wrong one also does.
    #
    # Superseded values are listed explicitly, because the verifier cannot know history. They are
    # allowed ONLY where the surrounding block marks itself as historical -- a spec-to-spec comparison,
    # a withdrawal, a before/after row. Anywhere else they are a stale claim.
    for value, what in SUPERSEDED_METRIC_VALUES:
        for name, body in text.items():
            for block in re.split(r"\n\s*\n", body):
                if value not in block:
                    continue
                if HISTORICAL_CONTEXT.search(block):
                    continue
                line = next((ln for ln in block.splitlines() if value in ln), block)
                bad.append(f"{name}: superseded value {value!r} ({what}) appears in an active block, "
                           f"not a historical comparison: {' '.join(line.split())[:88]}")

    # --- the scaffold-overlap figures must be attributed to the split that produced them
    # The 66.0-76.8% figure is easy to attribute to the TEMPORAL test sets, and three documents did. It
    # is the RANDOM 80/20 split: harness.py derives both scaf_tr and scaf[te] from train_test_split
    # indices, and never computes an overlap for tr_t/te_t. The error inverted the argument -- the
    # temporal test sets are 5.0-12.1% overlapping, i.e. already near scaffold-disjoint, so the
    # negative temporal R2 is a stronger result than the docs claimed rather than a flattered one.
    #
    # Both numbers are now asserted against records, so neither can drift and neither can be
    # re-attributed to the other split without failing here.
    for panel in PANELS:
        f = PROV / panel / "temporal_overlap.json"
        if not f.exists():
            bad.append(f"{panel}: no temporal_overlap.json — run scripts/derive_temporal_overlap.py")
            continue
        rec = json.loads(f.read_text())
        pub_random = (truth[panel] or {}).get("scaffold_overlap")
        rand_rec = rec.get("random_split_overlap_frac_for_contrast")
        if pub_random is not None and rand_rec is not None and abs(pub_random - rand_rec) > 1e-12:
            bad.append(f"{panel}: temporal_overlap.json records a different random-split overlap "
                       f"({rand_rec}) from the eval report ({pub_random})")
    t_pcts = [json.loads((PROV / p / "temporal_overlap.json").read_text())
              ["temporal_test_scaffold_overlap_pct"]
              for p in PANELS if (PROV / p / "temporal_overlap.json").exists()]
    if t_pcts:
        lo, hi = f"{min(t_pcts):.1f}%", f"{max(t_pcts):.1f}%"
        joined_all = "".join(text.values())
        for shown in (lo, hi):
            if shown not in joined_all:
                bad.append(f"derived temporal-split scaffold overlap {shown} appears in no doc; the "
                           f"docs must state the measured temporal range, not the random-split one")
        # The random-split range must never again be described as temporal.
        for name, body in text.items():
            for para in re.split(r"\n\s*\n", body):
                if re.search(r"66\.0.{0,3}76\.8", para) and re.search(
                    r"temporal|prospective", para, re.I
                ) and not re.search(r"random", para, re.I):
                    bad.append(f"{name}: the 66.0-76.8% RANDOM-split overlap is described in a "
                               f"temporal/prospective context without naming the random split: "
                               f"{' '.join(para.split())[:90]}")

    # --- the four interpretation corrections must not regress into stronger claims
    joined = "".join(text.values())
    for phrase, why in (
        ("real but weak", "withdrawn: overstates what the interval licenses"),
        ("Cleaner labels beat more labels", "withdrawn: causal claim across different cohorts"),
    ):
        for name, body in text.items():
            # Paragraph, not line: prose wraps, so a line-scoped check reports the withdrawal
            # sentence itself as a violation whenever the disclaimer lands on the previous line.
            for para in re.split(r"\n\s*\n", body):
                if phrase.lower() in para.lower() and not re.search(
                    r"superseded|withdraw|earlier draft|overstates|no longer|which the design|"
                    r"too strong|Status: FIXED|was wrong|corrected",
                    para, re.I,
                ):
                    bad.append(f"{name}: withdrawn claim {phrase!r} appears outside a "
                               f"withdrawal context ({why}): "
                               f"{' '.join(para.split())[:90]}")
    if "single permutation" not in joined:
        bad.append("docs never state that y-scramble is a single permutation")
    if "cannot isolate assay heterogeneity" not in joined:
        bad.append("docs never state the cohort comparison cannot isolate assay heterogeneity")

    # --- composition figures must be derived, not transcribed
    # This block exists because three of four BRD4 percentages and the JAK1 median gap were WRONG:
    # they came from an audit run before the classification precedence was frozen. Any number in this
    # family now has to match provenance/*/composition.json or the build fails.
    for panel in ("jak1", "brd4"):
        f = PROV / panel / "composition.json"
        if not f.exists():
            bad.append(f"{panel}: no composition.json — run scripts/derive_composition.py")
            continue
        comp = json.loads(f.read_text())
        for lab, r in comp["by_label"].items():
            # Only check labels the docs actually discuss; a label absent from prose is fine.
            if r["pct_records"] < 5.0:
                continue
            shown = f"{r['pct_records']:.1f}%"
            if lab in ("cell_based", "domain_1", "domain_2", "biochemical") and shown not in joined:
                bad.append(f"{comp['target']} {lab}: derived {shown} of records appears in no doc")
        for name, val in comp["median_gaps_log_units"].items():
            s = f"{abs(val):.2f}"
            if s not in joined:
                bad.append(f"{comp['target']} gap {name} = {val:+.2f} appears in no doc")
    # The specific withdrawn composition numbers must never reappear as claims.
    for wrong, right in (("1.08 log units", "0.45"), ("5.94 on", "5.85")):
        for name, body in text.items():
            for para in re.split(r"\n\s*\n", body):
                if wrong in para and not re.search(
                    r"superseded|withdraw|earlier draft|wrong|against a true|too strong|Status: FIXED",
                    para, re.I
                ):
                    bad.append(f"{name}: withdrawn figure {wrong!r} (correct: {right}) appears as a "
                               f"claim: {' '.join(para.split())[:80]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", dest="dump")
    args = ap.parse_args()

    truth = resolve()
    if args.dump:
        print(json.dumps(truth, indent=2, default=str))
        return 0

    print("=" * 96)
    print("DOCS vs MANIFESTS")
    print("=" * 96)
    for t in truth.values():
        print(f"\n  {t['label']:22s} {t['cohort']}")
        print(f"  {'':22s} n={t['compounds']:,}  scaffold={t['scaffold_cv_r2']:.4f}  "
              f"scramble={t['y_scramble_r2']:+.4f}  temporal={t['temporal_r2']:+.4f}")
        print(f"  {'':22s} scaffold overlap {t['scaffold_overlap']:.1%}  "
              f"selectivity {len(t['supported_pairs'])}/{t['n_pairs']} supported "
              f"{t['supported_pairs']}")
        print(f"  {'':22s} era-split labels confirmed; source: {t['source']}")

    bad = check(truth)
    print()
    if bad:
        print(f"  {len(bad)} MISMATCH(ES):")
        for b in bad:
            print(f"      - {b}")
        print()
        return 1
    # Scoped deliberately. This tool extracts the metrics of the CURRENT published panels and asserts
    # them against provenance/; it does not parse prose, and it cannot cover a figure from a superseded
    # run whose records were not retained. Saying "every documented figure" claimed the broader thing.
    print("  ALL CHECKS PASS — every published-panel metric quoted in the documentation traces to a "
          "frozen manifest.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
