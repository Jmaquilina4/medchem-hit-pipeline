"""Derive assay-label composition and per-label potency medians from the frozen raw inputs.

Why this is a script and not a paragraph: the composition numbers quoted in the documentation were
wrong. They had been computed by an ad-hoc audit *before* the classification precedence was frozen
(``cell_based`` is evaluated FIRST, ahead of domain), so a cell-based bromodomain-2 assay counted
towards ``domain_2`` in the audit and towards ``cell_based`` in the shipped rules. The README ended up
claiming BRD4's records were ~11% second-bromodomain and ~5% cell-based when the frozen rules give
6.3% and 15.8%.

An audit that is not re-run against the frozen rules is a number with no owner. This writes the
composition into ``provenance/<panel>/composition.json``, keyed to the same hash-verified inputs the
run consumed, and ``verify_docs_against_manifests.py`` asserts the documented figures against it.

Two denominators are reported because they answer different questions and differ by a factor of three:
  * **per assay** — how many distinct assays carry each label. BRD4 is 58.8% cell-based by this count.
  * **per activity record** — how many IC50 measurements carry each label. BRD4 is 15.8% cell-based.
Cell-based assays are numerous but small; domain-resolved assays are fewer and much larger. Quoting one
denominator while implying the other is how the original error read as plausible.

Usage:
    python scripts/derive_composition.py            # write provenance/<panel>/composition.json
    python scripts/derive_composition.py --print    # show the tables, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PANELS = {"jak1": "JAK1", "jak1_sensitivity": "JAK1", "brd4": "BRD4", "brd4_sensitivity": "BRD4"}


def _load_verified(manifest: dict, name: str) -> pd.DataFrame:
    info = manifest["raw_inputs"][name]
    p = REPO / info["path"]
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != info["sha256"]:
        raise SystemExit(f"{name}: hash mismatch — this is not the input the run consumed")
    return pd.read_csv(p)


def derive(panel: str) -> dict:
    from medchem.cohorts import COHORT_SPEC_VERSION, label_assays

    target = PANELS[panel]
    m = json.loads((REPO / "runs" / panel / "run_manifest.json").read_text())
    acts = _load_verified(m, f"{target}_raw.csv")
    assays = _load_verified(m, f"{target}_assays.csv")

    labels = label_assays(dict(zip(assays["assay_chembl_id"], assays["description"], strict=True)))
    acts = acts.assign(label=acts["assay_chembl_id"].map(labels))

    # The median that MATTERS is the one on curated labels: one median pIC50 per compound after
    # ChEMBL validity flags, IC50-only enforcement, relation checks and unit conversion. Taking a
    # median of the raw pchembl_value column instead answers a different question and gave a JAK1
    # biochemical/cell-based gap of 0.12 log units where the curated path gives 0.45. Call the
    # pipeline's own function so the number is definitionally the pipeline's, not a re-derivation.
    from medchem.data.curate import curate_activities

    curated: dict[str, pd.DataFrame] = {}
    for lab in sorted({v for v in labels.values()}):
        sub = acts[acts["label"] == lab]
        if sub.empty:
            continue
        cur = curate_activities(sub.copy(), target)
        if not cur.empty:
            curated[lab] = cur

    per_assay = pd.Series(labels).value_counts()
    per_record = acts["label"].value_counts(dropna=False)
    n_assay, n_record = len(labels), len(acts)

    rows = {}
    for lab in sorted(set(per_assay.index) | set(per_record.index), key=lambda k: -per_record.get(k, 0)):
        cur = curated.get(str(lab))
        rows[str(lab)] = {
            "assays": int(per_assay.get(lab, 0)),
            "pct_assays": round(100.0 * per_assay.get(lab, 0) / n_assay, 1),
            "records": int(per_record.get(lab, 0)),
            "pct_records": round(100.0 * per_record.get(lab, 0) / n_record, 1),
            "curated_compounds": int(len(cur)) if cur is not None else 0,
            "median_pIC50_curated": round(float(cur["pIC50"].median()), 2) if cur is not None else None,
        }

    def med(lab: str) -> float | None:
        return rows.get(lab, {}).get("median_pIC50_curated")

    gaps = {}
    if med("biochemical") is not None and med("cell_based") is not None:
        gaps["biochemical_minus_cell_based"] = round(med("biochemical") - med("cell_based"), 2)
    if med("domain_1") is not None and med("domain_2") is not None:
        gaps["domain_1_minus_domain_2"] = round(med("domain_1") - med("domain_2"), 2)

    # One clinical reference read across labels is the most legible evidence that a pooled label is a
    # mixture: the same molecule, the same target, two numbers.
    references = {}
    ref_smiles = {"BRD4": ("apabetalone",
                           "COc1cc(OC)c2c(=O)[nH]c(-c3cc(C)c(OCCO)c(C)c3)nc2c1")}.get(target)
    if ref_smiles:
        from rdkit import Chem
        name, smi = ref_smiles
        want = Chem.CanonSmiles(smi)
        per_lab = {}
        for lab, cur in curated.items():
            hit = cur[cur["canonical_smiles"].apply(
                lambda s: isinstance(s, str) and Chem.CanonSmiles(s) == want)]
            if not hit.empty:
                per_lab[lab] = round(float(hit["pIC50"].iloc[0]), 2)
        if per_lab:
            references[name] = per_lab
            if "domain_1" in per_lab and "domain_2" in per_lab:
                gaps[f"{name}_domain_2_minus_domain_1"] = round(
                    per_lab["domain_2"] - per_lab["domain_1"], 2)

    return {
        # NOT a "panel" field. This record describes a TARGET's assay mixture, computed from the raw
        # snapshot both panels of a pair share, so one composition serves both -- see the caching loop in
        # main(). Naming the panel it was first derived FROM meant the two sensitivity copies identified
        # themselves as the headline panel: provenance/jak1_sensitivity/composition.json said
        # "panel": "jak1". The content was right and its label was wrong, which is the worse of the two
        # failures because the label is what a reader checks first.
        "describes": f"the {target} target's assay-label mixture, shared by both of its panels",
        "target": target,
        "panels_sharing_this_record": sorted(k for k, v in PANELS.items() if v == target),
        "cohort_spec_version": COHORT_SPEC_VERSION,
        "n_assays": n_assay,
        "n_activity_records": n_record,
        "inputs": {n: m["raw_inputs"][n]["sha256"]
                   for n in (f"{target}_raw.csv", f"{target}_assays.csv")},
        "note": (
            "pct_assays and pct_records have DIFFERENT denominators and differ by up to 3x. "
            "Cell-based assays are numerous but small; domain-resolved assays are fewer and larger. "
            "Quote which one you mean."
        ),
        "by_label": rows,
        "median_gaps_log_units": gaps,
        "reference_compounds": references,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", dest="dump")
    args = ap.parse_args()

    # Headline and sensitivity share one snapshot, so one composition per TARGET is enough.
    done: dict[str, dict] = {}
    for panel, target in PANELS.items():
        comp = done.get(target) or derive(panel)
        done[target] = comp
        if not args.dump:
            out = REPO / "provenance" / panel / "composition.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(comp, indent=2) + "\n")

    for target, comp in done.items():
        print("=" * 88)
        print(f"{target}: {comp['n_assays']:,} assays, {comp['n_activity_records']:,} IC50 records")
        print(f"\n  {'label':14s} {'assays':>7s} {'%':>7s} {'records':>9s} {'%':>7s} "
              f"{'cmpds':>7s} {'median':>7s}")
        for lab, r in comp["by_label"].items():
            m = r["median_pIC50_curated"]
            print(f"  {lab:14s} {r['assays']:7,} {r['pct_assays']:6.1f}% "
                  f"{r['records']:9,} {r['pct_records']:6.1f}% {r['curated_compounds']:7,} "
                  f"{(f'{m:.2f}' if m is not None else '—'):>7s}")
        for nm, per in comp.get("reference_compounds", {}).items():
            print(f"\n  reference {nm}: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(per.items())))
        for k, v in comp["median_gaps_log_units"].items():
            print(f"\n  gap  {k} = {v:+.2f} log units")
    print()
    if not args.dump:
        print("  written to provenance/*/composition.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
