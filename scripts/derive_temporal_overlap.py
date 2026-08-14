"""Derive the TEMPORAL split's train/test scaffold overlap, which the evaluation harness never reports.

Why this exists
---------------
The documentation quoted a scaffold overlap of 66.0-76.8% and attributed it to the **temporal** test
sets, using it to argue that the temporal figures still flatter the models. That attribution was wrong,
and the error inverted the argument.

``eval_report.json``'s ``leakage.test_scaffold_overlap_frac`` is computed from the **random 80/20**
split: ``medchem/eval/harness.py`` builds ``tr, te`` with ``train_test_split`` and derives both
``scaf_tr`` and ``scaf[te]`` from those indices. The temporal indices (``tr_t``, ``te_t``, built by
year against the cutoff) are used only to fit and score the temporal model. No overlap statistic is
computed for them anywhere in the pipeline.

Measured here, the temporal test sets share **5.0-12.1%** of their scaffolds with training, against
66.0-76.8% for the random split. So the temporal split is already close to scaffold-disjoint, and the
negative temporal R2 is a stronger result than the documentation claimed, not a weaker one: it is not
an artifact of shared scaffolds, because there are few shared scaffolds to blame.

Why a script rather than a harness change
-----------------------------------------
Adding the metric to the harness would change results-determining source, move the scientific-source
digest, and require re-running four frozen panels to obtain a number that alters no result. This
follows ``derive_composition.py`` instead: derive it once, publish it as a tracked provenance record,
and let ``verify_docs_against_manifests.py`` assert the documented figures against that record. The
same reason the composition figures are derived rather than transcribed.

Scaffolds come from the pipeline's own ``featurize`` output, resolved by cache key, so the definition
cannot drift from the one the models used.

Usage:
    python scripts/derive_temporal_overlap.py           # write provenance/<panel>/temporal_overlap.json
    python scripts/derive_temporal_overlap.py --print   # show the table, write nothing
    python scripts/derive_temporal_overlap.py --check   # recompute and compare to the published records
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
PROV = REPO / "provenance"

PANEL_CONFIGS = {
    "jak1": "configs/jak1.yaml",
    "jak1_sensitivity": "configs/jak1_sensitivity.yaml",
    "brd4": "configs/brd4.yaml",
    "brd4_sensitivity": "configs/brd4_sensitivity.yaml",
}
TOL = 1e-12


def _compute(panel: str) -> dict:
    """Recompute the temporal overlap from this panel's own featurize output."""
    import numpy as np
    import pandas as pd

    from medchem.provenance import resolve_stage_outputs

    outs = resolve_stage_outputs(REPO, REPO / PANEL_CONFIGS[panel], "featurize")
    meta_f = next((v for v in outs.values() if v.name == "meta.csv"), None)
    if meta_f is None or not meta_f.exists():
        raise SystemExit(
            f"{panel}: featurize produced no meta.csv at this code and config. This script needs a run "
            f"tree; run the panel first, or use --check against the published records only."
        )
    meta = pd.read_csv(meta_f)
    scaf = meta["scaffold"].fillna("").astype(str).to_numpy()
    year = pd.to_numeric(meta["document_year"], errors="coerce").to_numpy()

    # The cutoff comes from the PUBLISHED report, so this cannot silently use a different era boundary
    # from the one the temporal model was scored against.
    ev = json.loads((PROV / panel / "eval_report.json").read_text())
    cutoff = (ev.get("temporal_split") or {}).get("cutoff_year")
    if cutoff is None:
        raise SystemExit(f"{panel}: published eval_report.json records no temporal cutoff_year")
    cutoff = int(cutoff)

    tr_t = np.where(year < cutoff)[0]
    te_t = np.where(year >= cutoff)[0]
    train_scaf = set(scaf[tr_t])
    overlap = float(np.mean([s in train_scaf for s in scaf[te_t]])) if len(te_t) else 0.0
    return {
        "what_this_is": (
            "fraction of TEMPORAL test-set compounds whose Murcko scaffold also appears in the "
            "temporal training set. Not reported by the evaluation harness, whose "
            "leakage.test_scaffold_overlap_frac is the RANDOM 80/20 split."
        ),
        "panel": panel,
        "cutoff_year": cutoff,
        "n_train": int(len(tr_t)),
        "n_test": int(len(te_t)),
        "n_train_scaffolds": int(len(train_scaf)),
        "temporal_test_scaffold_overlap_frac": overlap,
        "temporal_test_scaffold_overlap_pct": round(overlap * 100, 1),
        "random_split_overlap_frac_for_contrast": (ev.get("leakage") or {}).get(
            "test_scaffold_overlap_frac"
        ),
        "scaffold_source": "featurize meta.csv, resolved by stage cache key",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--print", action="store_true", dest="show", help="show the table, write nothing")
    g.add_argument("--check", action="store_true", help="recompute and compare to published records")
    args = ap.parse_args()

    print("=" * 96)
    print("TEMPORAL-SPLIT SCAFFOLD OVERLAP")
    print("=" * 96)
    print(f"  {'panel':20s} {'cutoff':>6s} {'n_train':>8s} {'n_test':>7s} {'TEMPORAL':>10s} "
          f"{'RANDOM':>9s}")

    problems: list[str] = []
    written = 0
    for panel in PANEL_CONFIGS:
        pub_f = PROV / panel / "temporal_overlap.json"
        if args.check and not pub_f.is_file():
            problems.append(f"{panel}: temporal_overlap.json is not published")
            continue
        try:
            rec = _compute(panel)
        except SystemExit as exc:
            if args.check:
                # No run tree: verify the published records are at least internally coherent.
                pub = json.loads(pub_f.read_text())
                if not (0.0 <= pub["temporal_test_scaffold_overlap_frac"] <= 1.0):
                    problems.append(f"{panel}: published overlap is not a fraction")
                print(f"  {panel:20s} {pub['cutoff_year']:>6d} {pub['n_train']:>8d} "
                      f"{pub['n_test']:>7d} {pub['temporal_test_scaffold_overlap_pct']:>9.1f}% "
                      f"{'(published; no run tree to recompute)':>9s}")
                continue
            raise SystemExit(str(exc)) from exc

        r = rec["random_split_overlap_frac_for_contrast"]
        print(f"  {panel:20s} {rec['cutoff_year']:>6d} {rec['n_train']:>8d} {rec['n_test']:>7d} "
              f"{rec['temporal_test_scaffold_overlap_pct']:>9.1f}% "
              f"{(r * 100 if r is not None else float('nan')):>8.1f}%")

        if args.check:
            pub = json.loads(pub_f.read_text())
            d = abs(pub["temporal_test_scaffold_overlap_frac"]
                    - rec["temporal_test_scaffold_overlap_frac"])
            if d > TOL:
                problems.append(f"{panel}: published {pub['temporal_test_scaffold_overlap_frac']} "
                                f"!= recomputed {rec['temporal_test_scaffold_overlap_frac']} "
                                f"(delta {d:.2e})")
            for k in ("cutoff_year", "n_train", "n_test"):
                if pub.get(k) != rec[k]:
                    problems.append(f"{panel}: {k} published {pub.get(k)} != recomputed {rec[k]}")
        elif not args.show:
            pub_f.parent.mkdir(parents=True, exist_ok=True)
            pub_f.write_text(json.dumps(rec, indent=2) + "\n")
            written += 1

    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"      - {p}")
        return 1
    if written:
        print(f"\n  wrote {written} record(s) to provenance/<panel>/temporal_overlap.json")
    else:
        print("\n  clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
