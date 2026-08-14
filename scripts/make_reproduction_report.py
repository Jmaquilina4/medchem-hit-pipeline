"""Write the machine-readable reproduction report: what was run, under what, and what came out.

Why a file rather than prose
---------------------------
The documentation states tolerances and quotes metrics, and a reader can check those by hand. What
prose cannot carry is the *shape* of the evidence: which commands produced the frozen artifacts, which
interpreter and locked dependency set they ran under, which source digest identifies that code, what the
per-panel differences were when the panels were last re-run from the published bytes, and which support
and production-model decisions came out. Recording that as JSON makes it checkable by a script instead of
by a careful human, and ``scripts/check_reproduction_report.py`` does exactly that in CI.

The report is DERIVED, never hand-edited: every value comes from the published provenance records, the
locked environment, or the digest tool. Regenerate it whenever the panels are re-run.

Usage:
    python scripts/make_reproduction_report.py            # write provenance/REPRODUCTION.json
    python scripts/make_reproduction_report.py --print    # show it, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
PROV = REPO / "provenance"
OUT = PROV / "REPRODUCTION.json"

PANELS = ("jak1", "jak1_sensitivity", "brd4", "brd4_sensitivity")
PANEL_CONFIGS = {p: f"configs/{p}.yaml" for p in PANELS}

# The six stages that produce every published metric. `receptor` is excluded deliberately: it fetches a
# structure from the RCSB PDB, so including it would make the documented reproduction command require
# network even when the frozen snapshots are restored.
METRIC_STAGES = ("data_pull", "curate", "featurize", "qsar", "selectivity", "evaluate")

# Tolerances, per metric family, with the reason each is what it is.
TOLERANCES = {
    "default": {
        "value": 1e-12,
        "applies_to": "every metric this documentation reports",
        "why": (
            "float64 rounding varies with BLAS kernel, thread count and CPU. This bound is far above the "
            "differences MEASURED on a same-machine cache-free re-run (see cache_free_reproduction "
            "below), and its remaining headroom is for cross-machine variation that has NOT been "
            "measured here: no independent reproduction on other hardware has been performed. An "
            "earlier version of this field said 'comfortably above cross-machine reports', which "
            "implied outside evidence that does not exist"
        ),
    },
    "rank_statistics": {
        "value": 1e-5,
        "applies_to": ["roc_auc", "pr_auc", "pr_auc_lift_over_baseline"],
        "in_files": ["selectivity_metrics.json"],
        "why": (
            "these are RANK statistics computed on RandomForestRegressor(n_jobs=-1) predictions. "
            "Scikit-learn accumulates each tree's contribution from a thread pool, so repeated predict() "
            "calls on one fitted model differ by up to ~1.3e-15 (exactly reproducible at n_jobs=1). A "
            "shift that small cannot move a continuous metric but can swap two near-tied predictions, "
            "and a rank statistic then moves by one discrete quantum. This bound is an OBSERVED upper "
            "bound with headroom, not a derived limit: the quantum's size depends on how many near-ties "
            "a dataset holds"
        ),
    },
    "decisions": {
        "value": "exact",
        "applies_to": [
            "support verdicts", "production_model.written", "supported_comparators",
            "basis_column", "paired counts", "support reasons",
        ],
        "why": ("threshold comparisons and integer counts, whose margins are orders of magnitude wider. "
                "NOTE the scope: these are the decisions carried by the two COMPARED artifacts. "
                "Assay-level cohort exclusion reasons live in run_manifest.json, which is not compared, "
                "so they are not covered here -- an earlier version of this list said 'exclusion "
                "reasons' and implied they were"),
    },
}


def _run(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout.strip()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _digest() -> dict:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("_ssd", REPO / "scripts" / "scientific_source_digest.py")
    assert spec is not None and spec.loader is not None
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    d = m.build() if hasattr(m, "build") else None
    if d is None:                                  # fall back to the CLI's own JSON
        d = json.loads(_run("uv", "run", "python", "scripts/scientific_source_digest.py", "--json"))
    return {
        "scientific_source_digest": d["scientific_source_digest"],
        "modules_hashed": len(d.get("modules") or {}),
        "entry_modules": list(getattr(m, "ENTRY_MODULES", ())),
        "frozen_analysis_digest": getattr(m, "FROZEN_ANALYSIS_DIGEST", None),
        "declared_source_delta": list(getattr(m, "EXPECTED_RENAME_DELTA", ()) or []),
        "interpreter_scope": (
            "the digest is comparable only within one CPython minor version, because the AST "
            "normalisation it relies on differs between them"
        ),
    }


def _panel(panel: str) -> dict:
    man = json.loads((PROV / panel / "run_manifest.json").read_text())
    ev = json.loads((PROV / panel / "eval_report.json").read_text())
    sel_f = PROV / panel / "selectivity_metrics.json"
    sel = json.loads(sel_f.read_text()) if sel_f.is_file() else None
    to_f = PROV / panel / "temporal_overlap.json"
    to = json.loads(to_f.read_text()) if to_f.is_file() else None

    pairs = (sel or {}).get("direct_delta_scaffold_cv") or {}
    pm = (sel or {}).get("production_model") or {}
    return {
        "config": PANEL_CONFIGS[panel],
        # --force and a FRESH --workdir: without both, a run resolves from cache and proves nothing
        # about reproduction. The workdir is per-panel and must not pre-exist.
        "command": (
            f"MEDCHEM_FROZEN_SNAPSHOT=data/frozen_snapshots uv run medchem run -p discovery "
            f"-c {PANEL_CONFIGS[panel]} --force --workdir <fresh-empty-dir>/{panel} "
            + " ".join(f"--stage {s}" for s in METRIC_STAGES)
        ),
        # `cohort.name` does not exist in the manifest; the requested cohort and the one actually run
        # are separate fields, and they are recorded separately here because a divergence between them
        # would be a real finding rather than a formatting detail.
        "cohort": {
            "requested": (man.get("cohort") or {}).get("requested"),
            "as_run": ((man.get("cohort") or {}).get("as_run") or {}).get("name"),
            "spec_version": (man.get("cohort") or {}).get("spec_version"),
            "primary_compounds": ((man.get("cohort") or {}).get("attrition") or {})
            .get("primary_compounds"),
        },
        "single_clean_revision": (man.get("code") or {}).get("single_clean_revision"),
        "metrics": {
            "scaffold_cv_r2": (ev.get("scaffold_cv") or {}).get("r2"),
            "temporal_r2": (ev.get("temporal_split") or {}).get("r2"),
            "temporal_cutoff_year": (ev.get("temporal_split") or {}).get("cutoff_year"),
            "train_label_source": (ev.get("temporal_split") or {}).get("train_label_source"),
            # TOP-LEVEL: there is no `y_scramble` object -- reading one yielded None silently.
            "y_scramble_r2": ev.get("y_scramble_r2"),
            "random_split_scaffold_overlap": (ev.get("leakage") or {})
            .get("test_scaffold_overlap_frac"),
            "temporal_split_scaffold_overlap": (to or {})
            .get("temporal_test_scaffold_overlap_frac"),
        },
        "decisions": {
            "pairs_supported": {k: bool((v.get("support") or {}).get("supported"))
                                for k, v in pairs.items()},
            "n_supported": sum(1 for v in pairs.values()
                               if (v.get("support") or {}).get("supported")),
            "n_pairs": len(pairs),
            "production_model_written": pm.get("written"),
            "supported_comparators": sorted(pm.get("supported_comparators") or []),
            "basis_column": pm.get("basis_column"),
        },
        "record_hashes": {
            f.name: _sha256(f)
            for f in sorted((PROV / panel).glob("*.json"))
        },
    }


def build() -> dict:
    lock = REPO / "uv.lock"
    pyver = (REPO / ".python-version")
    return {
        "what_this_is": (
            "a machine-readable record of how the frozen results were produced and what they are. "
            "Derived from the published provenance records, the locked environment and the digest tool; "
            "never hand-edited. scripts/check_reproduction_report.py verifies it in CI."
        ),
        "how_to_reproduce": {
            "install": "uv sync --extra science --extra dev --extra docking --frozen",
            "stages": list(METRIC_STAGES),
            "stages_note": (
                "these six produce every published metric. `receptor` is excluded deliberately: it "
                "fetches a structure from the RCSB PDB, so including it would make this command require "
                "network. With MEDCHEM_FROZEN_SNAPSHOT set, the six stages contact no network -- note "
                "that WITHOUT it, data_pull queries ChEMBL and so needs network itself"
            ),
            "per_panel_commands": {p: _panel(p)["command"] for p in PANELS},
            "compare_against": "provenance/<panel>/eval_report.json and selectivity_metrics.json",
            "why_force_and_workdir": (
                "--force ignores the content-addressed cache and --workdir sends output to a fresh "
                "directory. Without both, the run resolves cached artifacts and demonstrates nothing "
                "about reproducibility: it would compare a file to itself"
            ),
        },
        "environment": {
            "python_pinned": pyver.read_text().strip() if pyver.is_file() else None,
            "python_running_now": platform.python_version(),
            "dependency_lock": "uv.lock",
            "dependency_lock_sha256": _sha256(lock) if lock.is_file() else None,
            "extras": ["science", "dev", "docking"],
        },
        "source_identity": _digest(),
        "tolerances": TOLERANCES,
        # RE-DERIVED from the published records. scripts/check_reproduction_report.py recomputes every
        # field here and fails on drift. This is NOT reproduction evidence: it summarises what the
        # records say, and re-deriving a summary from the same records cannot corroborate them.
        "published_records": {p: _panel(p) for p in PANELS},
        # RECORDED from an actual cache-free run, and not re-derivable. Written by
        # scripts/record_reproduction_run.py after running the four panels with --force into fresh
        # workdirs; absent until that has been done, which is honest rather than reassuring.
        "cache_free_reproduction": _load_recorded_run(),
    }


def _load_recorded_run() -> dict:
    """The recorded cache-free comparison, or an explicit statement that none has been made.

    Kept in its own file so regenerating the derived summary cannot silently overwrite measured
    evidence, and so its absence is visible instead of implied.
    """
    f = PROV / "REPRODUCTION_RUN.json"
    if not f.is_file():
        return {
            "performed": False,
            "why_absent": (
                "no cache-free reproduction has been recorded for this source tree. The summary above "
                "is re-derived from the published records and is not evidence that they reproduce."
            ),
        }
    return json.loads(f.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", dest="show", help="show it, write nothing")
    args = ap.parse_args()

    missing = [p for p in PANELS if not (PROV / p / "run_manifest.json").is_file()]
    if missing:
        raise SystemExit(f"no published provenance for {missing}; run publish_provenance.py first")

    report = build()
    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.show:
        print(text)
        return 0
    OUT.write_text(text)
    print(f"  wrote {OUT.relative_to(REPO)}  ({len(text):,} bytes)")
    print(f"    digest {report['source_identity']['scientific_source_digest'][:16]}…  "
          f"{report['source_identity']['modules_hashed']} modules  "
          f"delta={report['source_identity']['declared_source_delta'] or 'none'}")
    for p, d in report["published_records"].items():
        print(f"    {p:18s} scaffold={d['metrics']['scaffold_cv_r2']:.4f}  "
              f"temporal={d['metrics']['temporal_r2']:+.4f}  "
              f"{d['decisions']['n_supported']}/{d['decisions']['n_pairs']} pairs  "
              f"model={d['decisions']['production_model_written']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
