"""Emit a provenance manifest for a cohort-scoped run. Closes review finding R5.

The previous docking record listed compound IDs and scores and nothing else — no input SMILES hashes,
no code revision, no receptor hash, no engine version — so it could not be shown to be the same run
twice. A results table without that is an assertion.

What this records, and why each field earns its place:

* the **code revision and dirty flag**. A number produced from uncommitted work is not reproducible,
  and saying so is cheaper than discovering it later.
* **sha256 of every raw input**, including the assay metadata. Cohort selection depends on assay
  descriptions, so a manifest that hashes activities but not descriptions cannot prove the cohort.
* the **cohort name AND spec version**, because the rules are versioned code: a cohort reproduced from
  a regex that has since changed is not the same cohort.
* **admitted assay IDs**, so a cohort is reconstructible without re-running the classifier.
* **attrition at every step**, activities and compounds separately — a cohort can keep most assays and
  drop most measurements, and neither number implies the other.
* the **structure block**, which the earlier record omitted entirely — and which this release records
  as ``not evidenced`` rather than filling in. Resolving a receptor by modification time attributed
  whichever file was written last to every panel, so no receptor hash or engine version is recorded
  here; see the note the block itself carries.
* the **evaluation gates**, unchanged across cohorts, so a cohort change cannot be rewarded with a
  looser bar.

Usage:
    uv run python scripts/make_run_manifest_v2.py --run runs/brd4 --cohort domain1_noncellular_explicit
    uv run python scripts/make_run_manifest_v2.py --run runs/brd4 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _resolve_pull_outputs(config_path: Path) -> tuple[str, list[Path]]:
    """The exact artifacts this run's ``data_pull`` stage produced, by recomputing its cache key.

    Globbing for ``*_raw.csv`` was wrong in a way that is easy to miss and impossible to detect from
    the manifest afterwards. A run directory accumulates one output directory per cache key, and
    several of them contain a file of the same name with the same row count. Keying a dict by
    ``f.name`` while walking them in sorted (i.e. hash) order means the LAST directory
    lexicographically wins — an arbitrary choice unrelated to what the run consumed.

    That produced manifests whose ``*_raw.csv`` hashes pointed at a stale pull output written before
    ``assay_chembl_id`` was retained. The metrics were unaffected (the run itself consumed the correct
    frame, which is why cohort filtering worked at all), but the provenance record was useless for its
    one purpose: someone holding the hashed file could not apply a cohort to it, because the file has
    no assay identity.

    ``data_pull`` has no upstream dependencies, so its key is a pure function of code version, stage
    source and config subtree — recomputable exactly, with no inference and no guessing.
    """
    import medchem.stages  # noqa: F401  (registration side-effect)
    from medchem import __version__
    from medchem.config import load_config
    from medchem.pipeline.cache import hash_source, stage_cache_key
    from medchem.pipeline.runner import _config_subtree
    from medchem.pipeline.stage import get_pipeline

    st = get_pipeline("discovery")["data_pull"]
    key = stage_cache_key(
        code_version=__version__,
        source_hash=hash_source(st.fn),
        config_subtree=_config_subtree(load_config(config_path), st.config_keys),
        upstream_hashes={},
    )
    cache_manifest = REPO / ".medchem_cache" / f"{key}.json"
    if not cache_manifest.exists():
        raise SystemExit(
            f"no cache manifest for data_pull key {key[:12]}… — cannot identify which artifacts this "
            f"run consumed. Refusing to fall back to globbing: that is what recorded the wrong file."
        )
    outs = json.loads(cache_manifest.read_text())["outputs"]
    paths = [Path(str(v)) for v in (outs.values() if isinstance(outs, dict) else outs)]
    return key, [p if p.is_absolute() else REPO / p for p in paths]


def build(run_dir: Path, cohort: str | None, snapshot_dir: Path | None = None,
          config_path: Path | None = None, run_sha: str | None = None) -> dict:
    newest = lambda pat: (  # noqa: E731
        sorted(run_dir.rglob(pat), key=lambda p: p.stat().st_mtime)[-1]
        if list(run_dir.rglob(pat)) else None
    )
    from medchem.cohorts import COHORT_SPEC_VERSION, FROZEN_COHORTS

    # A sensitivity run SHARES the headline's snapshot by design -- that is the whole point, and it is
    # why the cohort lives outside the hashed acquisition section. But the shared artifacts live in the
    # headline's directory, so globbing this run's directory would record zero raw inputs and leave the
    # sensitivity run's provenance incomplete. --snapshot-from names where they actually are.
    search_dir = snapshot_dir or run_dir
    pull_key: str | None = None
    raw_inputs = {}
    if config_path is not None:
        pull_key, files = _resolve_pull_outputs(config_path)
        candidates = [f for f in files if f.name.endswith(("_raw.csv", "_assays.csv"))]
    else:
        candidates = sorted(search_dir.rglob("*_raw.csv")) + sorted(search_dir.rglob("*_assays.csv"))

    for f in candidates:
        # An input outside the repository cannot be recorded as a repo-relative path, and recording an
        # absolute one would leak a local layout. This happened for real: a --workdir test run wrote a
        # data_pull cache entry pointing into a scratch directory, and the next manifest build inherited
        # it. Fail with the cause and the fix rather than a relative_to() traceback.
        if not f.is_relative_to(REPO):
            raise SystemExit(
                f"{f} lies outside the repository, so it cannot be recorded as provenance. Its cache "
                f"entry was almost certainly written by a run using --workdir outside the repo. Remove "
                f"the offending .medchem_cache entry and re-run the stage inside the repository."
            )
        if f.name in raw_inputs:
            raise SystemExit(
                f"two candidate inputs named {f.name} — ambiguous provenance. Pass --config so the "
                f"data_pull cache key resolves it exactly."
            )
        # Fail closed on the specific defect this replaced: an activities file with no assay identity
        # cannot have a cohort applied, so recording it as the input is recording a false claim.
        if f.name.endswith("_raw.csv"):
            with f.open(encoding="utf-8") as fh:
                header = fh.readline()
            if "assay_chembl_id" not in header:
                raise SystemExit(
                    f"{f.relative_to(REPO)} has no assay_chembl_id column. A cohort cannot be applied "
                    f"to it, so it cannot be the input this run consumed. Refusing to record it."
                )
        # The SHA256 is the identity; the path is context, and it needs a caveat rather than a bare
        # string. A headline panel and its sensitivity panel SHARE one data_pull cache key by design --
        # the cohort lives in `curation`, not in the hashed `data` section, precisely so a pair consumes
        # one pull. Whichever of the pair ran last therefore owns the directory the shared cache entry
        # points at, so a headline manifest legitimately resolves to `runs/<sibling>_sensitivity/...`.
        # Recorded without that note, it reads as a mix-up. `runs/` is gitignored and unpublished
        # anyway, so the directory identifies nothing a reader can open.
        raw_inputs[f.name] = {"path": str(f.relative_to(REPO)), "sha256": _sha256(f),
                              "bytes": f.stat().st_size,
                              "path_note": ("the directory belongs to whichever panel of this pair last "
                                            "ran data_pull; the pair shares one cache key, so this is "
                                            "one artifact set rather than two. The sha256 is the "
                                            "identity")}
    shared_note = None
    if snapshot_dir is not None:
        shared_note = (
            f"raw inputs are the SHARED snapshot under {snapshot_dir.relative_to(REPO)}; this run "
            f"differs from its headline only in the declared cohort, so the hashes below must match "
            f"the headline manifest exactly"
        )

    curate_metrics = newest("curate_metrics.json")
    cohort_block: dict = {"requested": cohort, "spec_version": COHORT_SPEC_VERSION}
    if cohort:
        cohort_block["definition"] = FROZEN_COHORTS.get(cohort)
    if curate_metrics:
        cm = json.loads(curate_metrics.read_text())
        cohort_block["as_run"] = cm.get("assay_cohort")
        cohort_block["attrition"] = {
            "curated_rows": cm.get("curated_rows"),
            "primary_compounds": cm.get("primary_compounds"),
            "primary_druglike": cm.get("primary_druglike"),
        }
        cohort_block["temporal_labels"] = cm.get("temporal_labels")

    # NOT `newest(...)`. Resolving a receptor artifact by modification time attributes whichever
    # receptor.pdbqt was written last to THIS panel, regardless of which configuration produced it or
    # whether this panel ran the receptor stage at all. Every panel's manifest -- including the two
    # sensitivity panels, which never run receptor -- carried a structure block resolved that way, so the
    # records claimed a receptor execution that this release cannot evidence.
    #
    # Receptor artifacts are therefore NOT recorded here. Re-introducing them requires resolving by the
    # panel's own stage cache key (medchem.provenance.resolve_stage_outputs), and recording them only for
    # panels whose configuration actually runs the stage.
    receptor = None
    eval_report = newest("eval_report.json")
    gates = None
    if eval_report:
        gates = json.loads(eval_report.read_text()).get("gates")

    return {
        "manifest_version": 2,
        "what_this_covers": (
            "One cohort-scoped run: raw inputs, the frozen cohort applied, attrition, the models' "
            "evaluation gates, and any structural inputs. Every count here is computational. No "
            "pipeline-nominated candidate was prospectively synthesised or assayed as part of this "
            "work."
        ),
        "code": {
            # The SHA that EXECUTED the run. Defaults to HEAD, which is only correct when the manifest
            # is written immediately after the run on a clean tree. --run-sha states it explicitly so a
            # provenance record can be corrected later without silently reattributing the results to
            # whatever commit happened to be checked out at the time.
            "git_sha": run_sha or _git("rev-parse", "HEAD"),
            "git_describe": _git("describe", "--tags", "--always", *( [run_sha] if run_sha else [] )),
            # None, not False, when --run-sha is given. This read `... else False`, which MINTED a
            # clean-tree attestation from an argument: --run-sha names the revision that produced the
            # results, and nothing about passing it establishes that the tree was clean when they were
            # produced. Since publish_provenance derives `single_clean_revision` from this flag, a dirty
            # run could have been published as a clean one by supplying the SHA. Unknown is the honest
            # value, and the attestation below requires an explicit False rather than a falsy one.
            "dirty": bool(_git("status", "--porcelain")) if run_sha is None else None,
            "dirty_note": None if run_sha is None else (
                "not determined: --run-sha attributes these results to a revision recorded after the "
                "fact, so this tool cannot observe the tree state at run time"
            ),
            "note": "a number produced from a dirty tree is not reproducible; the flag is the warning",
            "manifest_tool_sha": _git("rev-parse", "HEAD") if run_sha else None,
            "manifest_tool_note": (
                "this provenance record was regenerated by tooling at manifest_tool_sha; the RESULTS "
                "are those of git_sha and were not recomputed" if run_sha else None
            ),
        },
        "raw_inputs": raw_inputs,
        "raw_inputs_resolved_by": (
            f"data_pull cache key {pull_key}" if pull_key else
            "directory glob (AMBIGUOUS: pass --config to resolve by cache key)"
        ),
        "raw_inputs_shared_from": str(snapshot_dir.relative_to(REPO)) if snapshot_dir else None,
        "raw_inputs_note": shared_note,
        "cohort": cohort_block,
        "structure": {
            "receptor_pdbqt": str(receptor.relative_to(REPO)) if receptor else None,
            "receptor_sha256": _sha256(receptor) if receptor else None,
            "box": None,
            "structure_note": (
                "receptor artifacts are not recorded in this release. They were previously resolved by "
                "modification time, which attributed whichever receptor was written last to every panel "
                "-- including the sensitivity panels, which do not run the receptor stage. Recording "
                "them again requires resolution by the panel's own cache key."
            ),
            "docking_engine_version": None,
            "docking_engine_version_note": (
                "not recorded. This was probed from the machine writing the manifest, which "
                "describes that machine rather than any execution: no docking ran, and the "
                "block above already says no receptor is evidenced. A version number beside "
                "that reads as though something used it."
            ),
        },
        "gates": gates,
        "gates_note": (
            "gates are IDENTICAL across cohorts by design: a cohort change must not be rewarded with a "
            "looser bar, or 'the filtered cohort passes' becomes unfalsifiable"
        ),
        "software": {
            "python": sys.version.split()[0],
            "packages": {
                m: _version(m) for m in ("rdkit", "scikit-learn", "numpy", "pandas", "xgboost")
            },
        },
    }


def _version(mod: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(mod)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/brd4")
    ap.add_argument("--cohort", default=None, help="frozen cohort name applied to this run")
    ap.add_argument("--snapshot-from", default=None,
                    help="directory holding the SHARED raw snapshot, for a sensitivity run whose "
                         "pull artifacts live under its headline run")
    ap.add_argument("--config", default=None,
                    help="config for this run; lets raw inputs resolve by data_pull cache key instead "
                         "of an ambiguous directory glob (strongly recommended)")
    ap.add_argument("--run-sha", default=None,
                    help="SHA that EXECUTED the run. Use when regenerating a provenance record after "
                         "the fact so results are not reattributed to the current HEAD.")
    ap.add_argument("--verify", action="store_true", help="re-hash and report drift; exit 1 on change")
    ap.add_argument("--require-clean", action="store_true",
                    help="REFUSE to write a manifest from a dirty worktree. Recording dirty=true is "
                         "not enough: a number produced from uncommitted work cannot be reproduced, "
                         "so the manifest must not certify it.")
    args = ap.parse_args()

    run_dir = REPO / args.run if not Path(args.run).is_absolute() else Path(args.run)
    if not run_dir.is_dir():
        print(f"  no such run directory: {run_dir}")
        return 1
    out = run_dir / "run_manifest.json"
    snap = None
    if args.snapshot_from:
        snap = REPO / args.snapshot_from if not Path(args.snapshot_from).is_absolute() \
            else Path(args.snapshot_from)
        if not snap.is_dir():
            print(f"  no such snapshot directory: {snap}")
            return 1
    cfg = None
    if args.config:
        cfg = REPO / args.config if not Path(args.config).is_absolute() else Path(args.config)
        if not cfg.exists():
            print(f"  config not found: {cfg}")
            return 1
    fresh = build(run_dir, args.cohort, snapshot_dir=snap, config_path=cfg, run_sha=args.run_sha)

    if args.require_clean and fresh["code"]["dirty"]:
        print("  REFUSING: the worktree is dirty, so this run is not reproducible from any revision.")
        print("  Commit or stash first. Recording dirty=true would certify an unreproducible number.")
        return 2

    if args.verify:
        if not out.exists():
            print(f"  no manifest at {out.relative_to(REPO)}")
            return 1
        old = json.loads(out.read_text())
        drift = [
            f"{k}: {old['raw_inputs'].get(k, {}).get('sha256')} -> {v['sha256']}"
            for k, v in fresh["raw_inputs"].items()
            if old.get("raw_inputs", {}).get(k, {}).get("sha256") != v["sha256"]
        ]
        if old.get("code", {}).get("git_sha") != fresh["code"]["git_sha"]:
            drift.append(f"git_sha: {old['code']['git_sha'][:8]} -> {fresh['code']['git_sha'][:8]}")
        print(f"  {len(drift)} drifted item(s)")
        for d in drift:
            print(f"    {d}")
        return 1 if drift else 0

    out.write_text(json.dumps(fresh, indent=2) + "\n")
    print(f"  wrote {out.relative_to(REPO)}")
    print(f"    code {fresh['code']['git_describe']} dirty={fresh['code']['dirty']}")
    print(f"    raw inputs hashed: {len(fresh['raw_inputs'])}"
          + (f"  (shared from {fresh['raw_inputs_shared_from']})"
             if fresh["raw_inputs_shared_from"] else ""))
    if not fresh["raw_inputs"]:
        print("    WARNING: no raw inputs hashed -- provenance is incomplete. Pass --snapshot-from "
              "if this run consumes a shared snapshot.")
    print(f"    cohort: {fresh['cohort']['requested']} (spec {fresh['cohort']['spec_version']})")
    print(f"    structure: {fresh['structure']['receptor_pdbqt'] or 'not evidenced in this release'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
