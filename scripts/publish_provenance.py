"""Copy the frozen provenance records out of the gitignored run tree into a tracked directory.

``runs/`` is gitignored — it holds gigabytes of cache-keyed intermediates, docked poses and library
shards, none of which belong in a repository. But the *provenance* records inside it are the only
thing that makes a documented number checkable: raw-input hashes, the exact admitted and excluded
assay IDs, the attrition at every step, the gate values, the environment.

Without them a clean clone can read a claim and has no way to test it. So the small JSON records are
published into ``provenance/``, which IS tracked, and ``scripts/verify_docs_against_manifests.py``
prefers them — meaning the documentation check runs in CI from a fresh clone with no run tree at all.

What is published, per panel:
  * ``run_manifest.json``       — inputs, cohort, attrition, gates, environment
  * ``eval_report.json``        — the uniquely-resolved era-split report (see the verifier's docstring)
  * ``selectivity_metrics.json``— per-pair support decisions and the production-model basis

What is deliberately NOT published: any raw or curated ChEMBL rows. Those are regenerated from the
source under its own licence, never republished here. The hashes prove identity of inputs; they do
not reconstruct them, and that limit is stated in the README rather than papered over.

Usage:
    python scripts/publish_provenance.py            # publish, refusing on anything identifying
    python scripts/publish_provenance.py --check     # verify published copies match the run tree
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
RUNS = REPO / "runs"
OUT = REPO / "provenance"
PANELS = ("jak1", "jak1_sensitivity", "brd4", "brd4_sensitivity")

# Same spirit as check_no_leaks.py: no personal paths and no organisation names carried in through a
# config string. Names of the software packages themselves belong in prose, not in provenance keys.
# Personal-name patterns are NOT hardcoded here. Writing the name being redacted into a published
# script publishes it -- the same mistake as documenting a leak by quoting it. Home-directory paths
# carry the account name anyway, and scripts/check_no_leaks.py loads the private pattern list.
IDENTIFYING = re.compile(r"/Users/|/home/|[A-Z]:\\\\Users", re.I)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


# Fields stripped when a record is published, with the reason recorded IN the record so the redaction
# is visible rather than silent.
#
# ``code.git_describe`` is the case that prompted this: it read like
# ``<internal-tag-name>-<N>-g<sha>``, publishing an internal release-candidate tag name and the commit
# count since it. The same reasoning applies to the revision identifier itself: a SHA for a workspace
# that is not published cannot be resolved or checked by a reader, so it carries no verifiable
# information while still naming a tree outside this repository. What the SHA was EVIDENCE for -- that
# all four panels ran at one clean revision -- is preserved as a flag.
REDACT_FIELDS: dict[str, tuple[str, str]] = {
    "git_describe": ("code", "removed at publication: encodes an internal tag name and commit count"),
    "git_sha": ("code", "removed at publication: a revision identifier for a workspace that is not "
                        "published cannot be resolved or verified by a reader, and naming it "
                        "references a tree outside this repository. The durable identity is the "
                        "scientific-source digest in provenance/IDENTITY.json, which is recomputable "
                        "from THIS tree. single_clean_revision below records the property the SHA was "
                        "evidence for."),
    "manifest_tool_sha": ("code", "removed at publication: the same revision as git_sha, at full "
                                  "length"),
    "manifest_tool_note": ("code", "removed at publication: it referred to manifest_tool_sha, which "
                                   "is itself redacted"),
}

# The structure block is REPLACED, not itemised into three "redacted" pseudo-fields. It was populated by
# taking the NEWEST receptor.pdbqt on disk rather than the artifact this panel's own cache key resolves
# to, so it attributed a receptor execution to all four panels -- including the two sensitivity panels,
# whose configurations do not run the receptor stage. Three redaction notices per panel is a lot of text
# to say one thing: no receptor artifact is evidenced for this release.
STRUCTURE_STATUS = {
    "status": "not_evidenced_in_this_release",
    "why": (
        "receptor artifacts were previously resolved by modification time, which attributes whichever "
        "receptor was written last to every panel regardless of which configuration produced it or "
        "whether that panel runs the receptor stage at all. No artifact is published for the frozen "
        "panels. Recording one again requires resolution by the panel's own stage cache key, and only "
        "for panels whose graph includes the stage."
    ),
    "stage_implemented": True,
    "consumed_downstream_here": False,
}


def _redacted(body: str, *, all_revisions: set[str] | None = None) -> str:
    """The published form of a record: identical except for REDACT_FIELDS, which is stated.

    ``all_revisions`` is every ``code.git_sha`` across the panels being published. The SHA itself is
    redacted, but "all four panels ran at ONE clean revision" is a real property a reader should be able
    to rely on, so it is asserted here and recorded as a flag rather than left implicit in a value the
    reader cannot check anyway.
    """
    d = json.loads(body)
    removed = []
    blk = d.get("code")
    if isinstance(blk, dict) and all_revisions is not None:
        # `is False`, deliberately: a dirty flag of None means the tool could not observe the tree state
        # (--run-sha), and an unobserved tree must not attest to a clean one. `not blk.get("dirty")`
        # would have accepted None as clean, which is how an attestation gets minted from an absence.
        blk["single_clean_revision"] = (len(all_revisions) == 1 and blk.get("dirty") is False)
        blk["single_clean_revision_note"] = (
            "true when every published panel recorded the SAME source revision and none ran from a "
            "dirty tree. This is the checkable part of the redacted git_sha."
        )
    # Replace the structure block wholesale. Any positive receptor claim must come with an exact
    # artifact path and hash, resolved by cache key; until then the record says so in one place.
    if isinstance(d.get("structure"), dict):
        engine = d["structure"].get("docking_engine_version")
        d["structure"] = dict(STRUCTURE_STATUS)
        if engine is not None:
            # Probing the MACHINE for an engine version records what happened to be installed where the
            # manifest was written, not what produced anything here. Nothing ran, so nothing is claimed.
            d["structure"]["docking_engine_version_note"] = (
                "not recorded: the previous value was probed from the machine writing the manifest, "
                "which describes that machine rather than any execution in this release"
            )

    for field, (section, why) in REDACT_FIELDS.items():
        blk = d.get(section)
        if isinstance(blk, dict) and field in blk:
            blk.pop(field)
            blk[f"{field}_redacted"] = why
            removed.append(f"{section}.{field}")
    return json.dumps(d, indent=2) + "\n" if removed or all_revisions is not None else body


# Which config produced each panel. Needed because artifacts are now resolved EXACTLY, by recomputing
# the stage's cache key, rather than by matching a metric value.
PANEL_CONFIGS = {
    "jak1": "configs/jak1.yaml",
    "jak1_sensitivity": "configs/jak1_sensitivity.yaml",
    "brd4": "configs/brd4.yaml",
    "brd4_sensitivity": "configs/brd4_sensitivity.yaml",
}


def _resolve(panel: str, stage: str, filename: str) -> Path | None:
    """The exact artifact this panel's run produced, via the stage cache key.

    Replaces two heuristics that both failed. Matching an ``eval_report.json`` by the manifest's gate
    value worked only while gate values were unique; the moment a cohort correction re-ran every panel,
    the three UNCHANGED panels reproduced their scaffold-CV R² exactly, two era-split reports matched,
    and the rule correctly refused to choose. Taking the newest ``selectivity_metrics.json`` was worse:
    it would have silently returned whichever file was written last.
    """
    from medchem.provenance import resolve_stage_outputs

    outs = resolve_stage_outputs(REPO, REPO / PANEL_CONFIGS[panel], stage)
    for v in outs.values():
        if v.name == filename:
            return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare published copies to the run tree")
    args = ap.parse_args()

    if not RUNS.exists():
        # NO RUN TREE: this is the public-clone path, and what it can check is strictly less.
        #
        # With a run tree, --check resolves each artifact by RECOMPUTING the stage cache key and compares
        # the published copy to the resolved one. Without a run tree there is nothing to resolve against,
        # so it verifies that the tracked records exist and are internally consistent instead. That is a
        # real check and it is NOT cache-key verification; conflating the two would overstate what a
        # reader can confirm from the published tree alone.
        if not args.check:
            print("  no runs/ tree here — nothing to publish (expected in CI).")
            return 0
        missing = [f"{p}/{n}" for p in PANELS
                   for n in ("run_manifest.json", "eval_report.json")
                   if not (OUT / p / n).is_file()]
        if missing:
            print(f"  no runs/ tree, AND published provenance is incomplete: {missing}")
            return 1
        for p in PANELS:
            m = json.loads((OUT / p / "run_manifest.json").read_text())
            ev = json.loads((OUT / p / "eval_report.json").read_text())
            gate = (m.get("gates") or {}).get("scaffold_cv_r2_min", {}).get("value")
            got = (ev.get("scaffold_cv") or {}).get("r2")
            if gate is None or got is None or abs(float(gate) - float(got)) > 1e-12:
                print(f"  {p}: published manifest gate {gate} disagrees with its eval report {got}")
                return 1
        print(f"  no runs/ tree (expected in CI); verified {len(PANELS)} published record set(s) exist "
              f"and agree internally.")
        return 0

    # Every panel's recorded revision, gathered BEFORE publishing so the redacted records can state
    # whether they all agree -- the property the SHA was evidence for.
    all_revisions: set[str] = set()
    for panel in PANELS:
        man0 = RUNS / panel / "run_manifest.json"
        if man0.exists():
            code0 = (json.loads(man0.read_text()).get("code") or {})
            if code0.get("git_sha"):
                all_revisions.add(str(code0["git_sha"]))

    problems, published = [], []
    for panel in PANELS:
        man = RUNS / panel / "run_manifest.json"
        if not man.exists():
            problems.append(f"{panel}: no run_manifest.json")
            continue
        srcs = {"run_manifest.json": man}
        ev = _resolve(panel, "evaluate", "eval_report.json")
        if ev is None:
            problems.append(f"{panel}: evaluate stage produced no eval_report.json")
            continue
        # The era-split label source is still ASSERTED, not used for selection: an exactly-resolved
        # report that trained on leaked labels would be the right file with the wrong numbers.
        src_label = str((json.loads(ev.read_text()).get("temporal_split") or {})
                        .get("train_label_source") or "")
        if not src_label.startswith("pre-cutoff"):
            problems.append(f"{panel}: resolved eval report is not era-split ({src_label[:40]!r})")
            continue
        srcs["eval_report.json"] = ev
        sel = _resolve(panel, "selectivity", "selectivity_metrics.json")
        if sel:
            srcs["selectivity_metrics.json"] = sel

        dest = OUT / panel
        for name, src in srcs.items():
            body = src.read_text()
            hits = sorted({h.group(0) for h in IDENTIFYING.finditer(body)})
            if hits:
                problems.append(f"{panel}/{name}: identifying strings {hits} — NOT published")
                continue
            target = dest / name
            # Compare the REDACTED form, not the raw source: the published copy is a filtered copy by
            # design, so a byte comparison against the run tree would report every record as differing.
            want = _redacted(body, all_revisions=all_revisions if name == 'run_manifest.json'
                             else None)
            if args.check:
                if not target.exists():
                    problems.append(f"{panel}/{name}: not published")
                elif target.read_text() != want:
                    problems.append(f"{panel}/{name}: published copy differs from the run tree's "
                                    f"redacted form")
            else:
                dest.mkdir(parents=True, exist_ok=True)
                target.write_text(want)
                published.append(f"{panel}/{name}  ({target.stat().st_size:,} bytes)")

    verb = "CHECKED" if args.check else "PUBLISHED"
    print("=" * 88)
    print(f"PROVENANCE {verb}")
    print("=" * 88)
    for p in published:
        print(f"  {p}")
    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"      - {p}")
        return 1
    print(f"\n  clean — {len(published) or 'all'} record(s) {verb.lower()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
