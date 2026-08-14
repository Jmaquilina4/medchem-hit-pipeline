"""Compute the scientific-source digest: the durable identity of what determines the results.

Why this exists
---------------
A git SHA identifies one tree, and neither of the two SHAs in play can bridge the analysis to the
release. The revision that executed the analysis does not exist in the exported tree; the exported
tree's own revision did not exist when the results were computed. (This paragraph previously fused
those two clauses into one ungrammatical sentence, which is worth fixing in a file whose whole subject
is being precise about identity.)

What CAN bridge them is a digest over the things that actually determine a result: the source of the
modules the pipeline executes, and the configuration that parameterises them. That digest is
reproducible from either tree, so it lets anyone confirm that this repository contains the same
science as the run that produced the frozen numbers — without either SHA.

Four identities, kept separate on purpose
-----------------------------------------
=========================== ==========================================================================
analysis_run_sha            DELIBERATELY NOT RECORDED. The source revision that executed the runs
                            belongs to a workspace this release does not publish, so a reader can
                            neither resolve nor check it, while naming it points at a tree outside
                            this repository. What it evidenced -- that all four panels ran at one
                            clean revision -- is published as a boolean instead.
scientific_source_digest    THIS value. Hash of results-determining source + configs. The bridge.
public_release_sha / tag    assigned after the sanitized export is committed; unknown until then.
manifest_tool_version       which generation of the manifest tooling wrote the record.
=========================== ==========================================================================

Conflating any two of these is how a provenance chain quietly breaks. A reader given only the
analysis-run SHA and pointed at this repository gets nothing from it.

What is hashed, and what is deliberately not
--------------------------------------------
Hashed, and normalised so that only *meaning* counts:

* every first-party module reachable from the pipeline entry points, each hashed from its parsed AST
  with docstrings stripped -- the same normalisation the stage cache uses. Comment and docstring edits
  are free; a real behavioural change anywhere in the closure is not.
* the four frozen configs, hashed from their PARSED content with sorted keys -- not their bytes. An
  earlier version byte-hashed them, so correcting a wrong number in a YAML *comment* perturbed the
  digest. A digest that moves when prose moves cannot serve as a bridge.
* ``COHORT_SPEC_VERSION`` and the package version, read textually from source.

**The digest is INTERPRETER-SCOPED, and that is a real limitation rather than a caveat.** Module hashes
come from ``ast.dump`` of the parsed, docstring-stripped tree, and that output changes between CPython
minor versions: measured on this project's own source, 3.11, 3.12 and 3.13/3.14 give three different
values for byte-identical input. So a digest is a bridge *within a stated interpreter*, not an absolute
identity. The interpreter is recorded alongside it, ``.python-version`` pins it to 3.14, and
``--check-identity`` reports an interpreter mismatch as the cause instead of blaming the content.

The same applies to stage cache keys, which use the same normalisation: a cache built on one interpreter
will simply miss on another. That is safe -- a miss recomputes -- but it means "replay the cache" is
environment-bound in a way "re-run from the snapshot" is not.

Not hashed: documentation, figures, tests, scripts, the README. Those matter, but they cannot change a
computed number, and folding them in would make the digest churn on every prose edit -- which would
destroy the one property that makes it useful as a bridge.

Usage:
    python scripts/scientific_source_digest.py                  # digest + component breakdown
    python scripts/scientific_source_digest.py --json           # machine-readable
    python scripts/scientific_source_digest.py --expect <hex>    # exit 1 unless it matches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# The configs that produced the four frozen panels. Order is fixed so the digest is stable.
FROZEN_CONFIGS = (
    "configs/jak1.yaml",
    "configs/jak1_sensitivity.yaml",
    "configs/brd4.yaml",
    "configs/brd4_sensitivity.yaml",
)
# The revision that executed all four analyses. It carries the SAME results-determining digest the
# release ships -- not the same Git tree, which it cannot be, since the records and documents describing
# a run are written after it -- so the analysis and release digests are identical and there is no delta
# to explain. See FROZEN_ANALYSIS_DIGEST below.
# NOT the revision identifier itself. Publishing a SHA for a workspace that is not published gives a
# reader nothing they can resolve or check, while still naming a tree outside this repository -- so
# what is recorded is the PROPERTY it evidenced. The durable identity is the digest below, which is
# recomputable from this tree by anyone.
ANALYSIS_RUN_SHA = None
ANALYSIS_RUN_SCOPE = ("the frozen runs executed at a single clean revision of a source workspace that "
                      "is not published; its identifier is deliberately not recorded here")
MANIFEST_TOOL_VERSION = 2

# The digest of the SOURCE that actually produced the frozen numbers. All four analyses were re-run on a
# clean tree against the published snapshot, at this digest -- so this value EQUALS the digest the release
# computes and there is no delta to explain. "At this digest", not "from this tree": the shipped Git tree
# additionally contains the records, figures and documents that describe those runs, all written
# afterwards, none of which the digest covers. Earlier releases
# carried a three-component delta from a post-run cohort rename; re-running removed the need to explain
# it at all, which is the better outcome.
#
# Recorded as a constant because that revision does not resolve in the exported tree, so this is
# the only way a reader can check the claim from the public tree alone:
#
#     python scripts/scientific_source_digest.py            # must equal FROZEN_ANALYSIS_DIGEST
FROZEN_ANALYSIS_DIGEST = "e17ad66dc8c3f4dac3e651a6b04430475b1c215e45fec8823b376e3ccf234158"
FROZEN_ANALYSIS_SPEC = "1.2"
# EMPTY, and now empty by construction rather than by argument.
#
# The four panels were re-run cache-free at the SAME RESULTS-DETERMINING SOURCE AND CONFIGURATION DIGEST
# this release ships -- `--force` into per-panel workdirs that did not previously exist -- so there is no
# delta to declare and none to explain.
#
# Not "from this tree". The Git tree at the moment of the rerun and the Git tree that ships are not the
# same object and cannot be: the provenance records, the figures and the documents are regenerated AFTER
# the run they describe, and any documentation fix lands later still. The digest is what is invariant
# across that, which is precisely why it is the identity worth publishing.
#
# "Same digest", not "same tree": the digest covers 37 modules and the semantically-hashed configs, and
# deliberately not medchem.cli, scripts/, docs/ or tests/. Files outside that closure did change after the
# runs -- the documentation and the tooling that checks it, necessarily -- and calling that "the same
# tree" would claim more than was measured. What is claimed is what the digest is: nothing that
# determines a result differs.
#
# A non-empty value here would mean a results-determining file changed after the runs, which is exactly
# what should require an explanation. The measured evidence for the equality is in
# provenance/REPRODUCTION_RUN.json: reference and rerun hashes, per-key deltas against the per-family
# tolerances, and the exact support and production-model decisions from both.
#
# NOTE on isolation, learned while producing that evidence: a fresh `--workdir` is NOT sufficient for a
# cache-free run. A shared content cache lives outside the workdir, so a run without `--force` resolves
# stages from it and reports `[cache]` while writing nothing -- which looks like a successful
# reproduction and is not one. `--force` is what makes the run cache-free; the workdir only isolates
# where the output lands.
EXPECTED_RENAME_DELTA: tuple[str, ...] = ()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _interpreter() -> str:
    """CPython major.minor -- the scope within which a digest is comparable."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _identity_field(name: str) -> str | None:
    """Read a value already recorded in ``provenance/IDENTITY.json``, or None if absent.

    The display used to hardcode None for the release identities, so a reader running this inside a
    published tree was told the release was unidentified while the file beside it held the value. Report
    what is RECORDED, not only what this script can compute from scratch.
    """
    f = REPO / "provenance" / "IDENTITY.json"
    if not f.is_file():
        return None
    try:
        d = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    block = d.get(name)
    if isinstance(block, dict):
        return block.get("sha") or block.get("tag")
    return block if isinstance(block, str) else None


# Entry points of the executable surface. The closure is walked from here rather than from an imported
# registry, so the digest can be computed for ANY tree -- including a checkout of the analysis
# SHA, which is the whole point. Importing the target tree instead would resolve modules through
# whatever is installed in the active environment, not through the tree being measured.
# `medchem.cli` is deliberately NOT an entry point: it was one until editing a --help string moved the
# digest, and a help string cannot change a result, so a digest that moves for one is measuring the wrong
# surface. The CLI is presentation over the stages, and it imports them anyway.
#
# The stage closure alone is NOT enough, though: it omits three modules that
# do determine results:
#
#   medchem.config           supplies every default and every validated value the stages read. A changed
#                            default silently changes a run whose config omits that key.
#   medchem.pipeline.runner  decides what executes, in what order, and computes the cache keys.
#   medchem.pipeline.cache   decides what is REUSED rather than recomputed -- a change here can serve a
#                            stale artifact, which changes a result without changing any stage.
#
# None is reachable by import from `medchem.stages`: the stage modules import the decorator, not the
# machinery that drives them. So a green digest attested to the scientific stage code while saying nothing
# about the execution path around it. `medchem.pipeline.registry` arrives through the runner.
#
# The cost of the wider closure is that the digest now moves for a change in the runner or the config
# layer that cannot affect a result. That is the correct trade in this direction: a digest that misses a
# results-determining change is unsound, while one that moves too eagerly is merely inconvenient.
ENTRY_MODULES = (
    "medchem.stages",
    "medchem.config",
    "medchem.pipeline.runner",
    "medchem.pipeline.cache",
)
# NB `medchem.pipeline.registry` was listed here and DOES NOT EXIST. The closure records an unresolvable
# entry as the sentinel "no-source" rather than dropping it -- correct, because a file that disappears
# should move the digest -- but the effect was a phantom counted toward the covered-module total, so the
# coverage claim was one module too generous. Stage registration and dependency resolution actually live
# in `medchem.pipeline.stage` (reached through the stage closure) and `medchem.pipeline.runner` (an entry
# point above), so nothing is uncovered by its removal.


def _module_path(root: Path, name: str) -> Path | None:
    rel = name.replace(".", "/")
    for cand in (root / "src" / f"{rel}.py", root / "src" / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _closure_from_files(root: Path, entries: tuple[str, ...]) -> set[str]:
    """First-party import closure, walked by parsing files under ``root/src``.

    Mirrors ``medchem.pipeline.cache._first_party_closure`` but resolves names against a directory
    rather than against ``sys.modules``. A module that cannot be located is still recorded (as
    ``no-source`` by the caller) rather than skipped, so a missing file changes the digest instead of
    silently shrinking it.
    """
    import ast

    seen: set[str] = set()
    queue = [e for e in entries]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        f = _module_path(root, name)
        if f is None:
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            found: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("medchem"):
                found.append(node.module)
                # `from medchem.pkg import mod` may name a submodule, not an attribute.
                found += [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                found += [a.name for a in node.names if a.name.startswith("medchem")]
            for dep in found:
                if dep not in seen and _module_path(root, dep) is not None:
                    queue.append(dep)
    return seen


def _config_hash(path: Path) -> str:
    """Hash a config's SEMANTIC content, not its bytes.

    Byte-hashing was wrong for the same reason hashing raw module text was wrong for the cache: a
    comment cannot change a result, but it changes the bytes. Correcting a wrong number in a YAML
    comment perturbed all four config hashes and therefore the whole digest — which would have made the
    digest useless as a bridge, since any prose fix would look like a scientific change.

    Parsed and re-serialised with sorted keys, so key order and formatting are also neutralised.
    """
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _sha256_bytes(
        json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    )


def _spec_version_from_source(root: Path) -> str:
    """Read COHORT_SPEC_VERSION textually, so no import of the target tree is needed."""
    import ast

    f = _module_path(root, "medchem.cohorts")
    if f is None:
        return "unknown"
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "COHORT_SPEC_VERSION":
                    if isinstance(node.value, ast.Constant):
                        return str(node.value.value)
    return "unknown"


def _package_version_from_source(root: Path) -> str:
    import ast

    f = _module_path(root, "medchem")
    if f is None:
        return "unknown"
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__version__":
                    if isinstance(node.value, ast.Constant):
                        return str(node.value.value)
    return "unknown"


def compute(root: Path | None = None) -> dict:
    from medchem.pipeline.cache import _normalised_module_hash

    root = root or REPO
    modules = _closure_from_files(root, ENTRY_MODULES)
    module_hashes: dict[str, str] = {}
    for name in sorted(modules):
        f = _module_path(root, name)
        module_hashes[name] = _normalised_module_hash(f) if f else "no-source"

    missing = [c for c in FROZEN_CONFIGS if not (root / c).exists()]
    if missing:
        raise SystemExit(f"missing frozen configs, digest would be incomplete: {missing}")
    config_hashes = {c: _config_hash(root / c) for c in FROZEN_CONFIGS}

    __version__ = _package_version_from_source(root)
    COHORT_SPEC_VERSION = _spec_version_from_source(root)

    payload = json.dumps(
        {
            "schema": "scientific-source-digest/1",
            "package_version": __version__,
            "cohort_spec_version": COHORT_SPEC_VERSION,
            "modules": module_hashes,
            "configs": config_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "scientific_source_digest": _sha256_bytes(payload.encode()),
        "root": str(root),
        "schema": "scientific-source-digest/1",
        "package_version": __version__,
        "cohort_spec_version": COHORT_SPEC_VERSION,
        "n_modules": len(module_hashes),
        "n_configs": len(config_hashes),
        "modules": module_hashes,
        "configs": config_hashes,
        "identities": {
            "analysis_run_sha": ANALYSIS_RUN_SHA,   # None by design; see ANALYSIS_RUN_SCOPE
            "analysis_run_scope": ANALYSIS_RUN_SCOPE,
            "frozen_analysis_digest": FROZEN_ANALYSIS_DIGEST,
            "frozen_analysis_cohort_spec": FROZEN_ANALYSIS_SPEC,
            "frozen_vs_release_delta": list(EXPECTED_RENAME_DELTA),
            # Accurate to the CURRENT delta rather than a fixed sentence. It described a spec
            # 1.0->1.1 rename that no longer applies: the analyses were re-run at the tree that
            # ships, so the delta is empty, and a note asserting a difference beside a field
            # showing none is the kind of contradiction a provenance record exists to prevent.
            "frozen_vs_release_note": (
                "the release tree and the analysis tree produce the SAME digest, so no delta "
                "needs explaining"
                if not EXPECTED_RENAME_DELTA else
                f"the release digest differs from the frozen-analysis digest in these module(s) only: "
                f"{', '.join(EXPECTED_RENAME_DELTA)}. Validation was added after the runs; the frozen "
                f"configs load identically and every stage cache key is unchanged, so no frozen result "
                f"is affected"
            ),
            "analysis_run_sha_note": (
                # Two defects here, and the first is why source-fragment greps never caught it: the text
                # was split as "...in a " + "exported tree...", so neither fragment contained the error
                # and only the CONCATENATED output read "in a exported tree". A test now asserts against
                # the generated JSON rather than the source.
                "deliberately not recorded: the revision that executed the frozen runs belongs to a "
                "workspace this release does not publish, so it would not resolve in an exported tree "
                "and a reader could neither check nor use it. The digest above is the durable identity, "
                "and code.single_clean_revision publishes the property that revision evidenced"
            ),
            # NOT "public_release_sha": that name implied one value identifies the release, when the
            # recorded commit CANNOT be the release HEAD. Writing the SHA into IDENTITY.json is itself a
            # commit, so the file can only ever hold the commit BEFORE it -- a commit cannot record its
            # own SHA. Two separate fields say so plainly.
            "initial_content_commit": _identity_field("initial_content_commit"),
            "initial_content_commit_note": (
                "first commit holding the content. Release HEAD is one commit "
                "LATER, because recording this value is itself a commit"
            ),
            "release_tag": _identity_field("release_tag"),
            "release_tag_note": "assigned at publication; a separate authorised step",
            "manifest_tool_version": MANIFEST_TOOL_VERSION,
        },
        "excluded_from_digest": (
            "docs, figures, tests, scripts, README -- they cannot change a computed number, and "
            "including them would make the digest churn on prose edits"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--expect", default=None, help="exit 1 unless the digest equals this value")
    ap.add_argument("--check-identity", action="store_true",
                    help="exit 1 unless provenance/IDENTITY.json records THIS tree's digest")
    ap.add_argument("--release", action="store_true",
                    help="with --check-identity, additionally require a non-null release tag")
    ap.add_argument("--write-identity", action="store_true",
                    help="write provenance/IDENTITY.json with all four separate identities")
    ap.add_argument("--root", default=None,
                    help="tree to measure (default: this repo). Point at a worktree of the analysis "
                         "SHA to obtain the digest the frozen results were produced under.")
    args = ap.parse_args()

    d = compute(Path(args.root).resolve() if args.root else None)
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print("=" * 90)
        print("SCIENTIFIC-SOURCE DIGEST")
        print("=" * 90)
        print(f"\n  digest                  {d['scientific_source_digest']}")
        print(f"  package version         {d['package_version']}")
        print(f"  cohort spec             {d['cohort_spec_version']}")
        print(f"  covers                  {d['n_modules']} modules, {d['n_configs']} configs")
        print(f"  measured tree           {d['root']}")
        print("\n  separate identities:")
        for k, v in d["identities"].items():
            if k.endswith("_note"):
                continue
            # `analysis_run_sha` is None because it is WITHHELD, not because it is pending. Printing
            # "(not yet assigned)" for it invited someone to helpfully assign it later.
            absent = ("deliberately not recorded" if k == "analysis_run_sha" else "(not yet assigned)")
            print(f"    {k:26s} {v if v is not None else absent}")
        print("\n  (the analysis workspace's revision is deliberately not recorded; the")
        print("   digest above is the identity that is recomputable from any tree)")
        print()

    if args.write_identity:
        out = REPO / "provenance" / "IDENTITY.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "what_this_is": (
                "Four SEPARATE identities for this work. They are not interchangeable, and a reader "
                "given only one of them cannot verify the others."
            ),
            "analysis_run_sha": {
                "value": ANALYSIS_RUN_SHA,
                "why_absent": ANALYSIS_RUN_SCOPE,
                "scope": "the source revision that executed the frozen runs",
                "resolves_in_public_repository": False,
            },
            "scientific_source_digest": {
                "frozen_analysis": FROZEN_ANALYSIS_DIGEST,
                "frozen_analysis_cohort_spec": FROZEN_ANALYSIS_SPEC,
                "this_tree": d["scientific_source_digest"],
                "this_tree_cohort_spec": d["cohort_spec_version"],
                "differs_by": list(EXPECTED_RENAME_DELTA),
                # Only explain a delta that exists. This field described an obsolete spec-1.1 rename
                # long after re-running the panels removed the delta entirely, so the record contradicted
                # itself: differs_by empty, both digests equal, and a paragraph explaining a difference.
                "why_it_differs": (
                    None if d["scientific_source_digest"] == FROZEN_ANALYSIS_DIGEST else
                    "the results-determining source or configuration differs from the tree that produced "
                    "the frozen numbers; see differs_by for the components"
                ),
                "identical": d["scientific_source_digest"] == FROZEN_ANALYSIS_DIGEST,
                "identical_note": (
                    "the analysis and release digests are identical: the four panels were re-run at the "
                    "same results-determining source and configuration digest this release ships, so "
                    "there is no delta to explain. NOT the same Git tree -- documentation, tests, "
                    "scripts and comments sit outside the digest and are regenerated or corrected after "
                    "the run they describe"
                    if d["scientific_source_digest"] == FROZEN_ANALYSIS_DIGEST else
                    "the analysis and release digests differ; this REQUIRES an explanation"
                ),
                "recompute": "python scripts/scientific_source_digest.py [--root TREE]",
            },
            # A commit cannot record its own SHA, so these are separate identities and the export fills
            # the first one in a follow-up commit.
            "initial_content_commit": {
                "sha": None,
                "note": ("first commit holding the content. Release HEAD is one commit LATER, "
                         "because writing this value is itself a commit -- they are not the same "
                         "identity and are deliberately not named as if they were."),
            },
            "release_tag": {
                "tag": None,
                "note": "assigned at publication; a separate authorised step",
            },
            "interpreter": {
                # Comparison key is major.minor, because that is the granularity at which ast.dump
                # output was measured to change. The exact patch is recorded too, and pinned by
                # .python-version, because the run manifests name a patch version and CI floated off it.
                "python": _interpreter(),
                "python_exact": f"{sys.version_info.major}.{sys.version_info.minor}."
                                f"{sys.version_info.micro}",
                "pinned_by": ".python-version",
                "note": ("the digest is comparable only under this CPython minor version: ast.dump "
                         "output differs across 3.11 / 3.12 / 3.13+ for identical source. Pinned by "
                         ".python-version."),
            },
            "manifest_tool_version": MANIFEST_TOOL_VERSION,
        }, indent=2) + "\n")
        print(f"  wrote {out.relative_to(REPO)}")

    if args.check_identity:
        # Nothing checked this before, and it drifted: IDENTITY.json was written before later edits to
        # digest-covered files, so the one value the README calls "the durable bridge" failed the exact
        # recompute command the README tells the reader to run. A provenance claim nobody verifies is a
        # claim that will be wrong eventually.
        ident = REPO / "provenance" / "IDENTITY.json"
        if not ident.exists():
            print("  provenance/IDENTITY.json is absent; run with --write-identity")
            return 1
        recorded = json.loads(ident.read_text())
        # Check the INTERPRETER first. Otherwise a version difference is reported as a content
        # difference, which is what happened on a public CI runner: the tree was a clean checkout and the
        # message said the results-determining source differed.
        rec_py = (recorded.get("interpreter") or {}).get("python")
        if rec_py and rec_py != _interpreter():
            print(f"  IDENTITY.json was recorded under CPython {rec_py}; this is {_interpreter()}.")
            print("  The digest is interpreter-scoped -- ast.dump output differs across minor versions --")
            print("  so this is NOT evidence that the source differs. Run under the recorded interpreter")
            print("  (.python-version pins it) or re-record deliberately.")
            return 1
        rec = recorded.get("scientific_source_digest", {})
        got, want = d["scientific_source_digest"], rec.get("this_tree")
        if want != got:
            print(f"  IDENTITY.json records this_tree = {want}")
            print(f"  this tree actually hashes to      {got}")
            print("  Regenerate: python scripts/scientific_source_digest.py --write-identity")
            return 1
        if rec.get("frozen_analysis") != FROZEN_ANALYSIS_DIGEST:
            print("  IDENTITY.json's frozen_analysis digest does not match the recorded constant")
            return 1
        if args.release:
            # A published tree that says "no tag assigned" cannot name its own release. Enforced only in
            # release mode: a development tree legitimately has no tag.
            tag = (recorded.get("release_tag") or {}).get("tag")
            if not tag:
                print("  --release: IDENTITY.json records no release_tag. A published tree must name its")
                print("  own release. The release tag is assigned by the export tooling.")
                return 1
            print(f"  release tag recorded: {tag}")
        print(f"  IDENTITY.json matches this tree ({got[:16]}…) and the frozen-analysis constant.")
        return 0

    if args.expect and args.expect != d["scientific_source_digest"]:
        print(f"  MISMATCH: expected {args.expect}")
        print(f"            got      {d['scientific_source_digest']}")
        print("  The results-determining source or configuration differs.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
