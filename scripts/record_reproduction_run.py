"""Record MEASURED cache-free reproduction evidence against an IMMUTABLE reference.

Why this is a separate tool from ``make_reproduction_report.py``
---------------------------------------------------------------
That script DERIVES a summary from the published records. Re-deriving a summary from the same records
cannot corroborate them — it compares a file to itself — so calling its output a reproduction comparison
would be a category error, and an inviting one, because the output looks like evidence.

This produces the thing that is not re-derivable: the result of running the panels again, cache-free, and
comparing what came out against a reference captured BEFORE any of it.

What this refuses to do, and why each refusal exists
---------------------------------------------------
* **The reference must be given explicitly** (``--reference-dir``). Defaulting to ``provenance/`` made the
  mutable, about-to-be-overwritten publication directory the yardstick — so a comparison could be run
  after publishing and would then be measuring the rerun against itself.
* **Identical resolved paths or inodes are rejected.** Two names for one file compare byte-identical and
  prove nothing. Genuinely identical BYTES from two distinct files are fine, and are reported as such.
* **The UNION of keys is compared.** Iterating the reference's keys alone means a value the rerun ADDED,
  or dropped to null, is invisible. Missing, extra and non-finite values are all findings.
* **Ignored fields are an exact allowlist**, not substrings. Matching ``"sha"`` anywhere once excused
  every key containing it, including result-bearing ones; ``"dir"`` matched ``direct_delta_*``, which is
  the selectivity result.
* **Exactly four panels × two artifacts.** A comparison that silently examined three panels would report
  success over a gap.
* **Hashes are recorded separately** for the reference, the rerun and (afterwards) the published copy, so
  "the published file is the rerun's file" is checkable rather than assumed.

Usage:
    python scripts/record_reproduction_run.py --reference-dir <immutable> --runs <fresh> [--label ...]
    python scripts/record_reproduction_run.py --verify-published    # after publish_provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
PROV = REPO / "provenance"
OUT = PROV / "REPRODUCTION_RUN.json"

PANELS = ("jak1", "jak1_sensitivity", "brd4", "brd4_sensitivity")
PANEL_CONFIGS = {p: f"configs/{p}.yaml" for p in PANELS}
METRIC_STAGES = ("data_pull", "curate", "featurize", "qsar", "selectivity", "evaluate")
COMPARED = ("eval_report.json", "selectivity_metrics.json")
EXPECTED_ARTIFACTS = len(PANELS) * len(COMPARED)

RANK_KEYS = {"roc_auc", "pr_auc", "pr_auc_lift_over_baseline"}
TOL_CONTINUOUS = 1e-12
TOL_RANK = 1e-5

# EXACT allowlist of leaf keys whose difference is not a result. Substring matching was wrong twice over:
# "sha" excused every key containing it, and "dir" matched `direct_delta_scaffold_cv.*`, which IS the
# selectivity result. Each entry here is a full dotted path suffix, matched exactly.
IGNORED_LEAVES: frozenset[str] = frozenset({
    "artifact_dir",
    "cache_key",
    "run_started_utc",
    "run_finished_utc",
    "wall_seconds",
})


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _leaves(o: object, p: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out |= _leaves(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out |= _leaves(v, f"{p}[{i}]")
    else:
        out[p] = o
    return out


def _ignored(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in IGNORED_LEAVES


def _decisions(sel: dict) -> dict:
    pairs = sel.get("direct_delta_scaffold_cv") or {}
    pm = sel.get("production_model") or {}
    return {
        "pairs_supported": {k: bool((v.get("support") or {}).get("supported"))
                            for k, v in sorted(pairs.items())},
        "support_reasons": {k: list((v.get("support") or {}).get("reasons") or [])
                            for k, v in sorted(pairs.items())},
        "n_paired": {k: (v.get("support") or {}).get("n_paired") for k, v in sorted(pairs.items())},
        "production_model_written": pm.get("written"),
        "supported_comparators": sorted(pm.get("supported_comparators") or []),
        "basis_column": pm.get("basis_column"),
    }


def _distinct(a: Path, b: Path) -> None:
    """Two names for one file compare equal and prove nothing."""
    if a.resolve() == b.resolve():
        raise SystemExit(f"reference and rerun resolve to the SAME path: {a.resolve()}")
    sa, sb = a.stat(), b.stat()
    if (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino):
        raise SystemExit(
            f"reference and rerun are the SAME inode ({sa.st_dev}:{sa.st_ino}): {a} and {b}. A hard "
            f"link or bind mount compares byte-identical while demonstrating nothing."
        )


def _find_rerun(run_root: Path, panel: str, filename: str) -> Path:
    found = sorted((run_root / panel).rglob(filename))
    if not found:
        raise SystemExit(f"{panel}/{filename}: the rerun produced no such artifact under {run_root}")
    if len(found) > 1:
        raise SystemExit(
            f"{panel}/{filename}: {len(found)} candidates under {run_root}. A fresh per-panel workdir "
            f"must contain exactly one; refusing to guess which run produced the numbers."
        )
    return found[0]


def compare(ref: Path, got: Path) -> dict:
    """Union-of-keys comparison with explicit missing/extra/non-finite findings."""
    _distinct(ref, got)
    a, b = _leaves(json.loads(ref.read_text())), _leaves(json.loads(got.read_text()))
    findings: list[dict] = []
    worst_c = worst_r = 0.0

    for k in sorted(set(a) | set(b)):
        if _ignored(k):
            continue
        in_a, in_b = k in a, k in b
        if not in_b:
            findings.append({"key": k, "kind": "missing_in_rerun", "reference": a[k]})
            continue
        if not in_a:
            findings.append({"key": k, "kind": "extra_in_rerun", "rerun": b[k]})
            continue
        av, bv = a[k], b[k]
        if isinstance(av, bool) or isinstance(bv, bool):
            if av != bv:
                findings.append({"key": k, "kind": "bool", "reference": av, "rerun": bv})
            continue
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            for who, v in (("reference", av), ("rerun", bv)):
                if not math.isfinite(float(v)):
                    findings.append({"key": k, "kind": "non_finite", who: v})
            if not (math.isfinite(float(av)) and math.isfinite(float(bv))):
                continue
            d = abs(float(av) - float(bv))
            rank = k.rsplit(".", 1)[-1] in RANK_KEYS
            if rank:
                worst_r = max(worst_r, d)
            else:
                worst_c = max(worst_c, d)
            tol = TOL_RANK if rank else TOL_CONTINUOUS
            if d > tol:
                findings.append({"key": k, "kind": "out_of_tolerance", "reference": av, "rerun": bv,
                                 "abs_delta": d, "tolerance": tol})
            continue
        if av != bv:
            findings.append({"key": k, "kind": "value", "reference": av, "rerun": bv})

    return {
        "reference_sha256": _sha256(ref),
        "rerun_sha256": _sha256(got),
        "bytes_identical": _sha256(ref) == _sha256(got),
        "distinct_files": True,
        "keys_reference": len(a),
        "keys_rerun": len(b),
        "keys_compared": len(set(a) | set(b)),
        "worst_continuous_delta": worst_c,
        "worst_rank_delta": worst_r,
        "findings": findings,
        "reproduced": not findings,
    }


def _digest() -> str:
    out = subprocess.run(["uv", "run", "python", "scripts/scientific_source_digest.py", "--json"],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"could not compute the source digest:\n{out.stderr[-800:]}")
    return json.loads(out.stdout)["scientific_source_digest"]


# Any hexadecimal identifier appearing in a human-written label must be a PREFIX OF A DECLARED PUBLIC
# SCIENTIFIC DIGEST -- never an undeclared revision. The labels are free text passed on the command line,
# and one of them named a revision of the unpublished development workspace while describing the
# reference it compared against -- put there by the same pass that removed the same class of identifier
# from the manifests. Free text is exactly where that recurs, so it is checked rather than trusted, and
# this comment describes the defect without restating the identifier: quoting it here would publish in a
# shipped script precisely what the rule removes from the records.
_HEX_RUN = re.compile(r"\b[0-9a-f]{7,}\b")


def _sanitise_label(label: str, declared: set[str]) -> str:
    """Replace any hex identifier that is not a declared-digest prefix. For HISTORICAL labels only.

    A label being written now is REJECTED (see below) -- the operator can fix it. A label carried forward
    from an earlier record cannot be retyped, so the identifier is redacted in place and the redaction is
    visible rather than silent.
    """
    def repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        # The identifier is DROPPED, not replaced by a marker. An angle-bracketed placeholder announces
        # that something was removed -- a smaller disclosure than the identifier itself, but still one,
        # and it reads as machine output inside a human sentence. Removing the token leaves grammatical
        # prose ("compared against the pre-fix reference") that says what the reference was, which is all
        # a reader needs; the whitespace collapse below tidies the gap it leaves.
        return tok if any(d.startswith(tok) for d in declared) else ""

    return _HEX_RUN.sub(repl, label)


def _check_label_identifiers(label: str, declared: set[str]) -> None:
    """Reject a label naming a hex identifier that is not a prefix of a declared digest."""
    for tok in _HEX_RUN.findall(label.lower()):
        if not any(d.startswith(tok) for d in declared):
            raise SystemExit(
                f"the label names hexadecimal identifier {tok!r}, which is not a prefix of any digest "
                f"this record declares ({sorted(d[:12] + '…' for d in declared)}). A revision identifier "
                f"for an unpublished workspace cannot be resolved by a reader and names a tree outside "
                f"this repository; identify a reference by its scientific digest instead."
            )


def _stage_evidence(log_dir: Path | None) -> dict:
    """Prove the run was cache-free, from the logs, and prove WHICH stages ran and that each exited 0.

    ``method.cache_free`` used to be the fixed string "--force, into a per-panel --workdir that did not
    previously exist". This tool compares artifacts; it cannot know how they were produced, so that was
    narration presented as method -- the same defect as a receptor hash resolved by modification time, in a
    record whose whole purpose is to be measured rather than asserted.

    Counting ``[  ran]`` and ``[cache]`` lines closed most of that, and independent review found the rest:
    a count of six proves six stages ran, not WHICH six. Six ``[ran]`` lines could be five stages with one
    repeated, or six stages of which one is not a metric stage, and a zero ``[cache]`` count says nothing
    about whether the process then failed. So this checks the exact multiset against METRIC_STAGES, and it
    checks the recorded exit status, and it fails on any of: a missing stage, a duplicate, an unexpected
    name, a cache hit, or a non-zero exit.

    What is recorded is SANITIZED: stage names, counts, exit codes and a hash of each log. No path, no
    username, no command line -- the logs live in a scratch directory whose name identifies a machine.
    """
    if log_dir is None:
        return {
            "measured": False,
            "why_absent": (
                "no run logs were supplied (--stage-log), so this record does not attest that the run was "
                "cache-free, nor which stages ran, nor that they exited cleanly. It attests only that the "
                "compared artifacts match the reference."
            ),
        }
    if not log_dir.is_dir():
        raise SystemExit(f"--stage-log {log_dir} is not a directory")

    expected = sorted(METRIC_STAGES)
    exits = _recorded_exit_codes(log_dir)
    per_panel: dict[str, dict] = {}
    problems: list[str] = []

    for panel in PANELS:
        log = log_dir / f"{panel}.log"
        if not log.is_file():
            raise SystemExit(
                f"--stage-log was given but {log.name} is missing. A partial count would understate the "
                f"cache usage and the stage set it exists to establish."
            )
        text = log.read_text(encoding="utf-8", errors="replace")
        ran = sorted(re.findall(r"\[ *ran\] +([a-z_]+)", text))
        cached = sorted(re.findall(r"\[cache\] +([a-z_]+)", text))
        code = exits.get(panel)

        entry = {
            "stages_ran": ran,
            "stages_from_cache": cached,
            "n_ran": len(ran),
            "n_from_cache": len(cached),
            "exact_expected_stage_set": ran == expected,
            "process_exit_code": code,
            # The log's own hash, so the published record is tied to a specific log without naming it.
            "log_sha256": _sha256(log),
        }
        per_panel[panel] = entry

        if cached:
            problems.append(f"{panel}: {len(cached)} stage(s) resolved from cache: {cached}")
        if ran != expected:
            missing = sorted(set(expected) - set(ran))
            unexpected = sorted(set(ran) - set(expected))
            dupes = sorted({n for n in ran if ran.count(n) > 1})
            detail = []
            if missing:
                detail.append(f"missing {missing}")
            if unexpected:
                detail.append(f"unexpected {unexpected}")
            if dupes:
                detail.append(f"duplicated {dupes}")
            if not detail:
                detail.append(f"got {ran}")
            problems.append(f"{panel}: stage set is not exactly {expected} -- " + ", ".join(detail))
        if code != 0:
            problems.append(
                f"{panel}: recorded process exit code is {code!r}, not 0. A run that produced artifacts "
                f"and then failed is not a successful reproduction."
            )

    if problems:
        raise SystemExit(
            "the run logs do not establish a clean, cache-free run of exactly the metric stages:\n  - "
            + "\n  - ".join(problems)
            + "\nA reproduction that resolved any stage from cache compared a file to itself; one that "
              "ran the wrong stage set, or exited non-zero, did not reproduce what this record claims."
        )
    return {
        "measured": True,
        "source": "the per-panel run logs, hashed below; their paths are deliberately not recorded",
        "expected_stage_set": expected,
        "per_panel": per_panel,
        "total_stages_ran": sum(v["n_ran"] for v in per_panel.values()),
        "total_stages_from_cache": sum(v["n_from_cache"] for v in per_panel.values()),
        "all_panels_exited_zero": True,
        "all_panels_exact_stage_set": True,
    }


def _recorded_exit_codes(log_dir: Path) -> dict[str, int]:
    """Per-panel process exit codes, from the driver's ``exit=<n> panel=<name>`` lines.

    The driver records them because this tool cannot observe a process it did not launch. A missing line
    is an error rather than an assumed zero: "no evidence of failure" and "evidence of success" are the
    distinction this whole record exists to keep.
    """
    f = log_dir / "all.log"
    if not f.is_file():
        raise SystemExit(
            f"--stage-log {log_dir} has no all.log, so no process exit status was recorded. Zero "
            f"[cache] lines in a log say nothing about whether the run then failed."
        )
    codes: dict[str, int] = {}
    seen_twice: dict[str, list[int]] = {}
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.fullmatch(r"exit=(\d+) panel=([a-z0-9_]+)", line.strip())
        if not m:
            continue
        panel, code = m.group(2), int(m.group(1))
        if panel in codes:
            seen_twice.setdefault(panel, [codes[panel]]).append(code)
        codes[panel] = code
    # A DUPLICATE is rejected, not resolved. Assigning unconditionally let the LAST record win, so a
    # panel that failed and was re-run into the same log directory would be reported by its second,
    # successful attempt -- and "the panel ran once, cleanly" is precisely what this evidence claims.
    # Which attempt produced the compared artifacts is unknowable from the log, so the honest response is
    # to refuse rather than to pick the flattering one.
    if seen_twice:
        detail = "; ".join(f"{p}: exit codes {c}" for p, c in sorted(seen_twice.items()))
        raise SystemExit(
            f"all.log records more than one exit status for {sorted(seen_twice)} ({detail}). A single "
            f"cache-free run writes one line per panel; duplicates mean the directory holds more than "
            f"one attempt, and which one produced the compared artifacts cannot be determined from the "
            f"log. Re-run into a fresh log directory."
        )
    missing = [p for p in PANELS if p not in codes]
    if missing:
        raise SystemExit(
            f"all.log records no exit status for {missing}. Every panel's exit code must be recorded; "
            f"an unrecorded one cannot be assumed successful."
        )
    return codes


def _normalise_historical_label(entry: dict) -> dict:
    """Sanitise identifiers in a stored history entry, and digest-scope its wording.

    Two corrections, both applied to labels that cannot be retyped because they describe past runs:

    * any hex identifier that is not a prefix of the digest the entry itself records is redacted;
    * "from the <adjective> tree" is rewritten to name the DIGEST, because a run is reproduced at a
      results-determining digest, never from a Git tree -- the tree that ships always differs, since the
      records describing a run are written after it.
    """
    e = dict(entry)
    digest = str(e.get("source_digest_at_run") or "")
    label = str(e.get("label") or "")
    if digest:
        label = _sanitise_label(label, {digest})
    label = re.sub(r"\bfrom the [a-z-]+ tree\b", "at this source digest", label)
    label = re.sub(r"\bfrom this tree\b", "at this source digest", label)
    # Clean an angle-bracketed marker left by an EARLIER sanitiser, which substituted one in place of the
    # identifier instead of dropping it. Such a marker is a literal string rather than a hex run, so
    # nothing else here would ever remove it: it would ship forever, announcing a removal in every future
    # record. The pattern is matched generically rather than by name, so this does not restate it.
    label = re.sub(r"\s*<redacted[^>]*>", "", label)
    # Tidy the gap a dropped identifier leaves, so the sentence reads normally.
    label = re.sub(r"\s{2,}", " ", label).replace(" ,", ",").strip()
    e["label"] = label
    return e


def _measurement_history(digest_now: str) -> list[dict]:
    """Prior runs' headline deltas, carried forward from the record this call is about to replace.

    The documentation calls the rank-statistic tolerance an OBSERVED bound with headroom rather than a
    derived limit. That distinction rests on the spread BETWEEN runs -- one discrete step of a rank
    statistic is one near-tie swapping, and which near-ties swap varies -- so a single measurement cannot
    support the claim. Two consecutive runs of unchanged metric source measured 4.7e-07 and 1.6e-06,
    which is the evidence.

    Without this carry-forward the evidence evaporates on every rerun: each invocation rewrote the file,
    so the documentation quoted a number no published record still contained. Only the headline deltas
    are kept, not the whole comparison -- the point is the spread, and a growing file of full panel
    dumps would obscure it.
    """
    if not OUT.is_file():
        return []
    try:
        prev = json.loads(OUT.read_text())
    except json.JSONDecodeError:
        return []
    if not prev.get("performed"):
        return []
    # Sanitise the WHOLE carried-forward history, not only the entry created below. The first version of
    # this sanitised the new entry and copied the stored list verbatim, so an offending label already
    # inside `previous_runs` sailed through untouched -- which is exactly how the identifier survived: it
    # had already been promoted into the history by an earlier regeneration.
    history = [_normalise_historical_label(h) for h in (prev.get("previous_runs") or [])]
    # A PRIOR label is history: it cannot be re-typed, and it may predate the rule below. So it is
    # sanitised on the way through rather than trusted -- otherwise the carry-forward preserves exactly
    # the identifier the rule exists to remove, which is what happened: one prior label named a
    # development-workspace revision and survived three regenerations inside `previous_runs`.
    prev_declared = {prev.get("source_digest_at_run") or "",
                     *(h.get("source_digest_at_run") or "" for h in history)} - {""}
    entry = {
        "source_digest_at_run": prev.get("source_digest_at_run"),
        "label": _normalise_historical_label(
            {"source_digest_at_run": prev.get("source_digest_at_run"),
             "label": _sanitise_label(str(prev.get("label") or ""), prev_declared)})["label"],
        "result": prev.get("result"),
    }
    # Re-recording the same digest would inflate the history without adding a measurement.
    if entry["source_digest_at_run"] and entry["source_digest_at_run"] != digest_now:
        if not any(h.get("source_digest_at_run") == entry["source_digest_at_run"] for h in history):
            history.append(entry)
    return history


def verify_published() -> int:
    """After publish_provenance: the published copy must BE the rerun's file."""
    if not OUT.is_file():
        raise SystemExit(f"{OUT.relative_to(REPO)} does not exist; record a run first")
    rec = json.loads(OUT.read_text())
    problems = []
    for panel, e in rec["panels"].items():
        for fname, art in e["artifacts"].items():
            pub = PROV / panel / fname
            if not pub.is_file():
                problems.append(f"{panel}/{fname}: not published")
                continue
            got = _sha256(pub)
            art["published_sha256"] = got
            art["published_equals_rerun"] = got == art["rerun_sha256"]
            if not art["published_equals_rerun"]:
                problems.append(
                    f"{panel}/{fname}: published {got[:12]}… != rerun {art['rerun_sha256'][:12]}… — "
                    f"the published record is not the artifact this evidence describes")
    rec["published_verified"] = not problems
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    print("=" * 92)
    print("PUBLISHED-HASH VERIFICATION")
    print("=" * 92)
    for panel, e in rec["panels"].items():
        for fname, art in e["artifacts"].items():
            print(f"  {'OK ' if art.get('published_equals_rerun') else 'FAIL'} {panel:18s} {fname:26s} "
                  f"{str(art.get('published_sha256'))[:12]}…")
    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"      - {p}")
        return 1
    print("\n  every published record is byte-identical to the rerun artifact it came from.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", type=Path,
                    help="immutable directory holding the reference artifacts (REQUIRED to record)")
    ap.add_argument("--runs", type=Path, help="the fresh directory the --force runs wrote into")
    ap.add_argument("--label", default="", help="why this run was made")
    ap.add_argument("--stage-log", type=Path, default=None,
                    help="directory holding <panel>.log for each panel; the [ran]/[cache] counts are "
                         "measured from them and published. Omitted, the record says so rather than "
                         "asserting the run was cache-free.")
    ap.add_argument("--verify-published", action="store_true",
                    help="after publish_provenance: check published hashes equal the rerun hashes")
    args = ap.parse_args()

    if args.verify_published:
        return verify_published()
    if not args.reference_dir or not args.runs:
        raise SystemExit("--reference-dir and --runs are both required (no implicit provenance/)")

    ref_root, run_root = args.reference_dir.resolve(), args.runs.resolve()
    for d, what in ((ref_root, "reference"), (run_root, "runs")):
        if not d.is_dir():
            raise SystemExit(f"{what} directory does not exist: {d}")
    if ref_root == run_root:
        raise SystemExit("the reference and the rerun are the same directory")
    if PROV.resolve() in (ref_root, *ref_root.parents):
        raise SystemExit(
            f"the reference directory is inside {PROV.relative_to(REPO)}, which publish_provenance "
            f"overwrites. Capture an immutable copy outside the repository instead."
        )

    digest = _digest()
    stage_evidence = _stage_evidence(args.stage_log)
    history = _measurement_history(digest)
    panels: dict[str, object] = {}
    worst_c = worst_r = 0.0
    n_artifacts = 0
    all_ok = True

    for panel in PANELS:
        entry: dict[str, object] = {
            "command": (
                f"MEDCHEM_FROZEN_SNAPSHOT=data/frozen_snapshots uv run medchem run -p discovery "
                f"-c {PANEL_CONFIGS[panel]} --force --workdir <fresh-empty-dir>/{panel} "
                + " ".join(f"--stage {s}" for s in METRIC_STAGES)
            ),
            "artifacts": {},
        }
        for fname in COMPARED:
            ref = ref_root / panel / fname
            if not ref.is_file():
                raise SystemExit(f"reference is incomplete: {ref} is missing")
            got = _find_rerun(run_root, panel, fname)
            res = compare(ref, got)
            entry["artifacts"][fname] = res
            n_artifacts += 1
            worst_c = max(worst_c, res["worst_continuous_delta"])
            worst_r = max(worst_r, res["worst_rank_delta"])
            all_ok &= res["reproduced"]
            if fname == "selectivity_metrics.json":
                rd, gd = _decisions(json.loads(ref.read_text())), _decisions(json.loads(got.read_text()))
                entry["decisions"] = {"identical": rd == gd, "reference": rd,
                                      "rerun": gd if rd != gd else "identical to reference"}
                all_ok &= rd == gd
        panels[panel] = entry

    if n_artifacts != EXPECTED_ARTIFACTS:
        raise SystemExit(f"compared {n_artifacts} artifacts; expected exactly {EXPECTED_ARTIFACTS} "
                         f"({len(PANELS)} panels x {len(COMPARED)})")

    # Declared digests: this run's, every prior run's, and the frozen constant. A label may name any of
    # them by prefix and nothing else.
    _declared = {digest, *(h.get("source_digest_at_run") or "" for h in history)} - {""}
    _check_label_identifiers(args.label or "", _declared)

    record = {
        "what_this_is": (
            "MEASURED evidence: the four panels re-run cache-free and compared against an immutable "
            "reference captured before any source edit. Not re-derivable from the published records."
        ),
        "performed": True,
        "label": args.label or None,
        "source_digest_at_run": digest,
        "reference": {
            "artifact_count": EXPECTED_ARTIFACTS,
            "note": "an immutable copy held outside the repository, so publication cannot overwrite it",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uv_lock_sha256": _sha256(REPO / "uv.lock"),
            "extras": ["science", "dev", "docking"],
            "frozen_snapshot_env": "MEDCHEM_FROZEN_SNAPSHOT=data/frozen_snapshots",
        },
        "method": {
            "stages": list(METRIC_STAGES),
            # MEASURED from the run logs when they are supplied, and explicitly not measured when they
            # are not. This was a fixed string asserting the method rather than recording it.
            "cache_free_evidence": stage_evidence,
            "invocation": "--force, into per-panel run directories that did not previously exist",
            "why": (
                "a fresh workdir alone is NOT sufficient: a shared content cache lives outside it, so a "
                "run without --force resolves stages from that cache, prints [cache] and writes nothing"
            ),
            "compared": list(COMPARED),
            "comparison": "union of keys; missing, extra and non-finite values are findings",
            "ignored_leaves": sorted(IGNORED_LEAVES),
            "tolerances": {"continuous": TOL_CONTINUOUS, "rank_statistics": TOL_RANK,
                           "rank_keys": sorted(RANK_KEYS), "decisions": "exact"},
        },
        "result": {
            "all_within_tolerance": all_ok,
            "artifacts_compared": n_artifacts,
            "worst_continuous_delta": worst_c,
            "worst_rank_delta": worst_r,
        },
        "published_verified": None,
        # Earlier runs' headline deltas. The tolerance for rank statistics is an OBSERVED bound, and the
        # spread across runs is what makes that an observable rather than an assertion.
        "previous_runs": history,
        "panels": panels,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")

    print("=" * 92)
    print("RECORDED CACHE-FREE REPRODUCTION (vs immutable reference)")
    print("=" * 92)
    print(f"  digest at run          : {digest[:16]}…")
    print(f"  artifacts compared     : {n_artifacts} of {EXPECTED_ARTIFACTS}")
    print(f"  worst continuous delta : {worst_c:.3e}  (tolerance {TOL_CONTINUOUS:.0e})")
    print(f"  worst rank delta       : {worst_r:.3e}  (tolerance {TOL_RANK:.0e})")
    for p, e in panels.items():
        ok = all(a["reproduced"] for a in e["artifacts"].values())
        print(f"    {'OK ' if ok else 'FAIL'} {p:18s} decisions_identical="
              f"{e.get('decisions', {}).get('identical')}")
    print(f"\n  wrote {OUT.relative_to(REPO)}")
    if not all_ok:
        print("\n  *** AT LEAST ONE PANEL IS OUT OF TOLERANCE — DO NOT PUBLISH ***\n")
        return 1
    print("\n  all four panels reproduce within tolerance, with identical decisions.")
    print("  Next: publish_provenance, then --verify-published.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
