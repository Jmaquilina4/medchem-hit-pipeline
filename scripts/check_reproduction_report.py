"""Verify provenance/REPRODUCTION.json against the records it claims to summarise.

A derived file that nobody checks is a file that goes stale, and this one is the most quotable artifact in
the repository: it states the commands, the environment, the source digest, the tolerances and the exact
support decisions. If it drifts from the provenance records beside it, it becomes a confident, wrong
answer to every question a reader has.

So this re-derives the report from the published records and compares. It runs from a fresh clone with no
run tree, which is the tree that most needs it.

Usage:
    python scripts/check_reproduction_report.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "provenance" / "REPRODUCTION.json"


def _maker():
    spec = spec_from_file_location("_mrr", REPO / "scripts" / "make_reproduction_report.py")
    assert spec is not None and spec.loader is not None
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flat(o: object, p: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out |= _flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out |= _flat(v, f"{p}[{i}]")
    else:
        out[p] = o
    return out


EXPECTED_PANELS = ("jak1", "jak1_sensitivity", "brd4", "brd4_sensitivity")
EXPECTED_STAGES = ("curate", "data_pull", "evaluate", "featurize", "qsar", "selectivity")
EXPECTED_ARTIFACTS = 8
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_RUN = re.compile(r"\b[0-9a-f]{7,}\b")


def _check_measured_evidence(run: dict, stored: dict) -> list[str]:
    """Re-derive every condition the measured block claims, from the block itself and from disk.

    Deliberately does NOT read a summary flag as proof of the thing it summarises. ``measured: true``,
    ``all_panels_exited_zero`` and ``all_within_tolerance`` are all fields a hand-edited or half-written
    record could carry while the per-panel detail says otherwise, so each is recomputed from that detail.
    The published-file hashes are read from disk, because a record asserting a hash it no longer matches
    is the one failure that makes every other check vacuous.
    """
    bad: list[str] = []
    prov = REPO / "provenance"

    digest_now = (stored.get("source_identity") or {}).get("scientific_source_digest")
    declared = {run.get("source_digest_at_run") or "",
                *(h.get("source_digest_at_run") or "" for h in run.get("previous_runs") or [])} - {""}
    if run.get("source_digest_at_run") != digest_now:
        bad.append(f"the recorded reproduction ran at digest {run.get('source_digest_at_run')}, but this "
                   f"tree is {digest_now} -- the evidence describes different source")

    # `performed` must be EXACTLY True. Reached here via a truthiness test, so a record carrying
    # `"performed": "yes"` -- or 1, or a non-empty string from a hand edit -- was treated as a performed
    # run. A flag that gates every check below deserves an identity comparison, not a truthy one.
    if run.get("performed") is not True:
        bad.append(f"cache_free_reproduction.performed is {run.get('performed')!r}, not exactly true; a "
                   f"truthy value is not evidence that a run happened")

    # A hex identifier in free text must be a prefix of a declared digest, never an undeclared revision.
    # EVERY label, current and historical: the identifier that reached publication did so inside
    # `previous_runs`, which the first version of this check did not look at.
    labelled = [("label", str(run.get("label") or ""))]
    labelled += [(f"previous_runs[{i}].label", str(h.get("label") or ""))
                 for i, h in enumerate(run.get("previous_runs") or [])]
    for where, text in labelled:
        for tok in _HEX_RUN.findall(text.lower()):
            if not any(d.startswith(tok) for d in declared):
                bad.append(f"{where} names hexadecimal identifier {tok!r}, which is not a prefix of any "
                           f"digest this record declares. A revision identifier for an unpublished "
                           f"workspace must not appear in published material")

    ev = (run.get("method") or {}).get("cache_free_evidence") or {}
    if not ev.get("measured"):
        bad.append("the measured block does not claim measured stage evidence, so the record does not "
                   "establish that the run was cache-free")
        return bad

    per_panel = ev.get("per_panel") or {}
    if sorted(per_panel) != sorted(EXPECTED_PANELS):
        bad.append(f"stage evidence covers panels {sorted(per_panel)}, expected {sorted(EXPECTED_PANELS)}")
    ran = cached = 0
    for panel, e in sorted(per_panel.items()):
        stages = sorted(e.get("stages_ran") or [])
        if stages != sorted(EXPECTED_STAGES):
            bad.append(f"{panel}: stages ran {stages}, expected exactly {sorted(EXPECTED_STAGES)} once each")
        if e.get("stages_from_cache"):
            bad.append(f"{panel}: {len(e['stages_from_cache'])} stage(s) resolved from cache")
        if e.get("process_exit_code") != 0:
            bad.append(f"{panel}: recorded process exit code {e.get('process_exit_code')!r}, not 0")
        if not _SHA256.match(str(e.get("log_sha256") or "")):
            bad.append(f"{panel}: log_sha256 {e.get('log_sha256')!r} is not a 64-hex digest, so the "
                       f"evidence is not tied to a specific log")
        ran += len(stages)
        cached += len(e.get("stages_from_cache") or [])
    want_ran = len(EXPECTED_PANELS) * len(EXPECTED_STAGES)
    if ran != want_ran:
        bad.append(f"{ran} stage(s) ran in total, expected {want_ran}")
    if cached:
        bad.append(f"{cached} stage(s) resolved from cache; a cache hit compares a file to itself")

    # STORED SUMMARIES MUST AGREE WITH THE RECOMPUTED DETAIL. The detail above is authoritative -- and the
    # summaries are what a reader skims, so a record whose headline says 24/0 while its panels say
    # otherwise is worse than one with no headline at all. Checking agreement rather than trusting either.
    for field, recomputed in (("total_stages_ran", ran), ("total_stages_from_cache", cached)):
        if ev.get(field) != recomputed:
            bad.append(f"cache_free_evidence.{field} is {ev.get(field)!r}, but the per-panel detail gives "
                       f"{recomputed} -- the summary and the evidence disagree")
    if ev.get("all_panels_exited_zero") is not True:
        bad.append(f"cache_free_evidence.all_panels_exited_zero is "
                   f"{ev.get('all_panels_exited_zero')!r}, not exactly true")
    if ev.get("all_panels_exact_stage_set") is not True:
        bad.append(f"cache_free_evidence.all_panels_exact_stage_set is "
                   f"{ev.get('all_panels_exact_stage_set')!r}, not exactly true")

    # Artifacts: exactly eight, none with findings, decisions identical, and the PUBLISHED file on disk
    # must hash to both the recorded published hash and the recorded rerun hash.
    n_art = 0
    for panel, e in sorted((run.get("panels") or {}).items()):
        for fname, art in sorted((e.get("artifacts") or {}).items()):
            n_art += 1
            if art.get("findings"):
                bad.append(f"{panel}/{fname}: {len(art['findings'])} comparison finding(s) recorded")
            if not art.get("reproduced"):
                bad.append(f"{panel}/{fname}: recorded as not reproduced")
            pub = prov / panel / fname
            if not pub.is_file():
                bad.append(f"{panel}/{fname}: recorded as published but absent from provenance/")
                continue
            on_disk = hashlib.sha256(pub.read_bytes()).hexdigest()
            for field in ("published_sha256", "rerun_sha256"):
                if art.get(field) != on_disk:
                    bad.append(f"{panel}/{fname}: the published file hashes to {on_disk[:12]}… but the "
                               f"record's {field} is {str(art.get(field))[:12]}… -- the published record "
                               f"is not the artifact this evidence describes")
        dec = e.get("decisions") or {}
        if not dec.get("identical"):
            bad.append(f"{panel}: recorded decisions are not identical to the reference")
    if n_art != EXPECTED_ARTIFACTS:
        bad.append(f"{n_art} artifact(s) compared, expected exactly {EXPECTED_ARTIFACTS}")

    if not (run.get("result") or {}).get("all_within_tolerance"):
        bad.append("the recorded reproduction reports at least one value out of tolerance")
    res = run.get("result") or {}
    if res.get("artifacts_compared") != EXPECTED_ARTIFACTS:
        bad.append(f"result.artifacts_compared is {res.get('artifacts_compared')!r}, but "
                   f"{EXPECTED_ARTIFACTS} artifacts are required and {n_art} were found in the detail")
    # `--verify-published` sets this after comparing each published file to its recorded hash. The
    # on-disk recomputation above remains authoritative; this catches a record published without that
    # step ever having been run.
    if run.get("published_verified") is not True:
        bad.append(f"published_verified is {run.get('published_verified')!r}, not exactly true -- the "
                   f"published-hash verification step has not been recorded as passing")
    return bad


def main() -> int:
    print("=" * 92)
    print("REPRODUCTION REPORT")
    print("=" * 92)
    if not REPORT.is_file():
        print(f"\n  {REPORT.relative_to(REPO)} is missing.")
        print("  Regenerate: python scripts/make_reproduction_report.py\n")
        return 1

    stored = json.loads(REPORT.read_text())
    fresh = _moduleless_rebuild()
    problems: list[str] = []

    # Fields that must NOT be re-derived, for two different reasons:
    #
    #  * the running interpreter's patch version is an observation about this machine, not a claim about
    #    the frozen results;
    #  * `cache_free_reproduction` is MEASURED evidence from an actual rerun. Re-deriving it is
    #    impossible by construction -- that is what makes it evidence -- so this check verifies its
    #    internal coherence instead, below, and never attempts to reproduce its contents.
    #
    # BUT "not re-derivable" was read as "not checkable", and that was wrong. The block is a VERBATIM
    # COPY of provenance/REPRODUCTION_RUN.json, and a copy can always be compared against its original.
    # Skipping the whole subtree meant the one part of this file copied from elsewhere was the one part
    # nothing compared -- and it drifted: `previous_runs` was appended to the source file and the copy
    # kept an older shape, while this check still printed "every field re-derives".
    VOLATILE = {".environment.python_running_now"}
    MEASURED_PREFIX = ".cache_free_reproduction"

    a, b = _flat(stored), _flat(fresh)
    for k in sorted(set(a) | set(b)):
        if k in VOLATILE or k.startswith(MEASURED_PREFIX):
            continue
        if a.get(k) != b.get(k):
            problems.append(f"{k}: report says {a.get(k)!r}, records give {b.get(k)!r}")

    # The embedded measured block must EQUAL the record it was copied from. Not re-derived -- compared.
    run_file = REPO / "provenance" / "REPRODUCTION_RUN.json"
    embedded = stored.get("cache_free_reproduction") or {}
    if run_file.is_file():
        canonical = json.loads(run_file.read_text())
        if embedded != canonical:
            ea, eb = _flat(embedded), _flat(canonical)
            diff = sorted(k for k in set(ea) | set(eb) if ea.get(k) != eb.get(k))
            problems.append(
                f"cache_free_reproduction has drifted from provenance/REPRODUCTION_RUN.json in "
                f"{len(diff)} field(s), e.g. {diff[:3]}. It is a verbatim copy, so it must be "
                f"regenerated whenever the record changes: python scripts/make_reproduction_report.py"
            )
    elif embedded.get("performed"):
        problems.append(
            "cache_free_reproduction reports a performed run, but provenance/REPRODUCTION_RUN.json -- "
            "the record it is copied from -- does not exist. The evidence has no source."
        )

    # The measured block is also checked for COHERENCE -- INDEPENDENTLY, not by trusting its summary
    # flags. Every condition below was previously either unchecked or checked only via the field that
    # asserts it, which means a record could report success over a gap: three panels instead of four,
    # five stages instead of six, a non-zero exit beside a clean [ran] count.
    run = embedded

    # VALIDATION RUNS UNCONDITIONALLY. It used to sit in the `else` of the display branch below, so a
    # record whose `performed` flag was absent or falsy skipped every check -- including the check that
    # `performed` must be exactly True, which therefore could not fire in the one case it exists for.
    # The helper's own mutation tests passed because they call it directly; only the wiring was wrong,
    # which is the kind of defect a helper-level test cannot see.
    problems.extend(_check_measured_evidence(run, stored))

    # The branch below decides only what is PRINTED. It must never decide whether anything is checked.
    if run.get("performed") is not True:
        print("  cache-free reproduction: NOT RECORDED for this tree")
        print(f"    {run.get('why_absent', '(no reason given)')}")
    else:
        res = run.get("result") or {}
        ev = (run.get("method") or {}).get("cache_free_evidence") or {}
        print(f"  cache-free reproduction: recorded at digest "
              f"{str(run.get('source_digest_at_run'))[:16]}…, "
              f"worst continuous {res.get('worst_continuous_delta')}, "
              f"worst rank {res.get('worst_rank_delta')}")
        print(f"    stages ran {ev.get('total_stages_ran')} / from cache "
              f"{ev.get('total_stages_from_cache')} across {len(ev.get('per_panel') or {})} panel(s); "
              f"every panel exit 0 and exact stage set: "
              f"{bool(ev.get('all_panels_exited_zero')) and bool(ev.get('all_panels_exact_stage_set'))}")

    n_panels = len(stored.get("published_records") or {})
    n_leaves = len(a)
    print(f"\n  {n_panels} panel(s), {n_leaves} field(s) compared against the published records")
    print(f"  digest in report : {(stored.get('source_identity') or {}).get('scientific_source_digest')}")
    print(f"  declared delta   : "
          f"{(stored.get('source_identity') or {}).get('declared_source_delta') or 'none'}")

    if problems:
        print(f"\n  {len(problems)} MISMATCH(ES) — the report has drifted from the records:")
        for p in problems[:20]:
            print(f"      - {p}")
        if len(problems) > 20:
            print(f"      ... and {len(problems) - 20} more")
        print("\n  Regenerate: python scripts/make_reproduction_report.py\n")
        return 1
    # Two different guarantees, and saying "every field re-derives" claimed the stronger one for both.
    # The derived summary IS re-derived from the published records. The measured block is not
    # re-derivable -- that is what makes it evidence -- so it is compared byte-for-byte against
    # REPRODUCTION_RUN.json and its conditions are recomputed from its own detail.
    print("\n  clean — derived fields re-derive from the published provenance; the measured "
          "reproduction block matches provenance/REPRODUCTION_RUN.json and satisfies every "
          "independently recomputed condition.\n")
    return 0


def _moduleless_rebuild() -> dict:
    """Rebuild the report from the records, tolerating the absence of a run tree."""
    return _maker().build()


if __name__ == "__main__":
    sys.exit(main())
