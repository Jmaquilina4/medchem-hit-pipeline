"""Adversarial tests for the reproduction recorder.

This tool's output is the strongest claim the repository makes: that the published numbers came back when
the panels were run again. A comparison tool that reports success wrongly is worse than none, because the
output looks like evidence. Every test here is a way the previous version could report success while
comparing nothing, or while comparing a file to itself.
"""

from __future__ import annotations

import importlib.util as iu
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = iu.spec_from_file_location("rrr", REPO / "scripts" / "record_reproduction_run.py")
assert _spec is not None and _spec.loader is not None
rrr = iu.module_from_spec(_spec)
sys.modules["rrr"] = rrr
_spec.loader.exec_module(rrr)


def _write(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
    return p


BASE = {"scaffold_cv": {"r2": 0.75}, "temporal_split": {"r2": -0.36},
        "direct_delta_scaffold_cv": {"A-B": {"pr_auc": 0.8, "support": {"supported": True}}}}


def test_same_path_is_rejected(tmp_path):
    """Comparing a file to itself is the failure this tool exists to avoid."""
    f = _write(tmp_path / "a.json", BASE)
    with pytest.raises(SystemExit, match="SAME path"):
        rrr.compare(f, f)


def test_same_inode_via_hard_link_is_rejected(tmp_path):
    """Two names for one file compare byte-identical and demonstrate nothing."""
    a = _write(tmp_path / "a.json", BASE)
    b = tmp_path / "b.json"
    try:
        os.link(a, b)
    except OSError:
        pytest.skip("hard links unavailable on this filesystem")
    with pytest.raises(SystemExit, match="SAME inode"):
        rrr.compare(a, b)


def test_genuinely_identical_bytes_in_distinct_files_is_allowed(tmp_path):
    """Byte-identical output from two REAL files is the good outcome, not an error."""
    a = _write(tmp_path / "ref" / "a.json", BASE)
    b = _write(tmp_path / "run" / "a.json", BASE)
    res = rrr.compare(a, b)
    assert res["reproduced"] and res["bytes_identical"] and res["distinct_files"]


def test_a_key_missing_from_the_rerun_is_a_finding(tmp_path):
    """Iterating the reference's keys alone cannot see a value the rerun dropped."""
    ref = _write(tmp_path / "ref" / "a.json", BASE)
    thin = json.loads(json.dumps(BASE))
    del thin["temporal_split"]
    run = _write(tmp_path / "run" / "a.json", thin)
    res = rrr.compare(ref, run)
    assert not res["reproduced"]
    assert any(f["kind"] == "missing_in_rerun" for f in res["findings"])


def test_an_extra_key_in_the_rerun_is_a_finding(tmp_path):
    """A value the rerun ADDED is invisible to a reference-keyed loop."""
    ref = _write(tmp_path / "ref" / "a.json", BASE)
    fat = json.loads(json.dumps(BASE))
    fat["surprise"] = 1.0
    run = _write(tmp_path / "run" / "a.json", fat)
    res = rrr.compare(ref, run)
    assert not res["reproduced"]
    assert any(f["kind"] == "extra_in_rerun" for f in res["findings"])


def test_non_finite_values_are_findings(tmp_path):
    """NaN compares unequal to everything including itself; Infinity passes a delta check trivially."""
    ref = _write(tmp_path / "ref" / "a.json", BASE)
    bad = json.loads(json.dumps(BASE))
    run = tmp_path / "run" / "a.json"
    run.parent.mkdir(parents=True, exist_ok=True)
    run.write_text(json.dumps(bad).replace('"r2": 0.75', '"r2": NaN'))
    res = rrr.compare(ref, run)
    assert not res["reproduced"]
    assert any(f["kind"] == "non_finite" for f in res["findings"])


def test_ignored_fields_are_an_exact_allowlist_not_substrings():
    """Substring matching excused result-bearing keys.

    ``"sha"`` matched every key containing it; ``"dir"`` matched ``direct_delta_scaffold_cv.*``, which is
    the selectivity result the comparison exists to check.
    """
    assert rrr._ignored(".run.cache_key")
    assert rrr._ignored(".x.artifact_dir")
    # the cases substring matching wrongly excused
    assert not rrr._ignored(".direct_delta_scaffold_cv.A-B.pr_auc")
    assert not rrr._ignored(".raw_inputs.JAK1.sha256")
    assert not rrr._ignored(".temporal_split.cutoff_year")


def test_a_rank_statistic_gets_the_looser_bound_and_a_continuous_one_does_not(tmp_path):
    ref = _write(tmp_path / "ref" / "a.json", BASE)
    near = json.loads(json.dumps(BASE))
    near["direct_delta_scaffold_cv"]["A-B"]["pr_auc"] = 0.8 + 1e-6      # inside 1e-5
    near["scaffold_cv"]["r2"] = 0.75 + 1e-6                              # outside 1e-12
    run = _write(tmp_path / "run" / "a.json", near)
    res = rrr.compare(ref, run)
    keys = {f["key"] for f in res["findings"]}
    assert ".scaffold_cv.r2" in keys, "a continuous metric must use the tight bound"
    assert ".direct_delta_scaffold_cv.A-B.pr_auc" not in keys, "a rank statistic uses the looser bound"


def test_recording_requires_an_explicit_reference_directory():
    """Defaulting to provenance/ made the about-to-be-overwritten publication dir the yardstick."""
    r = subprocess.run([sys.executable, "scripts/record_reproduction_run.py", "--runs", "/tmp"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0
    assert "--reference-dir" in (r.stderr + r.stdout)


def test_a_reference_inside_provenance_is_rejected(tmp_path):
    """publish_provenance overwrites provenance/, so a reference there is not immutable."""
    r = subprocess.run(
        [sys.executable, "scripts/record_reproduction_run.py",
         "--reference-dir", str(REPO / "provenance"), "--runs", str(tmp_path)],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0
    assert "overwrites" in (r.stderr + r.stdout)


def test_the_expected_artifact_count_is_pinned():
    """A comparison that silently examined three panels would report success over a gap."""
    assert rrr.EXPECTED_ARTIFACTS == len(rrr.PANELS) * len(rrr.COMPARED) == 8


def test_the_record_carries_prior_measurements_forward(tmp_path, monkeypatch):
    """The rank tolerance is documented as an OBSERVED bound, and the spread between runs is the evidence.

    Each invocation rewrites the record, so without a carry-forward the previous measurement disappears
    and the documentation ends up quoting a number no published record contains. That happened: the
    README cited 4.7e-07 from a run whose record had already been overwritten by a 1.6e-06 one.
    """
    out = tmp_path / "REPRODUCTION_RUN.json"
    monkeypatch.setattr(rrr, "OUT", out)
    out.write_text(json.dumps({
        "performed": True,
        "source_digest_at_run": "a" * 64,
        "label": "first run",
        "result": {"worst_rank_delta": 4.7e-07, "worst_continuous_delta": 2.4e-15},
    }))
    hist = rrr._measurement_history("b" * 64)
    assert len(hist) == 1 and hist[0]["source_digest_at_run"] == "a" * 64
    assert hist[0]["result"]["worst_rank_delta"] == 4.7e-07


def test_recording_the_same_digest_twice_does_not_inflate_the_history(tmp_path, monkeypatch):
    """Re-running at unchanged source adds no measurement, so it must add no history entry — otherwise
    the "spread across runs" evidence could be manufactured by running the same thing repeatedly."""
    out = tmp_path / "REPRODUCTION_RUN.json"
    monkeypatch.setattr(rrr, "OUT", out)
    out.write_text(json.dumps({
        "performed": True, "source_digest_at_run": "a" * 64, "label": "run",
        "result": {"worst_rank_delta": 1e-07},
        "previous_runs": [{"source_digest_at_run": "9" * 64, "label": "older",
                           "result": {"worst_rank_delta": 2e-07}}],
    }))
    assert [h["source_digest_at_run"] for h in rrr._measurement_history("a" * 64)] == ["9" * 64]


def test_an_absent_or_unperformed_record_yields_no_history(tmp_path, monkeypatch):
    out = tmp_path / "REPRODUCTION_RUN.json"
    monkeypatch.setattr(rrr, "OUT", out)
    assert rrr._measurement_history("a" * 64) == []
    out.write_text(json.dumps({"performed": False, "why_absent": "none recorded"}))
    assert rrr._measurement_history("a" * 64) == []


# ---------------------------------------------------------------------------------------------
# Exact-stage and exit-status evidence. Counting six [ran] lines proves six stages ran, not WHICH
# six: six could be five stages with one repeated, or six of which one is not a metric stage. And a
# zero [cache] count says nothing about whether the process then failed.
# ---------------------------------------------------------------------------------------------

STAGES = ("data_pull", "curate", "featurize", "qsar", "selectivity", "evaluate")


def _logs(tmp_path, *, stages=STAGES, cached=(), exits=None, drop_all_log=False, omit_panel=None):
    """A driver-shaped log directory: one <panel>.log each, plus all.log carrying the exit codes.

    ``omit_panel`` drops only that panel's LOG FILE and keeps its exit line, so the missing-log path is
    exercised rather than the missing-exit-status path, which is checked first.
    """
    d = tmp_path / "logs"
    d.mkdir(exist_ok=True)
    lines = []
    for panel in rrr.PANELS:
        lines.append(f"exit={(exits or {}).get(panel, 0)} panel={panel}")
        if panel == omit_panel:
            continue
        body = "".join(f"  [  ran] {s}  {{}}\n" for s in stages)
        body += "".join(f"  [cache] {s}  {{}}\n" for s in cached)
        (d / f"{panel}.log").write_text(body)
    if not drop_all_log:
        (d / "all.log").write_text("\n".join(lines) + "\nALL DONE\n")
    return d


def test_a_clean_run_is_accepted_and_records_the_exact_stage_set(tmp_path):
    ev = rrr._stage_evidence(_logs(tmp_path))
    assert ev["measured"] and ev["total_stages_from_cache"] == 0
    assert ev["total_stages_ran"] == len(rrr.PANELS) * len(STAGES) == 24
    assert ev["all_panels_exited_zero"] and ev["all_panels_exact_stage_set"]
    for panel, e in ev["per_panel"].items():
        assert e["stages_ran"] == sorted(STAGES), panel
        assert e["process_exit_code"] == 0 and e["exact_expected_stage_set"]
        assert len(e["log_sha256"]) == 64


def test_a_missing_stage_is_rejected_even_though_nothing_was_cached(tmp_path):
    """Five [ran] lines and no [cache] lines: the old count-only check saw 'no cache' and a wrong total,
    but a stage that never ran means the artifact came from somewhere this record does not describe."""
    with pytest.raises(SystemExit, match="missing \\['selectivity'\\]"):
        rrr._stage_evidence(_logs(tmp_path, stages=[s for s in STAGES if s != "selectivity"]))


def test_a_duplicated_stage_is_rejected(tmp_path):
    """Six [ran] lines, six expected — the count matches and the SET does not."""
    dupe = ("data_pull", "curate", "featurize", "qsar", "selectivity", "selectivity")
    with pytest.raises(SystemExit, match="duplicated \\['selectivity'\\]"):
        rrr._stage_evidence(_logs(tmp_path, stages=dupe))


def test_an_unexpected_stage_is_rejected(tmp_path):
    """Also six lines: `vls` instead of `evaluate` still counts to six, and evaluate produces the
    metrics this record is about."""
    wrong = ("data_pull", "curate", "featurize", "qsar", "selectivity", "vls")
    with pytest.raises(SystemExit, match="unexpected \\['vls'\\]"):
        rrr._stage_evidence(_logs(tmp_path, stages=wrong))


def test_a_single_cache_hit_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="resolved from cache"):
        rrr._stage_evidence(_logs(tmp_path, cached=("qsar",)))


def test_a_nonzero_process_exit_is_rejected(tmp_path):
    """A run can print six [ran] lines, write its artifacts, and then fail. Zero [cache] lines say
    nothing about that, which is why the exit code is required rather than inferred."""
    with pytest.raises(SystemExit, match="exit code is 1, not 0"):
        rrr._stage_evidence(_logs(tmp_path, exits={"brd4": 1}))


def test_an_unrecorded_exit_status_is_rejected_rather_than_assumed_zero(tmp_path):
    with pytest.raises(SystemExit, match="no all.log"):
        rrr._stage_evidence(_logs(tmp_path, drop_all_log=True))
    d = _logs(tmp_path)
    lines = [ln for ln in (d / "all.log").read_text().splitlines()
             if ln.strip() != "exit=0 panel=brd4"]
    (d / "all.log").write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="records no exit status for \\['brd4'\\]"):
        rrr._stage_evidence(d)


def test_a_missing_panel_log_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="is missing"):
        rrr._stage_evidence(_logs(tmp_path, omit_panel="jak1_sensitivity"))


def test_the_published_stage_evidence_names_no_path_or_user(tmp_path):
    """The logs live in a scratch directory whose name identifies a machine and a user, so the record
    carries stage names, counts, exit codes and hashes — never the paths."""
    ev = rrr._stage_evidence(_logs(tmp_path))
    blob = json.dumps(ev)
    assert "/Users/" not in blob and "/home/" not in blob and "/private/tmp" not in blob
    assert str(tmp_path) not in blob


def test_duplicate_panel_exit_records_are_rejected_not_resolved(tmp_path):
    """The LAST duplicate used to win, which reports a failed-then-rerun panel by its second attempt.

    This evidence claims "the panel ran once, cleanly". Which attempt produced the compared artifacts
    cannot be told from the log, so the honest response is to refuse rather than to pick the flattering
    line — and picking the last one is precisely picking the flattering one.
    """
    d = _logs(tmp_path)
    (d / "all.log").write_text((d / "all.log").read_text() + "exit=0 panel=brd4\n")
    with pytest.raises(SystemExit, match="more than one exit status"):
        rrr._stage_evidence(d)


def test_a_duplicate_recording_a_failure_is_also_rejected(tmp_path):
    """Order must not matter: a failure followed by a success is the case the old code hid."""
    d = _logs(tmp_path)
    lines = (d / "all.log").read_text().replace("exit=0 panel=jak1\n", "exit=1 panel=jak1\nexit=0 panel=jak1\n")
    (d / "all.log").write_text(lines)
    with pytest.raises(SystemExit, match="more than one exit status"):
        rrr._stage_evidence(d)


def test_a_historical_label_naming_a_non_digest_identifier_is_redacted(tmp_path, monkeypatch):
    """A carried-forward label cannot be retyped, so it is sanitised rather than trusted.

    This is not hypothetical: one prior label named a revision of the unpublished development workspace,
    and because the carry-forward copied labels verbatim it survived three regenerations inside
    ``previous_runs`` -- in a published record, past a rule written to remove exactly that.
    """
    out = tmp_path / "REPRODUCTION_RUN.json"
    monkeypatch.setattr(rrr, "OUT", out)
    out.write_text(json.dumps({
        "performed": True,
        "source_digest_at_run": "a" * 64,
        "label": "compared against the pre-fix deadbee reference",
        "result": {"worst_rank_delta": 1e-07},
    }))
    hist = rrr._measurement_history("b" * 64)
    assert len(hist) == 1
    label = hist[0]["label"]
    assert "deadbee" not in label
    # The identifier is DROPPED rather than marked: a placeholder announces that something was removed,
    # which is itself a disclosure and reads as machine output inside a sentence.
    assert label == "compared against the pre-fix reference", label
    assert "redacted" not in label.lower() and "<" not in label


def test_a_historical_label_naming_its_own_digest_is_preserved(tmp_path, monkeypatch):
    """Sanitising must not destroy the legitimate form: a reference named BY its digest."""
    out = tmp_path / "REPRODUCTION_RUN.json"
    monkeypatch.setattr(rrr, "OUT", out)
    digest = "c" * 64
    out.write_text(json.dumps({
        "performed": True, "source_digest_at_run": digest,
        "label": f"re-run cache-free at digest {digest[:10]}",
        "result": {"worst_rank_delta": 1e-07},
    }))
    hist = rrr._measurement_history("b" * 64)
    assert hist[0]["label"] == f"re-run cache-free at digest {digest[:10]}"
