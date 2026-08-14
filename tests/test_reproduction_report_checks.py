"""Mutation tests for the reproduction-report checker's independent conditions.

The measured block in ``provenance/REPRODUCTION.json`` is the strongest claim this repository makes, and
every condition below was at some point either unchecked or checked by reading the field that ASSERTS it.
A record can carry ``measured: true`` and ``all_panels_exited_zero: true`` while its per-panel detail says
three panels, five stages and a non-zero exit — so each condition is recomputed from the detail, and each
test here mutates a healthy record in exactly one way and requires a complaint.

The tests are mutations rather than fixtures on purpose: a hand-built "bad record" tends to be bad in ways
the checker already rejected for other reasons, which is how a condition ends up covered only by accident.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util as iu
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = iu.spec_from_file_location("crr", REPO / "scripts" / "check_reproduction_report.py")
assert _spec is not None and _spec.loader is not None
crr = iu.module_from_spec(_spec)
sys.modules["crr"] = crr
_spec.loader.exec_module(crr)

DIGEST = "d" * 64
PANELS = ("jak1", "jak1_sensitivity", "brd4", "brd4_sensitivity")
STAGES = ["curate", "data_pull", "evaluate", "featurize", "qsar", "selectivity"]
COMPARED = ("eval_report.json", "selectivity_metrics.json")


@pytest.fixture
def healthy(tmp_path, monkeypatch):
    """A record that passes every condition, plus the published files it claims on disk."""
    monkeypatch.setattr(crr, "REPO", tmp_path)
    prov = tmp_path / "provenance"
    panels: dict[str, dict] = {}
    for p in PANELS:
        (prov / p).mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for f in COMPARED:
            body = json.dumps({"panel": p, "artifact": f}).encode()
            (prov / p / f).write_bytes(body)
            h = hashlib.sha256(body).hexdigest()
            artifacts[f] = {"findings": [], "reproduced": True,
                            "published_sha256": h, "rerun_sha256": h, "reference_sha256": h}
        panels[p] = {"artifacts": artifacts, "decisions": {"identical": True}}

    run = {
        "performed": True,
        "label": f"reference at digest {DIGEST[:8]}",
        "source_digest_at_run": DIGEST,
        "previous_runs": [],
        "result": {"all_within_tolerance": True, "artifacts_compared": 8,
                   "worst_continuous_delta": 1e-15, "worst_rank_delta": 1e-6},
        "method": {"cache_free_evidence": {
            "measured": True,
            "per_panel": {p: {"stages_ran": list(STAGES), "stages_from_cache": [],
                              "process_exit_code": 0, "log_sha256": "a" * 64} for p in PANELS},
            "total_stages_ran": 24, "total_stages_from_cache": 0,
            "all_panels_exited_zero": True, "all_panels_exact_stage_set": True,
        }},
        "panels": panels,
        "published_verified": True,
    }
    stored = {"source_identity": {"scientific_source_digest": DIGEST}}
    return run, stored


def _problems(healthy, mutate=None) -> list[str]:
    run, stored = copy.deepcopy(healthy[0]), copy.deepcopy(healthy[1])
    if mutate:
        mutate(run, stored)
    return crr._check_measured_evidence(run, stored)


def test_a_healthy_record_produces_no_complaints(healthy):
    assert _problems(healthy) == []


def test_measured_false_is_rejected(healthy):
    out = _problems(healthy, lambda r, s: r["method"]["cache_free_evidence"].update(measured=False))
    assert any("does not claim measured stage evidence" in p for p in out)


def test_three_panels_instead_of_four_is_rejected(healthy):
    """A record covering three panels reports success over a gap."""
    def m(r, s):
        r["method"]["cache_free_evidence"]["per_panel"].pop("brd4")
    out = _problems(healthy, m)
    assert any("covers panels" in p for p in out)


def test_a_missing_stage_is_rejected(healthy):
    def m(r, s):
        r["method"]["cache_free_evidence"]["per_panel"]["jak1"]["stages_ran"] = STAGES[:-1]
    out = _problems(healthy, m)
    assert any("expected exactly" in p for p in out)


def test_a_duplicated_stage_is_rejected(healthy):
    """Six entries, six expected -- the COUNT matches and the set does not."""
    def m(r, s):
        r["method"]["cache_free_evidence"]["per_panel"]["jak1"]["stages_ran"] = STAGES[:-1] + ["qsar"]
    out = _problems(healthy, m)
    assert any("expected exactly" in p for p in out)


def test_an_unexpected_stage_is_rejected(healthy):
    def m(r, s):
        r["method"]["cache_free_evidence"]["per_panel"]["jak1"]["stages_ran"] = STAGES[:-1] + ["vls"]
    out = _problems(healthy, m)
    assert any("expected exactly" in p for p in out)


def test_a_cache_hit_is_rejected(healthy):
    def m(r, s):
        r["method"]["cache_free_evidence"]["per_panel"]["brd4"]["stages_from_cache"] = ["qsar"]
    out = _problems(healthy, m)
    assert any("resolved from cache" in p for p in out)


def test_a_nonzero_exit_is_rejected_even_with_a_clean_stage_set(healthy):
    """The summary flag still says every panel exited zero; the detail is what is believed."""
    def m(r, s):
        r["method"]["cache_free_evidence"]["per_panel"]["brd4"]["process_exit_code"] = 1
    out = _problems(healthy, m)
    assert any("exit code 1" in p for p in out)


def test_a_summary_flag_cannot_stand_in_for_the_detail(healthy):
    """all_panels_exited_zero stays true while a panel's own record shows a failure."""
    def m(r, s):
        ev = r["method"]["cache_free_evidence"]
        ev["per_panel"]["jak1"]["process_exit_code"] = 2
        ev["all_panels_exited_zero"] = True
        ev["all_panels_exact_stage_set"] = True
    out = _problems(healthy, m)
    assert any("exit code 2" in p for p in out)


@pytest.mark.parametrize("bad_hash", ["", "not-a-hash", "abc123", "A" * 64, "f" * 63])
def test_an_invalid_log_hash_is_rejected(healthy, bad_hash: str):
    def m(r, s):
        r["method"]["cache_free_evidence"]["per_panel"]["jak1"]["log_sha256"] = bad_hash
    out = _problems(healthy, m)
    assert any("not a 64-hex digest" in p for p in out)


def test_seven_artifacts_instead_of_eight_is_rejected(healthy):
    def m(r, s):
        r["panels"]["brd4"]["artifacts"].pop("selectivity_metrics.json")
    out = _problems(healthy, m)
    assert any("expected exactly 8" in p for p in out)


def test_a_recorded_finding_is_rejected(healthy):
    def m(r, s):
        r["panels"]["jak1"]["artifacts"]["eval_report.json"]["findings"] = [{"key": ".x", "kind": "value"}]
    out = _problems(healthy, m)
    assert any("comparison finding" in p for p in out)


def test_non_identical_decisions_are_rejected(healthy):
    def m(r, s):
        r["panels"]["brd4"]["decisions"]["identical"] = False
    out = _problems(healthy, m)
    assert any("decisions are not identical" in p for p in out)


def test_a_published_file_that_no_longer_matches_its_recorded_hash_is_rejected(healthy, tmp_path):
    """The failure that makes every other check vacuous: the record describes a file that has changed."""
    def m(r, s):
        (tmp_path / "provenance" / "jak1" / "eval_report.json").write_bytes(b'{"edited": true}')
    out = _problems(healthy, m)
    assert any("published file hashes to" in p for p in out)
    assert any("published_sha256" in p or "rerun_sha256" in p for p in out)


def test_a_published_file_absent_from_provenance_is_rejected(healthy, tmp_path):
    def m(r, s):
        (tmp_path / "provenance" / "jak1" / "selectivity_metrics.json").unlink()
    out = _problems(healthy, m)
    assert any("absent from provenance" in p for p in out)


def test_a_digest_mismatch_against_the_tree_is_rejected(healthy):
    def m(r, s):
        s["source_identity"]["scientific_source_digest"] = "e" * 64
    out = _problems(healthy, m)
    assert any("describes different source" in p for p in out)


def test_out_of_tolerance_is_rejected(healthy):
    def m(r, s):
        r["result"]["all_within_tolerance"] = False
    out = _problems(healthy, m)
    assert any("out of tolerance" in p for p in out)


def test_a_label_naming_an_undeclared_revision_is_rejected(healthy):
    """The residue this rule exists for: a published record naming a private workspace revision."""
    def m(r, s):
        # A synthetic hex, not a real development revision: a test that hard-codes the identifier it
        # exists to reject publishes that identifier in the shipped test suite.
        r["label"] = "compared against the pre-fix deadbee reference"
    out = _problems(healthy, m)
    assert any("hexadecimal identifier" in p for p in out)


def test_a_label_naming_a_declared_digest_prefix_is_accepted(healthy):
    """The rule must not simply ban hex: identifying a reference BY ITS DIGEST is the correct form."""
    def m(r, s):
        r["label"] = f"compared against the immutable reference at scientific digest {DIGEST[:12]}"
    assert _problems(healthy, m) == []


def test_a_label_naming_a_prior_runs_digest_is_accepted(healthy):
    def m(r, s):
        r["previous_runs"] = [{"source_digest_at_run": "b" * 64, "result": {}}]
        r["label"] = f"successor to the run at {'b' * 10}"
    assert _problems(healthy, m) == []


# --- the final gaps: a truthy `performed`, a historical label, and summaries that disagree ------------

@pytest.mark.parametrize("value", ["yes", 1, "true", [1], {"a": 1}])
def test_a_truthy_but_non_true_performed_flag_is_rejected(healthy, value):
    """`performed` gates every check below it, and was read with a truthiness test.

    A hand-edited or half-written record carrying `"performed": "yes"` therefore passed the gate and was
    treated as a performed run. A flag that decides whether any evidence is examined deserves an identity
    comparison.
    """
    out = _problems(healthy, lambda r, s: r.update(performed=value))
    assert any("not exactly true" in p for p in out)


def test_an_absent_performed_flag_is_rejected(healthy):
    out = _problems(healthy, lambda r, s: r.pop("performed"))
    assert any("not exactly true" in p for p in out)


def test_a_historical_label_naming_an_undeclared_identifier_is_rejected(healthy):
    """Where the real identifier actually reached publication.

    The label rule originally checked only the CURRENT label. The development-workspace revision that
    survived into published records did so inside `previous_runs`, promoted there by an earlier
    regeneration, which the check did not look at.
    """
    def m(r, s):
        r["previous_runs"] = [{"source_digest_at_run": "b" * 64,
                               "label": "compared against the pre-fix deadbee reference",
                               "result": {}}]
    out = _problems(healthy, m)
    assert any("previous_runs[0].label" in p and "hexadecimal identifier" in p for p in out)


def test_a_historical_label_naming_its_own_declared_digest_is_accepted(healthy):
    def m(r, s):
        r["previous_runs"] = [{"source_digest_at_run": "b" * 64,
                               "label": f"re-run at digest {'b' * 12}", "result": {}}]
    assert _problems(healthy, m) == []


@pytest.mark.parametrize(("field", "wrong"), [("total_stages_ran", 23), ("total_stages_from_cache", 1)])
def test_a_summary_total_that_disagrees_with_the_detail_is_rejected(healthy, field, wrong):
    """The detail is authoritative, and the summary is what a reader skims -- so they must agree."""
    out = _problems(healthy, lambda r, s: r["method"]["cache_free_evidence"].update({field: wrong}))
    assert any(f"cache_free_evidence.{field}" in p and "disagree" in p for p in out)


@pytest.mark.parametrize("field", ["all_panels_exited_zero", "all_panels_exact_stage_set"])
@pytest.mark.parametrize("value", [False, "true", None, 1])
def test_a_summary_flag_that_is_not_exactly_true_is_rejected(healthy, field, value):
    out = _problems(healthy, lambda r, s: r["method"]["cache_free_evidence"].update({field: value}))
    assert any(field in p and "not exactly true" in p for p in out)


@pytest.mark.parametrize("value", [7, 9, None, "8"])
def test_a_wrong_artifacts_compared_summary_is_rejected(healthy, value):
    out = _problems(healthy, lambda r, s: r["result"].update(artifacts_compared=value))
    assert any("artifacts_compared" in p for p in out)


@pytest.mark.parametrize("value", [False, None, "yes", 1])
def test_published_verified_must_be_exactly_true(healthy, value):
    """Set by `--verify-published`. Absent or truthy means that step was never recorded as passing --
    while the on-disk hash recomputation above stays the authoritative check."""
    out = _problems(healthy, lambda r, s: r.update(published_verified=value))
    assert any("published_verified" in p for p in out)


# ---------------------------------------------------------------------------------------------
# MAIN-PATH tests. The helper mutation tests above call _check_measured_evidence directly, so they
# all passed while the wiring was wrong: the call sat inside the `else` of the display branch, so a
# record with `performed` absent or false skipped EVERY check -- including the check that `performed`
# must be exactly True, which could therefore never fire in the one case it exists for.
#
# A helper-level test cannot see that. These invoke main() and assert on its exit code.
# ---------------------------------------------------------------------------------------------

def _staged_tree(tmp_path, monkeypatch, mutate):
    """A provenance tree main() can read, with the embedded and canonical records mutated identically.

    `_moduleless_rebuild` is stubbed to echo the stored report so the DERIVED comparison passes trivially
    and the test isolates the measured-block wiring. Everything else is the real main().
    """
    prov = tmp_path / "provenance"
    prov.mkdir(parents=True, exist_ok=True)

    panels: dict[str, dict] = {}
    for p in PANELS:
        (prov / p).mkdir(exist_ok=True)
        artifacts = {}
        for f in COMPARED:
            body = json.dumps({"panel": p, "artifact": f}).encode()
            (prov / p / f).write_bytes(body)
            h = hashlib.sha256(body).hexdigest()
            artifacts[f] = {"findings": [], "reproduced": True,
                            "published_sha256": h, "rerun_sha256": h, "reference_sha256": h}
        panels[p] = {"artifacts": artifacts, "decisions": {"identical": True}}

    run = {
        "performed": True,
        "label": f"reference at digest {DIGEST[:8]}",
        "source_digest_at_run": DIGEST,
        "previous_runs": [],
        "result": {"all_within_tolerance": True, "artifacts_compared": 8,
                   "worst_continuous_delta": 1e-15, "worst_rank_delta": 1e-6},
        "method": {"cache_free_evidence": {
            "measured": True,
            "per_panel": {p: {"stages_ran": list(STAGES), "stages_from_cache": [],
                              "process_exit_code": 0, "log_sha256": "a" * 64} for p in PANELS},
            "total_stages_ran": 24, "total_stages_from_cache": 0,
            "all_panels_exited_zero": True, "all_panels_exact_stage_set": True,
        }},
        "panels": panels,
        "published_verified": True,
    }
    mutate(run)                                    # applied to BOTH copies, so no drift is introduced
    report = {"source_identity": {"scientific_source_digest": DIGEST},
              "published_records": {p: {} for p in PANELS},
              "cache_free_reproduction": run}

    (prov / "REPRODUCTION_RUN.json").write_text(json.dumps(run, indent=2))
    (prov / "REPRODUCTION.json").write_text(json.dumps(report, indent=2))
    monkeypatch.setattr(crr, "REPO", tmp_path)
    monkeypatch.setattr(crr, "REPORT", prov / "REPRODUCTION.json")
    monkeypatch.setattr(crr, "_moduleless_rebuild", lambda: json.loads(
        (prov / "REPRODUCTION.json").read_text()))
    return prov


def test_main_returns_zero_on_a_healthy_staged_tree(tmp_path, monkeypatch, capsys):
    """The control: without this, a test that always returns 1 would look like it was working."""
    _staged_tree(tmp_path, monkeypatch, lambda run: None)
    assert crr.main() == 0
    assert "clean" in capsys.readouterr().out


def test_main_returns_one_when_performed_is_absent(tmp_path, monkeypatch, capsys):
    """The wiring bug, at the level that exhibits it.

    With the call inside the display branch, an absent `performed` printed NOT RECORDED and returned 0 --
    reporting success over a record that established nothing.
    """
    _staged_tree(tmp_path, monkeypatch, lambda run: run.pop("performed"))
    assert crr.main() == 1, "an absent `performed` flag must fail, not print NOT RECORDED and pass"
    out = capsys.readouterr().out
    assert "NOT RECORDED" in out, "the display branch should still report the absence"
    assert "not exactly true" in out, "and the validation must have run and complained"


def test_main_returns_one_when_performed_is_false(tmp_path, monkeypatch, capsys):
    _staged_tree(tmp_path, monkeypatch, lambda run: run.update(performed=False))
    assert crr.main() == 1
    out = capsys.readouterr().out
    assert "NOT RECORDED" in out and "not exactly true" in out


@pytest.mark.parametrize("value", ["yes", 1, "true"])
def test_main_returns_one_when_performed_is_merely_truthy(tmp_path, monkeypatch, value):
    """Truthy is not true: these took the `else` branch and were validated as performed runs."""
    _staged_tree(tmp_path, monkeypatch, lambda run: run.update(performed=value))
    assert crr.main() == 1


def test_main_still_validates_the_rest_when_performed_is_absent(tmp_path, monkeypatch, capsys):
    """Unconditional means unconditional: a second, unrelated defect must also surface.

    This is what distinguishes "the flag check now fires" from "validation now runs at all".
    """
    def m(run):
        run.pop("performed")
        run["method"]["cache_free_evidence"]["per_panel"]["brd4"]["process_exit_code"] = 1
    _staged_tree(tmp_path, monkeypatch, m)
    assert crr.main() == 1
    out = capsys.readouterr().out
    assert "exit code 1" in out, "a non-flag defect must be reported even with `performed` absent"


def test_the_validation_call_is_not_inside_the_display_branch():
    """Structural guard, so the call cannot drift back under the branch.

    Asserted on the AST: the `if run.get("performed") ...` statement must not contain the
    _check_measured_evidence call anywhere in its body or its else clause.
    """
    import ast

    src = (REPO / "scripts" / "check_reproduction_report.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    def calls_checker(node) -> bool:
        return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_check_measured_evidence"
                   for n in ast.walk(node))

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "performed" in (ast.get_source_segment(src, node.test) or ""):
            for branch in (node.body, node.orelse):
                for stmt in branch:
                    assert not calls_checker(stmt), (
                        "_check_measured_evidence is inside the `performed` display branch again; "
                        "validation must not depend on the flag it validates"
                    )
    assert calls_checker(tree), "the checker is not called at all"
