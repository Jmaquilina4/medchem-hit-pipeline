"""CI and the local gate must be ONE executable path, and that path must run in the export.

What changed, and why the previous test could not have caught it
---------------------------------------------------------------
CI used to enumerate each check in the workflow while ``scripts/gate.sh`` enumerated them again, and this
file compared the two lists as TEXT. That test could confirm the strings matched. It could not confirm the
gate would RUN — and it did not, in the tree that matters: a gate step imported ``scripts/export_public.py``
to discover the shipped-script list, and the sanitized export deliberately does not ship that file. The
strings matched perfectly while ``bash scripts/gate.sh`` was broken in the published candidate.

So the workflow now has one step, ``bash scripts/gate.sh --ci``, and the checks below assert properties of
that arrangement instead of comparing two lists:

* the workflow really does delegate to the gate, and does not re-enumerate checks;
* every path the gate references is one the export actually ships;
* ``--ci`` differs from the default only in the leak-scan mode;
* and, as an integration test, the exported gate executes end to end in a sanitized fresh clone.

The integration test is opt-in via ``-m integration`` because it builds an export, clones it, installs a
locked environment and runs the whole suite inside itself — minutes, not milliseconds. It is the check
that would have caught the defect, so it exists; making it part of every unit run would make the unit run
useless.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CI = REPO / ".github" / "workflows" / "ci.yml"
GATE = REPO / "scripts" / "gate.sh"


def _ci_run_commands() -> list[str]:
    text = CI.read_text(encoding="utf-8")
    return [" ".join(c.strip().strip("|>").strip().split())
            for c in re.findall(r"^\s*run:\s*(.+)$", text, flags=re.MULTILINE) if c.strip()]


def test_ci_delegates_to_the_gate_and_does_not_reenumerate_checks():
    """One executable path. A second list of commands in the workflow is a second thing to drift."""
    cmds = _ci_run_commands()
    assert cmds == ["bash scripts/gate.sh --ci"], (
        f"CI must run exactly the gate; found {cmds}. Enumerating checks in the workflow recreates the "
        f"drift this arrangement removes."
    )


def test_gate_supports_the_ci_mode_ci_invokes():
    gate = GATE.read_text(encoding="utf-8")
    assert "--ci" in gate and "CI_MODE" in gate


def test_ci_mode_differs_only_in_the_leak_scan():
    """The two modes must not diverge in what they CHECK, only in what a runner can be asked about."""
    gate = GATE.read_text(encoding="utf-8")
    start = gate.index('if [ "$CI_MODE" -eq 1 ]')
    guarded = gate[start:gate.index("\nfi\n", start)]
    # Everything inside the mode split is leak-scan invocation and its narration.
    for line in guarded.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(("if", "else", "fi", "step", "printf")):
            continue
        assert "check_no_leaks.py" in s, (
            f"the CI/developer split must only change the leak scan; found: {s}"
        )


def test_everything_the_gate_needs_is_tracked_in_this_tree():
    """A file the export copies must be tracked here, or the export ships something git does not have.

    ``export_public.py`` copies from the FILESYSTEM, so an untracked new script would be copied into the
    candidate and then be absent from the development repository's own history. Checking tracked-ness
    here is what stops "works on my machine" from reaching a published tree.
    """
    if not (REPO / "scripts" / "export_public.py").is_file():
        pytest.skip("no exporter here; this tree IS the export")
    tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
                  .stdout.split())
    gate = GATE.read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"(scripts/[A-Za-z0-9_.\-]+\.(?:py|sh))", gate)))
    untracked = [r for r in referenced if (REPO / r).is_file() and r not in tracked]
    assert not untracked, f"the gate references untracked file(s): {untracked}. git add them."
    # The report the gate verifies must be tracked too, or a fresh clone has nothing to check.
    assert "provenance/REPRODUCTION.json" in tracked, (
        "provenance/REPRODUCTION.json is not tracked, so the exported gate would have nothing to verify"
    )


def test_exported_tree_actually_contains_every_file_the_gate_runs(tmp_path):
    """Build a real export and inspect what it TRACKS -- not what the allowlist declares.

    The declared allowlist and the built tree can disagree: a denylist can carve a file out, a rename can
    move it, and a new script can be added to the gate and forgotten in the allowlist. Only the built
    artifact settles it, so this builds one. Fast: no environment install, no gate run.
    """
    if not (REPO / "scripts" / "export_public.py").is_file():
        pytest.skip("no exporter here; this tree IS the export")

    export = tmp_path / "export"
    r = subprocess.run(
        [sys.executable, "scripts/export_public.py", "--out", str(export), "--commit"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"export failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"

    tracked = set(subprocess.run(["git", "ls-files"], cwd=export, capture_output=True, text=True)
                  .stdout.split())
    gate = (export / "scripts" / "gate.sh").read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"(scripts/[A-Za-z0-9_.\-]+\.(?:py|sh))", gate)))

    # Every referenced script must be tracked in the export, EXCEPT ones the gate guards with an
    # existence check -- the shipped-script lint reads the exporter's allowlist when present and no-ops
    # when it is not.
    guarded = {"scripts/export_public.py"}
    missing = [r for r in referenced if r not in tracked and r not in guarded]
    assert not missing, (
        f"the exported gate references {missing}, which the built export does not track. "
        f"`bash scripts/gate.sh` cannot run in the candidate."
    )
    for g in guarded:
        assert g not in tracked, f"{g} is shipped after all; remove it from the guarded set"
    assert "provenance/REPRODUCTION.json" in tracked


@pytest.mark.integration
def test_exported_gate_runs_end_to_end_in_a_sanitized_fresh_clone(tmp_path):
    """Build an export, clone it ``--no-local``, and run the gate a reader would run.

    MANDATORY in the release path: scripts/gate.sh runs this itself when the exporter is present, so a
    developer or CI run of the gate exercises it. It carries the `integration` marker so the ordinary
    unit sweep does not pay for it twice -- the gate invokes it explicitly with `-m integration`.

    NO RECURSION: the sanitized export deliberately does not ship ``export_public.py``, so the inner
    gate's copy of this test skips at the first line and the nesting terminates one level deep. The
    guard is that absence, which is a property of the allowlist rather than a flag someone must remember
    to set.
    """
    if not (REPO / "scripts" / "export_public.py").is_file():
        pytest.skip("no exporter here; this tree IS the export -- recursion stops here")
    if shutil.which("uv") is None:
        pytest.skip("uv is required to run the exported gate")

    export = tmp_path / "export"
    r = subprocess.run(
        [sys.executable, "scripts/export_public.py", "--out", str(export), "--commit"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"export failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"
    assert not (export / "scripts" / "export_public.py").exists(), (
        "the export ships the exporter, so the inner gate would recurse"
    )

    clone = tmp_path / "clone"
    assert subprocess.run(["git", "clone", "--no-local", "-q", str(export), str(clone)],
                          capture_output=True).returncode == 0

    r = subprocess.run(["bash", "scripts/gate.sh", "--ci"], cwd=clone,
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "the exported gate failed in a sanitized fresh clone:\n"
        f"{r.stdout[-6000:]}\n{r.stderr[-3000:]}"
    )
    assert "All gates passed." in r.stdout
