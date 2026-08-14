"""Regression tests for the publication-hygiene scanner.

Both cases below are failures this scanner actually had. It reported the tree CLEAN in each, which is
worse than reporting nothing, because a clean result gets treated as evidence.
"""

from __future__ import annotations

import importlib.util as iu
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = iu.spec_from_file_location("cnl", REPO / "scripts" / "check_no_leaks.py")
assert _spec is not None and _spec.loader is not None, "cannot load scripts/check_no_leaks.py"
cnl = iu.module_from_spec(_spec)
_spec.loader.exec_module(cnl)


def _compiled():
    return [(n, re.compile(rx), sev) for n, rx, sev, _ in cnl.PATTERNS + cnl._load_private_patterns()]


def _blocks(text: str) -> bool:
    return any(rx.search(text) and sev == "block" for _, rx, sev in _compiled())


def test_scanner_does_not_exempt_its_own_file_wholesale():
    """A blanket self-exemption once hid 13 real matches inside this scanner's own source."""
    assert not any(a.search("scripts/check_no_leaks.py") for a in cnl.ALLOWLIST_PATHS), (
        "the scanner must scan itself; only its pattern-definition block may be exempt"
    )


def test_pattern_definitions_do_not_spell_their_own_triggers():
    """Patterns are written with character classes so this file cannot trip its own audience rule."""
    # The exemption is now per-file: two scripts legitimately contain detection patterns, and each
    # names the exact region to skip. Asserted for BOTH, so adding a third scanner without its
    # sentinels fails here rather than silently exempting nothing.
    assert set(cnl._PATTERN_DEFINING_FILES) == {"check_no_leaks.py", "check_release_ready.py"}
    for fname, (start_s, end_s) in cnl._PATTERN_DEFINING_FILES.items():
        src = (REPO / "scripts" / fname).read_text()
        assert start_s in src and end_s in src, f"{fname}: sentinels no longer match its source"
        start, end = src.index(start_s), src.index(end_s)
        assert start < end, f"{fname}: sentinels are out of order, so the region is empty"
        outside = src[:start] + src[end:]
        for phrase in ("target " + "audience", "hiring " + "manager"):  # leak-scan: allow
            assert phrase not in outside, f"{phrase!r} appears outside {fname}'s exempt block"


def test_pattern_definition_exemption_is_one_bounded_region():
    """The self-exemption must cover the pattern block and nothing else.

    It covered almost everything. Both scanners located the region with a running toggle::

        if start_sentinel in line:  inside = True
        elif end_sentinel in line:  inside = False

    and each contains exactly one line holding BOTH sentinels -- the ``_PATTERN_DEFINING_FILES``
    declaration, which cannot avoid quoting them. That line takes the first branch, never the second,
    so the region reopened mid-file and ran to EOF: 377 of 476 lines of one scanner and 212 of 282 of
    the other were exempt from their own scans, including every identity check and both ``main()``
    bodies. A scanner that exempts its own file cannot see its own leaks -- the same defect this
    project documented for one scanner and then reproduced in the other.
    """
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("crr", REPO / "scripts" / "check_release_ready.py")
    assert spec is not None and spec.loader is not None
    crr = _iu.module_from_spec(spec)
    spec.loader.exec_module(crr)

    for scanner in (cnl, crr):
        for fname, (start_s, end_s) in scanner._PATTERN_DEFINING_FILES.items():
            lines = (REPO / "scripts" / fname).read_text(encoding="utf-8").splitlines()
            span = scanner._exempt_span(lines, start_s, end_s)
            assert span is not None, f"{fname}: no bounded region found"
            start, end = span
            assert start < end
            # The declaration line quoting both sentinels must NOT be treated as a region start.
            both = [i for i, ln in enumerate(lines) if start_s in ln and end_s in ln]
            assert both, "expected the declaration line that quotes both sentinels to still exist"
            for i in both:
                assert not (start <= i < end), (
                    f"{fname}: line {i + 1} quotes both sentinels and is inside the exempt region"
                )
            # Anything after the region is scanned -- most of the file.
            assert (len(lines) - end) > 0.4 * len(lines), (
                f"{fname}: only {len(lines) - end} of {len(lines)} lines remain scanned after the "
                f"exempt region; the exemption is too wide"
            )


def test_missing_end_sentinel_scans_the_whole_file():
    """Fail closed: a missing end sentinel must not blank a file to EOF."""
    lines = ["a", "START", "secret", "b"]
    assert cnl._exempt_span(lines, "START", "NO-SUCH-END") is None


def test_overlay_is_data_not_code():
    """The untracked overlay must be inert. A security check should not execute a hand-edited file."""
    assert not (REPO / "scripts" / "leak_patterns_local.py").exists(), (
        "the executable overlay must not exist; use the TOML form"
    )


def test_dummy_overlay_loads_and_matches_a_bare_token(tmp_path):
    """Runs in PUBLIC CI with a synthetic token, so the behaviour is never untested.

    The regression this guards: an organisation pattern written only in compound form missed the bare
    word followed by an ordinary noun, and the scan reported CLEAN. Using a dummy token means CI
    exercises the full load-and-match path without the real term appearing in a committed test.
    """
    overlay = tmp_path / "leak_patterns_local.toml"
    overlay.write_text(
        "[[pattern]]\n"
        'name = "dummy-org"\n'
        "regex = '\\bzorptech\\b|zorptech[\\s_-]?labs'\n"
        'severity = "block"\n'
        'why = "synthetic organisation name"\n'
    )
    loaded = cnl._load_private_patterns(overlay)
    assert len(loaded) == 1, "dummy overlay should load exactly one pattern"
    name, rx, sev, _ = loaded[0]
    assert sev == "block"
    # the compound form AND the bare token in context must both match
    assert re.search(rx, "zorptech labs owns this", re.IGNORECASE)
    assert re.search(rx, "do not run this on zorptech compute", re.IGNORECASE), (
        "bare token followed by an ordinary noun must match -- this is the case that slipped"
    )


def test_overlay_fails_closed_when_present_but_invalid(tmp_path):
    """A security check must not warn-and-continue on a broken rule set."""
    bad = tmp_path / "leak_patterns_local.toml"
    bad.write_text("not valid toml [[[")
    try:
        cnl._load_private_patterns(bad)
    except cnl.OverlayError:
        pass
    else:
        raise AssertionError("malformed overlay must raise OverlayError, not be silently dropped")

    incomplete = tmp_path / "incomplete.toml"
    incomplete.write_text('[[pattern]]\nname = "x"\n')
    try:
        cnl._load_private_patterns(incomplete)
    except cnl.OverlayError:
        pass
    else:
        raise AssertionError("overlay missing required fields must raise OverlayError")


def test_absent_overlay_is_legitimate(tmp_path):
    """A fresh clone has no overlay and must still run the generic checks."""
    assert cnl._load_private_patterns(tmp_path / "nope.toml") == []


def test_real_overlay_admits_bare_tokens_if_present():
    """A pattern requiring the compound form once missed the bare word followed by an ordinary noun.

    This cannot hardcode the token -- writing it here would publish it. Instead it derives a probe
    from each overlay pattern's own alternatives and asserts the pattern matches its shortest
    word-boundary alternative standing alone, which is exactly the case that previously slipped.
    """
    overlay = cnl._load_private_patterns()
    if not overlay:
        return  # private-only check; the synthetic-token test above covers CI
    bare = [
        (name, rx)
        for name, rx, sev, _ in overlay
        if sev == "block" and r"\b" in rx
    ]
    assert bare, "at least one blocking overlay pattern should use a word-boundary bare form"
    for name, rx in bare:
        for alt in rx.replace("(?i)", "").split("|"):
            token = alt.replace(r"\b", "").strip()
            if token.isalpha() and len(token) > 3:
                probe = f"do not run this on {token} compute"
                assert re.search(rx, probe, re.IGNORECASE), (
                    f"overlay pattern {name!r} does not match its own bare token in context"
                )
                break


def test_credential_and_path_patterns_still_fire():
    """The published set is SHAPE-based, not issuer-based: a scanner that spells out issuer-specific
    prefixes publishes them, which is content this project must not publish. So this asserts the shapes,
    and asserts that the hash-shaped things this repository legitimately contains do NOT fire."""
    assert _blocks("/Users/" + "someone/workspace/thing")            # leak-scan: allow
    assert _blocks("-----BEGIN RSA PRIVATE " + "KEY-----")
    assert _blocks('api_key = "' + "a1b2c3d4e5f6g7h8i9j0k1l2" + '"')
    assert _blocks("xyz_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5")          # prefixed opaque token
    assert _blocks("/private/tmp/" + "somewhere-1786479783940" + "/f")  # machine-local scratch

    # Must NOT fire on things this repository is legitimately full of.
    assert not _blocks("a" * 64)                                       # a bare SHA-256 digest
    assert not _blocks("https://files.example.org/packages/aa/bb/" + "c" * 48 + "/pkg.whl")
    # Prose: a vendor name and a chemical name run together in a ChEMBL assay description. This exact
    # shape tripped the token pattern inside the PUBLISHED snapshots until it required a digit.
    assert not _blocks("Sigma-" + "AldrichEthylenediaminetetraacetic" + " acid")


def test_git_identity_check_exists():
    assert callable(cnl._check_git_identity)


# --- developer mode vs --ci --------------------------------------------------------------------------
#
# The scan failed in every clean GitHub Actions checkout, because it required a repository-LOCAL
# noreply git identity and a runner has none. That finding protects a developer's NEXT commit from
# inheriting a global address; CI creates no commits, so on a runner it is unreachable rather than
# merely unlikely. `--ci` narrows exactly that one branch.
#
# The risk in adding a mode to a security check is that the mode quietly disables more than intended,
# so each case below pins one edge of the boundary. The two history cases matter most: they are what
# actually protects published history, and they must stay blocking on a runner.

# Built by concatenation so no line of this file is email-SHAPED.
#
# scripts/check_release_ready.py scans every blob in history with its own pattern set and, unlike
# check_no_leaks.py, honours no inline exemption marker -- deliberately: it is the last gate before
# publication and an exemption mechanism there is a way to wave something through. So it flagged
# these fixtures and failed the release check on the first export it ever examined. Keeping that
# gate exemption-free and writing the fixtures in parts is the better trade; the split-literal
# idiom is already used further up this file for the same reason.
NOREPLY = "12345+someone@" + "users.noreply.github.com"
NON_NOREPLY = "someone@" + "example.invalid"        # reserved TLD; cannot be a real address
GITHUB_SERVICE = "noreply@" + "github.com"          # GitHub's own service identity


def _git(args: list[str], cwd: Path, **env: str):
    """Run git with the developer's global and system config neutralised.

    Without this the tests inherit whatever is in ~/.gitconfig, so "no local identity" would not be
    reproducible on another machine -- and these tests are specifically about identity resolution.
    """
    e = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    e.update(env)
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=e, check=False)


def _repo(tmp_path: Path, *, author: str, committer: str | None = None,
          local_email: str | None = None) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    _git(["init", "-q", "-b", "main"], r)
    if local_email is not None:
        _git(["config", "--local", "user.email", local_email], r)
    (r / "f.txt").write_text("nothing interesting here\n")
    _git(["add", "-A"], r)
    # Author and committer are set per-commit so history can be built independently of config.
    _git(["commit", "-q", "-m", "initial"], r,
         GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL=author,
         GIT_COMMITTER_NAME="T", GIT_COMMITTER_EMAIL=committer or author)
    assert _git(["rev-parse", "HEAD"], r).returncode == 0, "test fixture has no commit"
    return r


def test_ci_mode_passes_a_fresh_checkout_with_no_local_identity(tmp_path, monkeypatch):
    """The exact condition that failed in Actions: clean history, no local identity."""
    monkeypatch.setattr(cnl, "REPO", _repo(tmp_path, author=NOREPLY))
    assert cnl._check_git_identity(ci=True) == []


def test_developer_mode_still_requires_a_local_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(cnl, "REPO", _repo(tmp_path, author=NOREPLY))
    problems = cnl._check_git_identity()
    assert len(problems) == 1 and "LOCAL" in problems[0], problems


def test_developer_mode_rejects_a_configured_non_noreply_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(cnl, "REPO",
                        _repo(tmp_path, author=NOREPLY, local_email=NON_NOREPLY))
    problems = cnl._check_git_identity()
    assert any("repository-local" in p for p in problems), problems
    assert not any(NON_NOREPLY in p for p in problems), "the address must be redacted, not echoed"


def test_ci_mode_still_rejects_a_configured_non_noreply_identity(tmp_path, monkeypatch):
    """--ci forgives an ABSENT identity, not a wrong one. If a runner has one, something set it."""
    monkeypatch.setattr(cnl, "REPO",
                        _repo(tmp_path, author=NOREPLY, local_email=NON_NOREPLY))
    assert any("repository-local" in p for p in cnl._check_git_identity(ci=True))


def test_ci_mode_still_rejects_a_non_noreply_author_in_history(tmp_path, monkeypatch):
    monkeypatch.setattr(cnl, "REPO", _repo(tmp_path, author=NON_NOREPLY))
    problems = cnl._check_git_identity(ci=True)
    assert any("history" in p for p in problems), problems
    assert not any(NON_NOREPLY in p for p in problems), "the address must be redacted, not echoed"


def test_ci_mode_still_rejects_a_non_noreply_committer_in_history(tmp_path, monkeypatch):
    """Author and committer are separate fields; checking only %ae would miss a rebase or a merge."""
    monkeypatch.setattr(cnl, "REPO",
                        _repo(tmp_path, author=NOREPLY, committer=NON_NOREPLY))
    assert any("history" in p for p in cnl._check_git_identity(ci=True))


def test_githubs_synthetic_pr_merge_commit_is_not_a_project_finding(tmp_path, monkeypatch):
    """CI failed on a commit the project never made.

    For a pull_request event `actions/checkout` checks out refs/pull/N/merge -- a commit GitHub
    manufactures, committed by its own GitHub-service noreply identity and AUTHORED with the
    account's public profile address. It exists only in the runner's working copy. The scan flagged
    both addresses as findings about this repository, which is the same defect as reading the global
    git identity: judging an identity the repository does not choose.
    """
    r = _repo(tmp_path, author=NOREPLY)
    (r / "g.txt").write_text("x\n")
    _git(["add", "-A"], r)
    _git(["commit", "-q", "-m", "Merge into main"], r,
         GIT_AUTHOR_NAME="GH", GIT_AUTHOR_EMAIL="public-profile@" + "example.invalid",
         GIT_COMMITTER_NAME="GitHub", GIT_COMMITTER_EMAIL=GITHUB_SERVICE)
    monkeypatch.setattr(cnl, "REPO", r)
    assert cnl._check_git_identity(ci=True) == [], (
        "a commit committed by GitHub's service identity is GitHub's construction, not this "
        "repository's identity hygiene"
    )


def test_a_real_commit_is_still_checked_alongside_a_github_merge(tmp_path, monkeypatch):
    """The skip must be per-commit, not a global amnesty triggered by one GitHub merge."""
    r = _repo(tmp_path, author=NON_NOREPLY)          # a REAL commit with a bad author
    (r / "g.txt").write_text("x\n")
    _git(["add", "-A"], r)
    _git(["commit", "-q", "-m", "Merge into main"], r,
         GIT_AUTHOR_NAME="GH", GIT_AUTHOR_EMAIL="public-profile@" + "example.invalid",
         GIT_COMMITTER_NAME="GitHub", GIT_COMMITTER_EMAIL=GITHUB_SERVICE)
    monkeypatch.setattr(cnl, "REPO", r)
    problems = cnl._check_git_identity(ci=True)
    assert any("history" in p for p in problems), problems
    assert not any("public-profile" in p for p in problems), "only the real commit should be reported"


def test_ci_checks_out_full_history_so_history_gates_are_not_vacuous():
    """Two gates scan history; at fetch-depth 1 they would pass by examining one commit."""
    wf = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "fetch-depth: 0" in wf


def test_ci_mode_keeps_content_and_credential_findings_blocking(tmp_path, monkeypatch, capsys):
    """The mode must not become a general amnesty: content checks still fail the run in CI."""
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "conf.py").write_text('api_key = "' + "a1b2c3d4e5f6g7h8i9j0k1l2" + '"\n')
    (tree / "note.md").write_text("/Users/" + "someone/workspace/thing\n")   # leak-scan: allow
    monkeypatch.setattr(cnl, "REPO", tree)
    monkeypatch.setattr(sys, "argv", ["check_no_leaks.py", "--ci"])
    assert cnl.main() == 1
    out = capsys.readouterr().out
    assert "BLOCKING" in out
    assert "--ci" in out, "the report must say it ran in runner mode"


def test_ci_flag_is_wired_into_the_executed_path():
    """A durable mode is only useful if CI actually reaches it.

    CI runs one command -- `bash scripts/gate.sh --ci` -- so the flag is asserted where it is now
    invoked. Checking the workflow for the scanner's own command line would pass only while the workflow
    re-enumerated every check, which is the duplication that arrangement removed.
    """
    wf = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    gate = (REPO / "scripts" / "gate.sh").read_text()
    assert "bash scripts/gate.sh --ci" in wf, "CI must invoke the gate in --ci mode"
    assert "check_no_leaks.py --ci" in gate, "the gate must run the scanner in --ci mode"


# --- the binary scanner must not report clean over nothing -----------------------------------------

def test_binary_scanner_finds_files_outside_a_git_repo(tmp_path, monkeypatch):
    """A sanitized export has no .git by design, and `git ls-files` returns nothing there. The scanner
    accepted that silently and printed "CLEAN — no identifying metadata in 0 binary file(s)" — a clean
    report over zero files, on precisely the tree that most needs checking. Same shape as this
    scanner's earlier self-exemption: success by looking at nothing."""
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("cbm", REPO / "scripts" / "check_binary_metadata.py")
    assert spec is not None and spec.loader is not None
    cbm = _iu.module_from_spec(spec)
    spec.loader.exec_module(cbm)

    # a directory that is NOT a git repo, containing one candidate binary
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    monkeypatch.setattr(cbm, "REPO", tmp_path)

    found = cbm._tracked_binaries(False)
    assert [p.name for p in found] == ["fig.png"], (
        "outside a git repo the scanner must walk the filesystem, not scan nothing"
    )


def test_binary_scanner_refuses_to_pass_on_zero_candidates(tmp_path, monkeypatch, capsys):
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("cbm2", REPO / "scripts" / "check_binary_metadata.py")
    assert spec is not None and spec.loader is not None
    cbm = _iu.module_from_spec(spec)
    spec.loader.exec_module(cbm)
    monkeypatch.setattr(cbm, "REPO", tmp_path)          # empty dir: no candidates at all
    monkeypatch.setattr(cbm.sys, "argv", ["check_binary_metadata.py"])

    rc = cbm.main()
    assert rc == 1, "zero candidates must not be reported as a pass"
    assert "refusing to report clean" in capsys.readouterr().out


def test_a_bare_presentation_reference_is_caught():
    """The gap review found and this scanner did not.

    Every earlier alternative wanted a QUALIFIED noun -- a "validation" one, a "deck" one, a numbered
    one -- so two shipped configs and a shipped ADR referred to a bare one, in a presentation sense, and
    passed the scan. They would have been published.

    The preposition is what makes the new alternative safe: this codebase uses the same word as a VERB
    for real (a lead-likeness ceiling that moves with its target), and a ceiling moves WITH something,
    never ON one of these. Both directions are asserted, because a pattern that fires on the domain
    vocabulary gets reverted the first time it blocks a legitimate commit.

    Trigger strings are assembled rather than written out, like every other case in this file, so that
    this test cannot trip the rule it is testing.
    """
    rx = next(r for n, r, _ in _compiled() if n == "presentation-framing")
    S = "sli" + "de"
    for bad in (f"before committing this to a {S}", f"before this goes on a {S}",
                f"shown in a {S}", f"put it into the {S}"):
        assert rx.search(bad), f"presentation framing not caught: {bad!r}"
    # The verb sense, and the cheminformatics noun: a compound library.
    D = "de" + "ck"
    for ok in ("the ceiling SLIDES with the target", "slides with the target instead of a constant",
               f"cap the {D}", f"prepare the {D}", f"parse warnings on a big {D}"):
        assert not rx.search(ok), f"domain vocabulary wrongly flagged: {ok!r}"
