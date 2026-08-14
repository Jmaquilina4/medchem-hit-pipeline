"""Pre-publication leak scan. Exits non-zero if anything that must stay private is present.

Research code accumulates incidental disclosure: absolute paths, orchestrator identifiers, execution
ids pasted into notes, credentials in example commands. This scans for those before they become
permanent in version history.

Organisation-specific terms are NOT hardcoded here -- see the local overlay below. A scanner that
names what it detects publishes what it detects.

Designed to be run three ways:
    python scripts/check_no_leaks.py              # scan, human-readable report
    python scripts/check_no_leaks.py --staged     # only git-staged files (pre-commit hook)
    python scripts/check_no_leaks.py --quiet      # CI mode: exit code only

Install as a pre-commit hook:
    printf '#!/bin/sh\\nexec python scripts/check_no_leaks.py --staged\\n' > .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

Anything committed is permanent in git history, so this must pass BEFORE the first commit, not after.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class OverlayError(RuntimeError):
    """The local pattern overlay exists but is unusable. Fatal: a security check must not fail open."""

# ---------------------------------------------------------------------------------------------
# Patterns. Each entry: (name, regex, severity, why)
# severity "block" -> exit non-zero. "warn" -> report only.
# Data-driven so adding a term is a one-line change.
# ---------------------------------------------------------------------------------------------
PATTERNS: list[tuple[str, str, str, str]] = [
    # NOTHING PROVIDER-SPECIFIC IS PUBLISHED HERE.
    #
    # This file previously carried detector patterns naming specific credential issuers and a specific
    # machine-local session path layout. They were patterns, not secrets -- but a published scanner that
    # spells out provider identifiers publishes provider identifiers, which is the rule this project
    # states and then has to keep, so they are a publication blocker in their own right.
    #
    # So the published set is SHAPE-BASED and vendor-neutral: it detects "this looks like an opaque
    # credential" and "this looks like a machine-local scratch path" without naming who issues them.
    # Issuer-specific prefixes live in the untracked overlay (see PRIVATE_PATTERNS below), which is where
    # organisation-specific terms already lived for the same reason.
    #
    # Consequence, stated rather than hidden: a fresh public clone with no overlay runs the shape-based
    # checks only, and says so at the end of every run. That is weaker than the pre-publication scan and
    # is the intended trade.
    # --- credentials, by SHAPE ------------------------------------------------------------------
    ("pem-private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "block",
     "PEM private key block (a format, not a vendor)"),
    ("inline-secret", r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|bearer)\s*[:=]\s*"
                      r"[\"']?[A-Za-z0-9_\-./+]{16,}", "block",
     "a credential-looking value assigned to a credential-looking name"),
    # Requires a DIGIT in the run. Without that it matched prose: a vendor name and a chemical name run
    # together in a ChEMBL assay description ("Sigma-Aldrich" + a long IUPAC-ish word) tripped it in the
    # published snapshots. Issued tokens are alphanumeric; a long run of pure letters is language.
    ("opaque-token",
     r"(?<![A-Za-z0-9])[A-Za-z]{2,6}[_-](?=[A-Za-z0-9]*\d)[A-Za-z0-9]{28,}(?![A-Za-z0-9])", "block",
     "short prefix followed by a long alphanumeric run -- the shape most issued tokens share"),
    # A generic high-entropy detector was tried here and REMOVED. Because `/` belongs to the base64
    # alphabet it matched long URL paths, producing 1019 findings in uv.lock alone; tightening it to
    # require base64-only characters still matched every SHA-256 digest in the provenance records. A
    # warn-level check with a thousand false positives is worse than no check -- a reader learns to skip
    # the section, and a real finding hides inside it. The three patterns above cover the realistic
    # cases (a PEM block, a credential-shaped assignment, a prefixed opaque token) without the noise,
    # and issuer-specific prefixes live in the untracked overlay.

    # --- personal information / machine fingerprints -----------------------------------------
    ("home-path", r"/Users/[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+", "block",
     "absolute home path -- leaks username AND breaks portability"),
    # Any address that is not a platform noreply. Organisation domains belong in the local overlay,
    # not here -- embedding one would publish it.
    ("non-noreply-email", r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
     "warn", "email address -- confirm it is not an organisation mailbox"),
    # Shape-based: an absolute temp path whose component carries a long numeric or hex run is a
    # machine-local scratch location regardless of which tool created it.
    ("machine-local-scratch", r"(?:/private)?/tmp/[A-Za-z0-9_.\-]*(?:\d{4,}|[0-9a-f]{8,})[A-Za-z0-9_.\-]*",
     "block", "machine-local scratch path -- not portable, and identifies a local session"),
    ("macos-abs-font", r"/System/Library/Fonts", "warn",
     "macOS-only absolute path -- breaks reproducibility on Linux"),

    # --- generic organisational-affiliation smell ----------------------------------------------
    # Deliberately name-free: catches the FRAMING rather than any specific company, so this list is
    # safe to publish and still useful to anyone reusing the script.
    # Written with character classes so this file does not itself contain the phrases it blocks.
    ("audience-framing", r"(?i)targ[e]t audien[c]e|for an audien[c]e|pitch(?:ing)? to|"
                         r"position[i]ng (?:for|toward)|inter[v]iew|recruit(?:er|ing)|hir[i]ng manager",
     "block", "positioning framing -- research code should not name an intended reader"),
    ("affiliation-claim", r"(?i)\bmy (?:compan[y]|emplo[y]er|team at)\b|\bour (?:compan[y]|CT[O]|VP)\b|"
                          r"\bCT[O]\b|\bemplo[y]er\b|compan[y] secrets?|outsid[e] work",
     "block", "employment-relationship content"),

    # --- presentation framing -------------------------------------------------------------------
    # Added after an audit found "if shown in the talk", "validation slide" and "the deck" surviving
    # into a published tree. The audience-framing pattern above had no term for a talk, a deck or a
    # slide, so it could not see any of them. Research code should not be organised around a
    # presentation; and a reference to an unpublished deck is a dangling citation besides.
    # NB "the deck" alone is NOT matched: in cheminformatics a deck is the compound library being
    # screened ("cap the deck", "prepare the deck"), which is correct domain vocabulary. Only
    # presentation senses are matched -- a deck that has SLIDES, or a talk.
    #
    # The `(?:on|onto|in|into|to)\s+(?:a|the)\s+slide` alternative closes a gap found by review, not by
    # this scanner: two shipped configs said "before committing this to a slide" and a shipped ADR said
    # "before this goes on a slide". Every existing alternative wanted a QUALIFIED slide -- "validation
    # slide", "deck slide", "slide 15" -- so a bare "a slide" passed. The preposition is what keeps this
    # from firing on "slide" as a verb, which this codebase uses for real ("the ceiling SLIDES with the
    # target"): a lead-likeness ceiling slides WITH something, it never slides ON a slide.
    ("presentation-framing", r"(?i)\bin the tal[k]\b|\bdec[k] sli[d]e\b|\bv1 dec[k]\b|"
                             r"\b(?:validation|title|closing)\s+sli[d]e\b|\bsli[d]e\s+\d+\b|"
                             r"\b(?:on|onto|in|into|to)\s+(?:a|the)\s+sli[d]e\b|"
                             r"\bdec[k]'s\s+(?:mislabel|selectivity|multi-model|column|graphic)|"
                             r"\bpitch dec[k]\b|\bif shown in the\b",
     "block", "presentation framing, or a citation to an unpublished deck"),

    # --- self-framing for an evaluating reader ---------------------------------------------------
    # "fast-follower" is deliberately absent: it names a real drug-discovery strategy (a follow-on
    # compound against an already-validated target), which is this project's actual scientific premise.
    ("self-framing", r"(?i)portfoli[o] (?:repositor|project|piece)|showcas(?:e|ing) (?:my|our)\b",
     "block", "self-framing for an evaluating reader"),
]


def _author_name_patterns() -> list[tuple[str, str, str, str]]:
    """Derive personal-name patterns AT RUNTIME from the files entitled to hold the name.

    A real first name reached a published ADR, in a sentence that also disclosed privileged access
    to a commercial software licence, and this scanner passed it,
    because no pattern here matches a personal name -- deliberately: hardcoding the name would publish
    it in the very script meant to catch it. The project's own provenance sanitizer *did* hardcode it,
    which is the same mistake wearing the opposite hat.

    So the name is read from ``pyproject.toml``, a file that legitimately contains it, and turned into
    patterns at runtime. Nothing identifying is stored in this source file, and the check still fires.
    ``LICENSE``, ``CITATION.cff`` and ``pyproject.toml`` are exempt by path -- attribution belongs there.
    """
    py = REPO / "pyproject.toml"
    if not py.is_file():
        return []
    m = re.search(r'authors\s*=\s*\[\s*\{\s*name\s*=\s*"([^"]+)"', py.read_text(encoding="utf-8"))
    if not m:
        return []
    pats = []
    for token in {t for t in re.split(r"[\s,]+", m.group(1)) if len(t) > 2}:
        pats.append((
            "author-name", rf"(?i)\b{re.escape(token)}\b", "block",
            "the author's real name outside LICENSE / CITATION.cff / pyproject.toml",
        ))
    return pats


# Files entitled to carry the author's name. Attribution is the point of these three.
NAME_EXEMPT_PATHS = {"LICENSE", "CITATION.cff", "pyproject.toml", "NOTICE"}

# Patterns judged only against what is actually PUBLISHED. A development tree legitimately contains
# development notes that name the author, cite an unpublished deck, or reference a slide -- none of which
# matters if the file never ships. Credentials and absolute paths are NOT in this set: those must not
# exist anywhere, published or not, because history is permanent.
PUBLICATION_SCOPED = {"author-name", "presentation-framing", "self-framing",
                      "audience-framing", "affiliation-claim"}


def _published_paths() -> set[str] | None:
    """Repo-relative paths the sanitized export would ship, or None if that cannot be determined.

    Returning None means "no scope information", and the caller then applies every pattern to every
    file -- fail-closed. A scanner that silently narrowed its scope on a missing import would be the
    same defect this project has now hit three times in other tools.
    """
    exporter = REPO / "scripts" / "export_public.py"
    if not exporter.is_file():
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_export_public", exporter)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                                 text=True).stdout.split()
        if not tracked:
            return None
        return set(mod._selected(tracked))
    except Exception:  # noqa: BLE001 - unavailable scope must widen the scan, never narrow it
        return None

# ---------------------------------------------------------------------------------------------
# Local overlay, UNTRACKED and INERT (TOML, never executed). Put organisation-specific terms --
# names, tenant ids, internal pipeline names -- in scripts/leak_patterns_local.toml:
#
#     [[pattern]]\n#     name = "org-name"\n#     regex = '\\bacmecorp\\b'\n#     severity = "block"
#
# The scan degrades gracefully if the file is absent, so a fresh clone still gets the generic checks.
# ---------------------------------------------------------------------------------------------
def _load_private_patterns(path: Path | None = None) -> list[tuple[str, str, str, str]]:
    """Read the untracked overlay as INERT DATA.

    An earlier version exec'd a .py file. For a file that is untracked, edited by hand, and loaded by
    a security check, executing it is the wrong default: a scanner that runs arbitrary code to decide
    what to scan has a larger attack surface than the thing it protects. TOML cannot execute.
    """
    import tomllib

    if path is None:
        path = Path(__file__).with_name("leak_patterns_local.toml")
    if not path.exists():
        return []          # absent is a legitimate state: a fresh clone gets the generic checks
    # Present-but-broken is NOT. Warning and dropping the overlay lets the scan report CLEAN while
    # the rules it was supposed to apply are silently absent -- a fail-open in a security check.
    try:
        doc = tomllib.loads(path.read_text())
    except Exception as exc:
        raise OverlayError(f"{path.name} exists but could not be parsed: {exc}") from exc
    rows = doc.get("pattern")
    if not rows:
        raise OverlayError(f"{path.name} exists but defines no [[pattern]] entries")
    out = []
    for i, row in enumerate(rows, 1):
        missing = [k for k in ("name", "regex", "severity") if k not in row]
        if missing:
            raise OverlayError(f"{path.name} pattern #{i} is missing {missing}")
        if row["severity"] not in ("block", "warn"):
            raise OverlayError(f"{path.name} pattern #{i} severity must be block|warn")
        try:
            re.compile(row["regex"])
        except re.error as exc:
            raise OverlayError(f"{path.name} pattern #{i} ({row['name']}) has an invalid regex: {exc}") from exc
        out.append((row["name"], row["regex"], row["severity"], row.get("why", "")))
    return out

# Paths never scanned: third-party code, caches, virtualenvs, and outputs that are gitignored anyway.
SKIP_DIRS = {
    ".git", ".venv", "venv", "external", "__pycache__", ".ruff_cache", ".pytest_cache",
    ".medchem_cache", "node_modules", ".mypy_cache", "runs", ".DS_Store",
}
# Binary / non-text suffixes we cannot meaningfully grep (flagged separately for metadata stripping).
BINARY_SUFFIXES = {".pptx", ".png", ".jpg", ".jpeg", ".pdf", ".gz", ".zip", ".joblib", ".pkl",
                   ".npz", ".ckpt", ".prior", ".pdbqt", ".so", ".dylib", ".ico", ".svg"}

# Lines matching these are exempt: this scanner's own pattern definitions, and documented placeholders.
# Path-level exemptions. Deliberately does NOT include this file: a blanket self-exemption previously
# made the scanner unable to see 13 genuine blocking matches inside its own source, which is how an
# organisation domain survived in a committed pattern. Only the pattern-definition block below is
# exempt, delimited by sentinels, so the rest of this file is scanned like any other.
ALLOWLIST_PATHS = [
    re.compile(r"leak_patterns_local\.(py|toml)"),  # the local overlay -- untracked by design
    # A generated lockfile: nothing but package names, URLs and digests. Its content cannot carry a
    # disclosure that is not already visible in pyproject.toml, and scanning it is pure noise.
    re.compile(r"(^|/)uv\.lock$"),
]
ALLOWLIST_RE = [
    re.compile(r"#\s*leak-scan:\s*allow"),      # explicit per-line opt-out
    re.compile(r"<YOUR_[A-Z_]+>"),              # documented placeholder
]
# Lines between these markers in THIS file are pattern definitions, not content.
# Files that legitimately CONTAIN the strings they detect: a scanner's pattern list. Each names the
# exact region to skip, so the rest of the file is scanned like any other. Enumerated per file rather
# than granted by a generic opt-in marker, because a generic marker could exempt anything.
_PATTERN_DEFINING_FILES: dict[str, tuple[str, str]] = {
    "check_no_leaks.py": ("PATTERNS: list[tuple[str, str, str, str]] = [", "# Path-level exemptions."),
    "check_release_ready.py": ("HISTORY_PATTERNS: tuple[tuple[str, str], ...] = (",
                               "NAME_EXEMPT = "),
}


def _iter_files(staged_only: bool) -> list[Path]:
    if staged_only:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                             cwd=REPO, capture_output=True, text=True)
        return [REPO / p for p in out.stdout.split() if (REPO / p).is_file()]
    # Prefer git's own view: report exactly what WOULD be committed. A full-tree walk otherwise
    # descends into whatever a working tree happens to hold outside version control, and buries the
    # findings that matter under files that were never going to be published.
    #
    # This comment named two of those paths until review pointed out that this script SHIPS. Naming
    # the private directories a scanner skips publishes their names -- the same defect as documenting
    # a leak by quoting it, and the reason the export's own allowlist tool is not shipped either.
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip() == "true"
    if inside:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=REPO, capture_output=True, text=True)
        return [REPO / p for p in out.stdout.split() if (REPO / p).is_file()]
    files = []
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
            continue
        files.append(p)
    return files


def _exempt_span(lines: list[str], start_s: str, end_s: str) -> tuple[int, int] | None:
    """The single pattern-definition region: first start sentinel, first end sentinel AFTER it.

    Resolved by index, not by a running toggle. The toggle looked equivalent and was not: a line
    containing BOTH sentinels takes the ``if`` branch and can never take the ``elif``, so the region
    opens and never closes. Exactly one such line exists here -- the ``_PATTERN_DEFINING_FILES``
    declaration, which necessarily quotes both -- and it sits mid-file, so most of this scanner was
    exempt from its own scan. Same bug, independently, in check_release_ready.py.

    Returns None when the end sentinel is missing, so the caller scans the WHOLE file: a missing
    sentinel must never be able to blank a file to EOF.
    """
    start = next((i for i, ln in enumerate(lines) if start_s in ln), None)
    if start is None:
        return None
    end = next((i for i, ln in enumerate(lines[start + 1:], start + 1) if end_s in ln), None)
    return None if end is None else (start, end)


def _redact_email(addr: str) -> str:
    """``name@host`` -> ``n***@h***``. A diagnostic must not print the thing it is warning about."""
    local, _, host = addr.partition("@")
    keep = lambda t: (t[:1] + "***") if t else "***"  # noqa: E731
    return f"{keep(local)}@{keep(host)}" if host else keep(local)


def _check_git_identity(*, ci: bool = False) -> list[str]:
    """Commit authorship is metadata, not file content -- a file scan cannot see it.

    An organisation address in git config becomes permanent in every commit it signs, and rewriting
    authorship later means rewriting history. This checks the identity the NEXT commit would use, plus
    every author/committer already in the log.

    Two constraints on the check itself, rather than on the repository it inspects:

    * Reading the GLOBAL git identity makes a fresh clone report the *operator's own* personal
      address as a finding about this project. ``--local`` only: a clone that has not set a local identity
      has nothing for this check to judge, and saying so is correct.
    * Printing the address it warns about would put that address into local output and into any CI log,
      so values are redacted -- the operator knows their own address and does not need it echoed.

    ``ci`` narrows exactly ONE branch: the absence of a local identity. That finding exists to stop a
    developer's next commit inheriting a global address, and a CI checkout creates no commits, so on a
    runner it is structurally unreachable rather than merely unlikely -- it made the scan fail in every
    clean checkout, which trains people to ignore it. Everything else is unchanged and still blocking in
    CI, including a configured non-noreply local identity (if a runner has one, something set it, and
    that is worth failing over) and every author/committer address already in the log, which is the
    check that actually protects published history.
    """
    problems = []
    cfg = subprocess.run(["git", "config", "--local", "user.email"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    if not cfg:
        if not ci:
            problems.append("no repository-LOCAL git user.email is set; the next commit would inherit "
                            "the global identity, which this check deliberately does not read. Set one: "
                            "git config --local user.email <user>@users.noreply.github.com")
    elif not cfg.endswith("users.noreply.github.com"):
        problems.append(f"repository-local git user.email is '{_redact_email(cfg)}' -- the next commit "
                        f"would embed it permanently. Prefer a noreply address.")
    # GitHub's own service identity. Commits COMMITTED by it were manufactured by GitHub's
    # infrastructure rather than by anyone's git config -- most visibly the ephemeral
    # refs/pull/N/merge commit that `actions/checkout` checks out for a pull_request event, which
    # exists only in the runner's working copy and is never pushed anywhere.
    #
    # Such a commit must be skipped ENTIRELY, not just in its committer field, because GitHub sets its
    # AUTHOR to the account's public profile address -- a value this repository does not choose and
    # cannot change from here. Judging it reported a finding about the project that was not about the
    # project, which is the same defect as reading the global git identity: the check has to judge
    # identities this repository is responsible for.
    #
    # The trade: a merge performed through GitHub's web UI is committed by this same identity, so a
    # personal author address on such a merge would not be caught. Nothing in the release path uses
    # one -- the export's commits are made locally by git commit, so every commit in a published tree
    # is still fully checked.
    # Concatenated so this line is not email-SHAPED. scripts/check_release_ready.py scans history
    # blobs with its own patterns and no exemption marker, and its address pattern DOES match this
    # value; that it passed relied on a whole-file exemption rather than on the line being safe.
    GITHUB_SERVICE = "noreply@" + "github.com"
    log = subprocess.run(["git", "log", "--format=%ae%x00%ce"], cwd=REPO,
                         capture_output=True, text=True).stdout.splitlines()
    bad: set[str] = set()
    for line in log:
        author, _, committer = line.partition("\x00")
        if committer == GITHUB_SERVICE:
            continue
        bad |= {e for e in (author, committer)
                if e and not e.endswith("users.noreply.github.com")}
    for e in sorted(bad):
        problems.append(f"commit history contains author/committer email '{_redact_email(e)}' -- already "
                        f"permanent; rewriting requires filter-repo")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged", action="store_true", help="only git-staged files (pre-commit hook)")
    ap.add_argument("--quiet", action="store_true", help="exit code only, no report")
    ap.add_argument("--ci", action="store_true",
                    help="runner mode: a checkout with no repository-local git identity is not a "
                         "finding (CI creates no commits). Every other check, including all "
                         "author/committer addresses already in history, stays blocking.")
    args = ap.parse_args()

    try:
        private = _load_private_patterns()
    except OverlayError as exc:
        print(f"\n  FATAL: {exc}\n  The overlay is present but unusable. Refusing to report a result "
              f"from an incomplete rule set.\n")
        return 2
    if not private and not args.quiet:
        print("  NOTE: scripts/leak_patterns_local.toml not found -- running GENERIC checks only.\n"
              "        Create it (untracked) to also scan organisation-specific terms.\n")
    compiled = [(n, re.compile(rx), sev, why)
                for n, rx, sev, why in PATTERNS + private + _author_name_patterns()]
    published = _published_paths()
    hits: list[tuple[str, str, int, str, str, str]] = []   # sev, file, line, pattern, why, snippet
    binaries: list[Path] = []

    for path in _iter_files(args.staged):
        rel = path.relative_to(REPO).as_posix()
        if path.suffix.lower() in BINARY_SUFFIXES:
            binaries.append(path)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(a.search(rel) for a in ALLOWLIST_PATHS):
            continue
        sentinels = next((v for k, v in _PATTERN_DEFINING_FILES.items() if rel.endswith(k)), None)
        all_lines = text.splitlines()
        span = _exempt_span(all_lines, *sentinels) if sentinels is not None else None
        for i, line in enumerate(all_lines, 1):
            # [start, end) by index, resolved once. A running toggle could not close the region: the
            # _PATTERN_DEFINING_FILES declaration below quotes BOTH sentinels on one line, so
            # `if start ... elif end ...` reopened the exemption and ran to EOF -- exempting most of
            # each scanner from its own scan. See _exempt_span.
            if span is not None and span[0] <= i - 1 < span[1]:
                continue
            if any(a.search(line) for a in ALLOWLIST_RE):
                continue
            for name, rx, sev, why in compiled:  # noqa: B007
                # Attribution files are entitled to the author's name; every other pattern still applies
                # to them, so this is a per-pattern exemption rather than a per-file one.
                if name == "author-name" and Path(rel).name in NAME_EXEMPT_PATHS:
                    continue
                # Publication-scoped patterns downgrade to a warning in files the export drops: worth
                # seeing, not worth blocking a private commit over.
                scoped_out = (name in PUBLICATION_SCOPED and published is not None
                              and str(rel) not in published)
                m = rx.search(line)
                if m:
                    snip = line.strip()[:100]
                    # redact the match itself so the report is safe to paste anywhere
                    snip = snip.replace(m.group(0), "<REDACTED>")
                    hits.append(("warn" if scoped_out else sev, rel, i, name,
                                 why + (" [not published; informational]" if scoped_out else ""), snip))

    git_problems = _check_git_identity(ci=args.ci)

    blocking = [h for h in hits if h[0] == "block"]
    warnings = [h for h in hits if h[0] == "warn"]

    if not args.quiet:
        print("=" * 92)
        print(f"PRE-PUBLICATION LEAK SCAN{'  [--ci: runner mode]' if args.ci else ''}")
        print("=" * 92)
        if blocking:
            print(f"\n  {len(blocking)} BLOCKING finding(s) — must be fixed before committing:\n")
            for _, rel, ln, name, why, snip in sorted(blocking):
                print(f"    {rel}:{ln}")
                print(f"      [{name}] {why}")
                print(f"      {snip}")
        if warnings:
            print(f"\n  {len(warnings)} warning(s) — review, not blocking:\n")
            seen: dict[str, int] = {}
            for _, rel, _ln, name, _why, _snip in sorted(warnings):
                seen[f"{rel} [{name}]"] = seen.get(f"{rel} [{name}]", 0) + 1
            for k, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"    {k}: {n} occurrence(s)")
        if binaries:
            print(f"\n  {len(binaries)} binary file(s) not scanned — check embedded author metadata "
                  f"before publishing (e.g. `exiftool`, or strip on export):")
            for b in sorted(binaries)[:10]:
                print(f"    {b.relative_to(REPO).as_posix()}")
            if len(binaries) > 10:
                print(f"    ... and {len(binaries) - 10} more")
        if git_problems:
            print(f"\n  {len(git_problems)} git-identity issue(s):\n")
            for g in git_problems:
                print(f"    {g}")
        if not blocking and not warnings and not git_problems:
            # State what was actually verified. In --ci with no local identity there is no next-commit
            # identity to vouch for, and claiming one was checked would be the same class of overclaim
            # this scan exists to catch.
            print("\n  CLEAN — no leaks detected; every author/committer address in history is a "
                  "noreply address." if args.ci else
                  "\n  CLEAN — no leaks detected, git identity is a noreply address.")
        print()

    return 1 if (blocking or git_problems) else 0


if __name__ == "__main__":
    sys.exit(main())
