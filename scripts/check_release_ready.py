"""Three release checks that no existing tool performs. Run against a sanitized EXPORT, not the source.

Each of these exists because a previous check looked at the wrong thing and reported clean:

1. **Full history, not just the final tree.** ``check_no_leaks.py`` scans the working tree. A two-commit
   export whose FIRST commit contained a disclosure later removed in the second would pass every
   tree-level scan and still publish the disclosure permanently — git history is the artifact, not the
   checkout. This walks every blob in every commit.

2. **Decompressed snapshot CONTENT.** ``check_binary_metadata.py`` inspects gzip *headers*; the leak
   scanner skips ``.gz`` as binary. So the 16 published ChEMBL files — the only data in the repository —
   have never had their actual text examined. A ChEMBL assay description is free text written by
   depositors; nothing guarantees it is free of paths, addresses or names.

3. **Every relative documentation link resolves.** Excluding a document from the export is exactly the
   operation that breaks a link in a surviving one. Four such links have existed at once, and every one
   was found by reading the documents rather than by any check — which is what this replaces.

4. **Every path a documented command names exists.** A README instruction that fails is worse than a
   missing one, because the reader concludes the project does not work. Commands are extracted from
   fenced bash blocks and every repo-relative path token in them is checked for existence.

   It does **not execute** them: running the documented pipeline commands would take tens of minutes and
   write into the tree being audited. So this catches the common failure (a command naming a file that
   was renamed or dropped) and not the rarer one (a command whose paths all exist but which fails for
   another reason). The distinction is stated because a checker that implies more coverage than it
   has is worse than one that states its limits.

Usage:
    python scripts/check_release_ready.py --export /path/to/export
"""

from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys
from pathlib import Path

# Patterns that must not appear in ANY historical blob or inside ANY decompressed data file.
# Kept name-free: the author name is derived at runtime, exactly as check_no_leaks.py does it, so this
# script does not itself publish what it looks for.
HISTORY_PATTERNS: tuple[tuple[str, str], ...] = (
    # SHAPE-BASED and vendor-neutral, for the same reason as check_no_leaks.py: a published scanner that
    # spells out issuer-specific prefixes publishes issuer-specific prefixes, which is precisely the rule
    # this project states, so a provider-named pattern set is a publication blocker in its own right.
    # Issuer specifics live in the untracked overlay; what ships detects the SHAPE of a credential and of
    # a machine-local path.
    ("home-path", r"/Users/[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+"),
    ("machine-local-scratch",
     r"(?:/private)?/tmp/[A-Za-z0-9_.\-]*(?:\d{4,}|[0-9a-f]{8,})[A-Za-z0-9_.\-]*"),
    ("pem-private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # Requires a DIGIT: without it this matched a vendor name and a chemical name run together in a
    # ChEMBL assay description, inside the published snapshots.
    ("opaque-token",
     r"(?<![A-Za-z0-9])[A-Za-z]{2,6}[_-](?=[A-Za-z0-9]*\d)[A-Za-z0-9]{28,}(?![A-Za-z0-9])"),
    ("inline-secret", r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|bearer)\s*[:=]\s*"
                      r"[\"']?[A-Za-z0-9_\-./+]{16,}"),
    ("non-noreply-email", r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("org-directory-term", r"(?i)\btenant[_ -]?id\b|\bdirectory (?:object )?id\b"),
)
NAME_EXEMPT = {"LICENSE", "CITATION.cff", "pyproject.toml", "NOTICE"}

# Both scanners legitimately CONTAIN the strings they detect, in their pattern lists. Each names the
# exact region to skip -- per file, never a blanket self-exemption, which is how check_no_leaks.py once
# became unable to see 13 real matches inside its own source. THIS file is in the map too: the first
# version exempted only the other scanner, and its own pattern list then failed the history check.
_PATTERN_DEFINING_FILES: dict[str, tuple[str, str]] = {
    "check_no_leaks.py": ("PATTERNS: list[tuple[str, str, str, str]] = [", "# Path-level exemptions."),
    "check_release_ready.py": ("HISTORY_PATTERNS: tuple[tuple[str, str], ...] = (", "NAME_EXEMPT = "),
}


def _exempt_span(lines: list[str], start_s: str, end_s: str) -> tuple[int, int] | None:
    """The single pattern-definition region: first start sentinel, first end sentinel AFTER it.

    Resolved by index rather than by a running toggle. The toggle looked equivalent and was not::

        if start in line:  inside = True
        elif end in line:  inside = False

    A line containing BOTH sentinels takes the first branch and can never take the second, so the
    region opens and never closes. Exactly one such line exists in each scanner -- the
    ``_PATTERN_DEFINING_FILES`` declaration itself, which necessarily quotes both sentinels -- and it
    sits in the middle of the file. The result was that 377 of 476 lines of one scanner and 212 of 282
    of the other were blanked before this history scan ran: a silent, near-total self-exemption in the
    last gate before publication, which is the exact defect this project documented in one scanner and
    then reproduced in the other.

    Returns None when the end sentinel is absent, so the caller scans the WHOLE file. Failing closed
    matters more here than a clean report: a missing sentinel must not be able to blank a file to EOF.
    """
    start = next((i for i, ln in enumerate(lines) if start_s in ln), None)
    if start is None:
        return None
    end = next((i for i, ln in enumerate(lines[start + 1:], start + 1) if end_s in ln), None)
    return None if end is None else (start, end)


def _strip_pattern_definitions(path: str, text: str) -> str:
    """Blank out a scanner's pattern-definition region, for that file only."""
    sentinels = next((v for k, v in _PATTERN_DEFINING_FILES.items() if path.endswith(k)), None)
    if sentinels is None:
        return text
    lines = text.splitlines()
    span = _exempt_span(lines, *sentinels)
    if span is None:
        return text
    start, end = span
    # [start, end): the end sentinel line itself is ordinary content and stays scanned.
    return "\n".join("" if start <= i < end else ln for i, ln in enumerate(lines))


def _author_tokens(export: Path) -> list[str]:
    py = export / "pyproject.toml"
    if not py.is_file():
        return []
    m = re.search(r'authors\s*=\s*\[\s*\{\s*name\s*=\s*"([^"]+)"', py.read_text(encoding="utf-8"))
    return [t for t in re.split(r"[\s,]+", m.group(1))] if m else []


def check_history(export: Path) -> list[str]:
    """Every blob in every commit, not the checkout."""
    problems: list[str] = []
    commits = subprocess.run(["git", "rev-list", "--all"], cwd=export,
                             capture_output=True, text=True).stdout.split()
    if not commits:
        return ["no commits found — refusing to report a clean history scan over nothing"]

    # BLOBS REACHED FROM COMMIT TREES ONLY.
    #
    # `rev-list --objects --all` also emits ANNOTATED TAG objects, and a tag object's content is its
    # message plus tagger attribution -- so scanning it flagged ordinary tagger metadata as a finding in
    # repository content. That is a false positive about the wrong kind of object: a tag is not a file,
    # and its authorship is checked by the identity check, not by a content scan.
    #
    # `--filter=blob:none --objects` still lists trees; `cat-file --batch-check` gives the type, so
    # non-blobs are dropped explicitly rather than inferred from a filename.
    listing = subprocess.run(["git", "rev-list", "--objects", "--no-object-names", "--all"],
                             cwd=export, capture_output=True, text=True).stdout.split()
    named = subprocess.run(["git", "rev-list", "--objects", "--all"], cwd=export,
                           capture_output=True, text=True).stdout.splitlines()
    types = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
                           cwd=export, input="\n".join(listing), capture_output=True, text=True).stdout
    blob_shas = {ln.split()[0] for ln in types.splitlines()
                 if len(ln.split()) == 2 and ln.split()[1] == "blob"}
    blobs: dict[str, str] = {}
    for line in named:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0] in blob_shas:
            blobs[parts[0]] = parts[1]

    name_pats = [(f"author-name:{t}", rf"(?i)\b{re.escape(t)}\b") for t in _author_tokens(export)
                 if len(t) > 2]
    compiled = [(n, re.compile(p)) for n, p in (*HISTORY_PATTERNS, *name_pats)]
    skip_suffix = {".gz", ".png", ".jpg", ".pdf", ".joblib", ".pkl", ".so", ".dylib"}
    scanned = 0
    for sha, path in sorted(blobs.items(), key=lambda kv: kv[1]):
        if Path(path).suffix.lower() in skip_suffix or "uv.lock" in path:
            continue
        raw = subprocess.run(["git", "cat-file", "-p", sha], cwd=export, capture_output=True).stdout
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        text = _strip_pattern_definitions(path, text)
        scanned += 1
        base = Path(path).name
        for name, rx in compiled:
            if name.startswith("author-name") and base in NAME_EXEMPT:
                continue
            if rx.search(text):
                problems.append(f"[{name}] blob {sha[:10]} at {path} (present in history)")
    print(f"    {len(commits)} commit(s), {scanned} text blob(s) scanned")
    if scanned == 0:
        problems.append("scanned zero text blobs — discovery is broken, not the history")
    return problems


def check_snapshot_content(export: Path) -> list[str]:
    """Decompress every published snapshot and scan the actual CSV text."""
    problems: list[str] = []
    snaps = sorted((export / "data" / "frozen_snapshots").glob("*.gz"))
    if not snaps:
        return ["no snapshots found — refusing to report clean over nothing"]

    name_pats = [(f"author-name:{t}", rf"(?i)\b{re.escape(t)}\b") for t in _author_tokens(export)
                 if len(t) > 2]
    compiled = [(n, re.compile(p)) for n, p in (*HISTORY_PATTERNS, *name_pats)]
    total_bytes = 0
    for f in snaps:
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        total_bytes += len(text)
        for name, rx in compiled:
            m = rx.search(text)
            if m:
                # Report the finding WITHOUT echoing the match, per docs/PITFALLS.md.
                line_no = text.count("\n", 0, m.start()) + 1
                problems.append(f"[{name}] {f.name} line {line_no} (decompressed content)")
    print(f"    {len(snaps)} snapshot(s), {total_bytes / 1e6:.1f} MB of decompressed text scanned")
    return problems


def check_relative_links(export: Path) -> list[str]:
    """Every relative markdown link and image must resolve inside the export.

    Added because broken links are found by hand otherwise, one at a time: three of them, then a fourth
    introduced by excluding a document that a surviving page still linked. Excluding a file is exactly
    the operation that creates this defect, so the export is the right place to check for it.
    """
    problems: list[str] = []
    docs = [f for f in sorted(export.rglob("*.md"))
            if not any(x in f.parts for x in (".venv", ".git"))]
    if not docs:
        return ["no markdown found — refusing to report clean over nothing"]
    checked = 0
    for f in docs:
        for m in re.finditer(r"\[[^\]]+\]\(([^)#\s]+)(#[^)\s]*)?\)", f.read_text(encoding="utf-8")):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            if not (f.parent / target).exists():
                problems.append(f"{f.relative_to(export)} -> {target!r} does not resolve")
    print(f"    {len(docs)} document(s), {checked} relative link(s) checked")
    return problems


def check_documented_commands(export: Path) -> list[str]:
    """Every fenced bash command in the README, checked for referenced paths that exist.

    Existence only — see the module docstring. Nothing here is executed."""
    problems: list[str] = []
    readme = export / "README.md"
    if not readme.is_file():
        return ["README.md absent"]
    blocks = re.findall(r"```bash\n(.*?)```", readme.read_text(encoding="utf-8"), re.S)
    if not blocks:
        return ["no fenced bash blocks found in README — refusing to report clean over nothing"]
    checked = 0
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            checked += 1
            # Any repo-relative path token the command names must exist.
            for tok in re.findall(r"(?:configs|scripts|data|provenance|docs)/[A-Za-z0-9_./-]+", line):
                if not (export / tok).exists():
                    problems.append(f"README command references a missing path: {tok!r} in {line!r}")
    print(f"    {len(blocks)} bash block(s), {checked} command line(s) checked")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, help="path to the sanitized export")
    args = ap.parse_args()
    export = Path(args.export).resolve()
    if not (export / ".git").is_dir():
        raise SystemExit(f"{export} is not a git repository; the history check needs one")

    # These checks judge a SANITIZED EXPORT. Pointed at the development tree they fail by design: its
    # history legitimately contains working notes that name the author, and that is the reason the export
    # exists. SKIPPED is reported distinctly from CLEAN -- a skip is not a pass, and the export path can
    # never skip, because export_public.py always writes the manifest.
    if not (export / "EXPORT_MANIFEST.json").is_file():
        print("=" * 92)
        print("RELEASE READINESS — SKIPPED")
        print("=" * 92)
        print(f"\n  {export} has no EXPORT_MANIFEST.json, so it is not a sanitized export.")
        print("  These checks scan full history and published data for content that must not be")
        print("  published; a development tree legitimately contains such content in its history.")
        print("  This check applies to a sanitized export, identified by EXPORT_MANIFEST.json at its "
              "root.\n")
        return 0

    print("=" * 92)
    print("RELEASE READINESS — four checks no other tool performs")
    print("=" * 92)
    failed = 0
    for title, fn in (
        ("Full git history (every blob in every commit)", check_history),
        ("Decompressed snapshot content (16 CSVs, not their gzip headers)", check_snapshot_content),
        ("Relative documentation links resolve", check_relative_links),
        ("Documented commands' paths exist (not executed)", check_documented_commands),
    ):
        print(f"\n  {title}")
        problems = fn(export)
        if problems:
            failed += 1
            print(f"    {len(problems)} PROBLEM(S):")
            for p in problems[:15]:
                print(f"        {p}")
        else:
            print("    clean")
    print()
    if failed:
        print(f"  {failed} of 4 checks FAILED — not release ready.")
        return 1
    print("  all 4 checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
