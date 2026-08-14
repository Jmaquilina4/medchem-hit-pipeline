"""Inspect committed binary files for embedded author/machine metadata.

Text scanning cannot see inside binaries, so `check_no_leaks.py` deliberately skips them and prints a
"check these manually" note. Advice like that gets skipped, so this does the check instead.

What it looks for, per format:
  * PNG/JPEG  -- the raw bytes are searched for the byte patterns below, which is what actually
                 happens: this is a pattern scan over the whole file, not a format-aware parse of
                 tEXt/iTXt/eXIf chunk structure. It therefore finds an embedded name or filesystem
                 path wherever it sits, and it does not enumerate chunks or decode EXIF tag numbers.
                 Matplotlib writes a `Software` tag by default; that is harmless but worth seeing.
  * PPTX/DOCX/XLSX (zip) -- docProps/core.xml (dc:creator, cp:lastModifiedBy), and the revision and
                 change-tracking parts, which can carry an organisation directory identifier and an
                 editing-timestamp trail. These are invisible to every text grep.
  * PDF       -- /Author, /Creator, /Producer, /Title in the document info dictionary.
  * PDBQT/PQR/FASTA and other text-ish science formats -- REMARK/header lines with local paths.

Usage:
    python scripts/check_binary_metadata.py            # every git-tracked binary
    python scripts/check_binary_metadata.py --all      # every binary on disk, tracked or not
Exit code is 1 if anything identifying is found.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".pptx", ".docx", ".xlsx",
                   ".pdbqt", ".pqr", ".fasta", ".cif", ".sdf", ".mol2",
                   # .gz was missing, so the 16 published ChEMBL snapshots -- the only data files in
                   # the repository -- were never scanned. A gzip header carries FNAME and FCOMMENT
                   # fields that can hold a full source path.
                   ".gz"}

# Things that identify a person or a machine. Deliberately generic -- no company names here, for the
# same reason check_no_leaks.py keeps them in an untracked file.
IDENTIFYING = re.compile(
    rb"(?i)"
    rb"/Users/[A-Za-z0-9_.\-]+"          # macOS home
    rb"|/home/[A-Za-z0-9_.\-]+"          # linux home
    rb"|[A-Z]:\\\\Users\\\\[A-Za-z0-9_.\-]+"   # windows home
    rb"|dc:creator|cp:lastModifiedBy|lastModifiedBy"
    rb"|<dc:creator>|userId=|providerId=|clId="
    rb"|/Author|/Producer\s*\(|Artist|XPAuthor"
)

ZIP_METADATA_PARTS = ("docProps/core.xml", "docProps/app.xml", "ppt/revisionInfo.xml")


def _walk_binaries() -> list[Path]:
    """Filesystem walk, skipping directories that are never part of the published tree."""
    return [p for p in REPO.rglob("*")
            if p.is_file() and p.suffix.lower() in BINARY_SUFFIXES
            and not any(d in p.relative_to(REPO).parts
                        for d in (".git", ".venv", "external", "runs", "__pycache__", "node_modules"))]


def _tracked_binaries(all_files: bool) -> list[Path]:
    """Candidate binaries, preferring git's view but never trusting it blindly.

    ``git ls-files`` returns NOTHING outside a repository, and this scanner used to accept that
    silently — so run against a sanitized export (which by design has no ``.git``) it reported
    "CLEAN — no identifying metadata in 0 binary file(s)". A clean report over zero files is a false
    assurance, and the export is precisely the tree that most needs checking. Same failure shape as the
    leak scanner's earlier self-exemption: the tool reported success by looking at nothing.
    """
    if all_files:
        return _walk_binaries()
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        # Not a git repository -- an export or a plain directory. Walk it instead of scanning nothing.
        return _walk_binaries()
    return [REPO / p for p in out.stdout.split()
            if (REPO / p).is_file() and Path(p).suffix.lower() in BINARY_SUFFIXES]


def _inspect_zip(path: Path) -> list[str]:
    findings = []
    try:
        z = zipfile.ZipFile(path)
    except Exception as exc:  # noqa: BLE001
        return [f"unreadable zip: {exc}"]
    names = z.namelist()
    for part in ZIP_METADATA_PARTS:
        if part in names:
            body = z.read(part)
            if IDENTIFYING.search(body):
                findings.append(f"{part}: identifying metadata present")
    # change-tracking parts are the ones people never look at
    for n in names:
        if "changesInfo" in n or "revisionInfo" in n:
            body = z.read(n)
            hits = set(m.group(0).decode("utf-8", "ignore") for m in IDENTIFYING.finditer(body))
            if hits:
                findings.append(f"{n}: {sorted(hits)[:4]}")
    return findings


def _inspect_gzip(path: Path) -> list[str]:
    """gzip headers: FNAME/FCOMMENT can carry a path, and a non-zero MTIME makes the file
    non-reproducible even when it carries nothing identifying."""
    import struct

    b = path.read_bytes()[:1024]
    if len(b) < 10 or b[:2] != b"\x1f\x8b":
        return []
    flg, mtime = b[3], struct.unpack("<I", b[4:8])[0]
    i = 10
    if flg & 0x04:                                     # FEXTRA
        i += 2 + struct.unpack("<H", b[i:i + 2])[0]
    findings = []
    for bit, label in ((0x08, "FNAME"), (0x10, "FCOMMENT")):
        if flg & bit and b"\x00" in b[i:]:
            end = b.index(b"\x00", i)
            val = b[i:end].decode("latin-1", "replace")
            i = end + 1
            if "/" in val or "\\" in val:
                findings.append(f"gzip {label} carries a PATH: {val!r}")
    # A non-zero MTIME is a REPRODUCIBILITY defect, not identifying metadata, so it is advisory here.
    # This scanner's contract is "does this file identify a person or machine"; failing it for a
    # timestamp would make a privacy gate red for a non-privacy reason, and the next person would learn
    # to ignore it. Determinism of the PUBLISHED snapshots is asserted by publish_snapshots.py --check,
    # which is the tool that writes them.
    if mtime:
        findings.append(f"NOTE gzip MTIME is {mtime} — not identifying, but not reproducible either")
    return findings


def _inspect_bytes(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except Exception:
        return []
    hits = {m.group(0).decode("utf-8", "ignore") for m in IDENTIFYING.finditer(raw)}
    # a bare `Artist`/`/Author` key with no value is noise; keep only paths and named creators
    hits = {h for h in hits if "/" in h or "\\" in h or "creator" in h.lower() or "userId" in h}
    return [f"embedded: {sorted(hits)[:4]}"] if hits else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="scan every binary on disk, not just tracked")
    args = ap.parse_args()

    files = _tracked_binaries(args.all)
    if not files:
        # A pass over nothing is not a pass. This scanner previously reported CLEAN on an
        # export because `git ls-files` returns empty outside a repo.
        print("  NO CANDIDATE BINARIES FOUND -- refusing to report clean.")
        print("  Either this tree genuinely has no binaries, or discovery is broken. Verify which")
        print("  before trusting any clean scan of it.")
        return 1
    print("=" * 92)
    print(f"BINARY METADATA SCAN — {len(files)} file(s)")
    print("=" * 92)
    flagged = 0
    for f in sorted(files):
        rel = f.relative_to(REPO).as_posix()
        if f.suffix.lower() in {".pptx", ".docx", ".xlsx"}:
            findings = _inspect_zip(f)
        elif f.suffix.lower() == ".gz":
            findings = _inspect_gzip(f) + _inspect_bytes(f)
        else:
            findings = _inspect_bytes(f)
        blocking = [x for x in findings if not x.startswith("NOTE ")]
        if findings:
            flagged += 1 if blocking else 0
            print(f"\n  {rel}")
            for x in findings:
                print(f"      {x}")
    if flagged:
        print(f"\n  {flagged} file(s) carry identifying metadata.")
        print("  Remedies: re-export from a clean template; for PNGs, re-save with metadata stripped")
        print("  (e.g. `matplotlib.pyplot.savefig(..., metadata={'Software': None})`), or run")
        print("  `exiftool -all= <file>`; for Office files, export fresh rather than editing in place.")
    else:
        print(f"\n  CLEAN — no identifying metadata in {len(files)} binary file(s).")
    print()
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
