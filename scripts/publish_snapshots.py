"""Publish the frozen raw ChEMBL snapshots so the frozen metrics can be recomputed from the same bytes.

The problem this solves
----------------------
Hashes prove identity; they cannot reconstruct data. A repository that records
``BRD4_raw.csv sha256=…`` lets a reader confirm they hold the same file — and gives them nothing if they
do not. Every claim then rests on trust, which is precisely what a provenance record is supposed to
replace.

ChEMBL is queried live and grows, so a cache-free run reproduces the *workflow* but not the *numbers*:
the numbers belong to one moment. The only way to make the frozen metrics independently checkable is to
publish the bytes they were computed from.

Why that is practical here
--------------------------
The raw snapshots are 79.9 MB uncompressed, which would be an unreasonable thing to commit. But
``assay_description`` repeats the same long strings across thousands of rows, so gzip takes the whole
set to **2.7 MB** — small enough that there is no excuse for not shipping it.

Licensing
---------
ChEMBL is distributed by EMBL-EBI under **CC BY-SA 3.0**, which permits redistribution with attribution
under the same licence. The published directory therefore carries its own licence and attribution file,
separate from the repository's code licence, and the share-alike term is stated rather than assumed.

How the snapshot is consumed
----------------------------
``medchem.data.pull`` restores from it when ``MEDCHEM_FROZEN_SNAPSHOT`` points at the directory,
verifying every checksum and contacting no network. Fail-closed: a missing file or a bad checksum
raises rather than falling back to a live fetch, because a live fetch that looks like a frozen one is
the exact failure this exists to prevent.

Usage:
    python scripts/publish_snapshots.py             # write data/frozen_snapshots/
    python scripts/publish_snapshots.py --check     # verify the published set against the run tree
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "frozen_snapshots"
PANELS = ("jak1", "brd4")          # headline and sensitivity share one snapshot per target

ATTRIBUTION = r"""\
# Frozen ChEMBL snapshots

These files are the exact bioactivity records the frozen results were computed from. They are published
so the metrics in this repository can be **recomputed from the same bytes**, rather than re-derived
against a moving database. Agreement is expected within the numerical tolerance the README documents,
not to the last bit: float64 rounding varies with BLAS kernel, thread count and CPU.

## Source and licence

Data source: **ChEMBL**, European Molecular Biology Laboratory — European Bioinformatics Institute
(EMBL-EBI), <https://www.ebi.ac.uk/chembl/>. Release **34 as requested** in the run configs. The REST
query cannot pin a release and the served release was never confirmed, so what is pinned is these
**bytes**, by the SHA-256 values in `SHA256SUMS`.

ChEMBL data is made available under the **Creative Commons Attribution-ShareAlike 3.0 Unported licence**
(CC BY-SA 3.0): <https://creativecommons.org/licenses/by-sa/3.0/>. Redistribution is permitted with
attribution under the same terms, so **these data files are distributed under CC BY-SA 3.0**, not under
the repository's code licence. Anyone redistributing them further must preserve this notice and the
share-alike term.

The files are unmodified subsets: the records ChEMBL returned for the configured targets and activity
types, with a fixed column selection. No values were altered, imputed, or filtered — all curation
happens downstream in the pipeline, from these bytes.

## Contents

One gzipped CSV per target, plus assay metadata per target. Cohort selection depends on assay
*descriptions*, so a snapshot of activities alone would not be sufficient to reproduce a cohort.

`SHA256SUMS` lists the checksum of each file **after decompression** — that is what the pipeline
verifies and what the run manifests record. Only the `.csv.gz` files ship, so a plain
`sha256sum -c SHA256SUMS` will not find them. Verify like this instead:

```bash
cd data/frozen_snapshots
while read -r want name; do
  case "$want" in \#*) continue ;; esac
  got=$(gzip -dc "$name.gz" | shasum -a 256 | cut -d' ' -f1)
  [ "$got" = "$want" ] && echo "OK   $name" || echo "FAIL $name"
done < SHA256SUMS
```

## Reproducing the frozen metrics

```bash
MEDCHEM_FROZEN_SNAPSHOT=data/frozen_snapshots \
  uv run medchem run -p discovery -c configs/brd4.yaml \
  --stage data_pull --stage curate --stage featurize \
  --stage qsar --stage selectivity --stage evaluate
```

The stage list matters: it restricts the run to the stages that produce the metrics, all of which are
offline once the snapshot is restored. The full graph also runs `receptor`, which **fetches a structure
from the RCSB PDB** — so "no network" is true of the stages above and false of the whole graph.

Without the environment variable the pipeline pulls live from ChEMBL, which reproduces the workflow
against current data and will *not* match the frozen numbers.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(panel: str) -> dict | None:
    """This panel's run manifest, preferring the TRACKED copy.

    ``--check`` has to work where ``runs/`` does not exist -- the published export and CI -- and the
    tracked ``provenance/`` copy carries the same ``raw_inputs`` block. Reading only ``runs/`` made the
    check crash with a traceback in exactly the tree that most needs verifying.
    """
    for base in (REPO / "provenance" / panel, REPO / "runs" / panel):
        f = base / "run_manifest.json"
        if f.is_file():
            return json.loads(f.read_text())
    return None


def _sources(*, require_files: bool) -> dict[str, tuple[Path | None, str]]:
    """Map filename -> (source path or None, expected sha256), from the run manifests.

    ``require_files`` is False when only checking published copies: the export verifies the gzipped
    files against the manifests' checksums without needing the original uncompressed inputs, which are
    part of the gitignored run tree and legitimately absent.
    """
    out: dict[str, tuple[Path | None, str]] = {}
    for panel in PANELS:
        m = _manifest(panel)
        if m is None:
            raise SystemExit(
                f"{panel}: no run_manifest.json in provenance/ or runs/. Refusing to report on "
                f"snapshots whose expected checksums cannot be established."
            )
        for name, info in m["raw_inputs"].items():
            p = REPO / info["path"]
            if not p.is_file():
                if require_files:
                    raise SystemExit(f"{name}: {p} is missing; cannot publish an incomplete snapshot")
                p = None
            prev = out.get(name)
            if prev and prev[1] != info["sha256"]:
                raise SystemExit(
                    f"{name}: two panels disagree on its checksum — the snapshots are not shared as "
                    f"the manifests claim, and publishing either would be misleading"
                )
            out[name] = (p, info["sha256"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify published files, write nothing")
    args = ap.parse_args()

    src = _sources(require_files=not args.check)
    problems: list[str] = []
    lines: list[str] = []
    raw_total = gz_total = 0

    print("=" * 92)
    print(f"FROZEN SNAPSHOTS — {'CHECK' if args.check else 'PUBLISH'} ({len(src)} files)")
    print("=" * 92)

    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)

    for name in sorted(src):
        path, expect = src[name]
        if path is not None:
            got = _sha256(path)
            if got != expect:
                problems.append(f"{name}: source no longer matches the manifest checksum")
                continue
            raw_total += path.stat().st_size
        gz = OUT / f"{name}.gz"
        if args.check:
            if not gz.exists():
                problems.append(f"{name}.gz not published")
                continue
            # Determinism is part of the contract for a PUBLISHED archive: a non-zero embedded
            # mtime means the bytes change on every republish, so the archive cannot be checked
            # byte-for-byte and every commit touching it churns all 16 files.
            head = gz.read_bytes()[:8]
            if len(head) >= 8 and head[4:8] != b"\x00\x00\x00\x00":
                problems.append(f"{name}.gz embeds a non-zero mtime — republish for determinism")
            with gzip.open(gz, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != expect:
                    problems.append(f"{name}.gz decompresses to the wrong content")
        else:
            assert path is not None      # require_files=True on the publish path
            # DETERMINISTIC gzip: mtime=0 and an empty embedded filename. Python's default writes the
            # current time and the output basename into the header, so the bytes differed on every
            # publish -- which would show all 16 files as modified in every commit that touched them,
            # and would mean the published archive could not be checked byte-for-byte. Nothing here
            # leaked (the basename is not a path, and OS is written as 255/unknown), but a
            # non-reproducible artifact undermines the point of publishing it.
            with path.open("rb") as f, gz.open("wb") as raw, \
                    gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as g:
                shutil.copyfileobj(f, g)
        if gz.exists():
            gz_total += gz.stat().st_size
        lines.append(f"{expect}  {name}")

    if not args.check:
        (OUT / "SHA256SUMS").write_text(
            "# sha256 of each file AFTER decompression -- what the pipeline verifies on restore.\n"
            "# Generated by scripts/publish_snapshots.py from the run manifests.\n"
            + "\n".join(sorted(lines)) + "\n"
        )
        (OUT / "README.md").write_text(ATTRIBUTION)
        print(f"  {len(lines)} file(s) -> {OUT.relative_to(REPO)}")
        print(f"  {raw_total / 1e6:.1f} MB uncompressed  ->  {gz_total / 1e6:.1f} MB gzipped "
              f"({100 * gz_total / raw_total:.0f}%)")
        print("  wrote SHA256SUMS and README.md (CC BY-SA 3.0 attribution)")

    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"      - {p}")
        return 1
    print(f"\n  clean — {len(lines)} snapshot(s) {'verified' if args.check else 'published'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
