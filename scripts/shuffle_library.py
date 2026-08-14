"""Seeded shuffle of a pulled library so that ANY PREFIX is a uniform random sample.

ZINC22 `.smi` files are written in **tranche order** (heavy-atom count x logP), so the first
N lines are the first few tranches — biased in molecular weight and lipophilicity, not a
sample of the library. Docking a head-slice and then reporting per-stratum enrichment would
confound the strata with tranche.

One seeded shuffle fixes it permanently: after this, `read_smiles(path, start=0, n=200_000)`
is a genuine 200k random sample, contiguous sharding stays trivial, and the whole thing is
reproducible from (source manifest + seed). The original file is left untouched — it is what
the sha256 manifest pins.

Usage:
  uv run python scripts/shuffle_library.py <library>.smi.gz \\
      --out <library>_shuffled.smi.gz --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np

    opener = gzip.open if args.source.suffix == ".gz" else open
    with opener(args.source, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    print(f"read {len(lines):,} lines from {args.source}")

    order = np.random.default_rng(args.seed).permutation(len(lines))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for i in order:
            fh.write(lines[int(i)] + "\n")

    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    print(f"  seed   : {args.seed}")
    print(f"  sha256 : {sha}")
    print("\nfirst 3 lines after shuffle (should span different tranches):")
    with gzip.open(args.out, "rt", encoding="utf-8") as fh:
        for _, ln in zip(range(3), fh, strict=False):
            print("  " + ln.strip()[:78])


if __name__ == "__main__":
    main()
