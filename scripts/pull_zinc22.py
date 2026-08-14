"""Pull a tractable, reproducible in-stock lead-like subset of ZINC22 (generation g).

``files.docking.org/zinc22/zinc-22g`` is the on-the-shelf ("ZINC20 in stock"), no-auth,
purchasable subset. It is organized by heavy-atom count (H04..H29) then fine logP tranche
(``H{HAC}{P|M}{logP*100}``); leaf shards are gzipped ``SMILES<TAB>ZINC_ID``. We pull a
focused HAC x logP window, concatenate to one 2-column SMILES file, and write a sha256
manifest — the reproducibility pin, since the tree carries no immutable version tags.

Usage:
  uv run python scripts/pull_zinc22.py --hac 24 25 26 --logp-min 200 --logp-max 350 \\
      --out data/vls --tag zinc22_instock_leadlike
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://files.docking.org/zinc22/zinc-22g"
UA = {"User-Agent": "medchem-vls/0.1 (open-source kinase VLS demo)"}


def _get(url: str, timeout: int = 90, retries: int = 4) -> bytes:
    """GET with retries — files.docking.org drops SSL connections under concurrency."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise  # 404 etc. are terminal — don't retry
        except Exception as exc:  # noqa: BLE001 — SSL EOF / timeout / conn reset: back off + retry
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise last if last else RuntimeError(f"failed: {url}")


def list_shards(hac: int, logp: int) -> list[str]:
    """Return leaf shard URLs for one HxxPyyy tranche (empty list if the dir 404s)."""
    tranche = f"H{hac:02d}P{logp:03d}"
    url = f"{BASE}/H{hac:02d}/{tranche}/"
    try:
        html = _get(url, timeout=45).decode("utf-8", "ignore")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return []
    names = sorted(set(re.findall(rf"{tranche}-[A-Za-z0-9]+\.g\.smi\.gz", html)))
    return [f"{url}{n}" for n in names]


def fetch_shard(url: str) -> tuple[str, bytes, list[str]]:
    """Download one shard; return (url, raw_gz_bytes, decoded SMILES-TSV lines)."""
    raw = _get(url)
    text = gzip.decompress(raw).decode("utf-8", "ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return url, raw, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hac", type=int, nargs="+", default=[24, 25, 26])
    ap.add_argument("--logp-min", type=int, default=200)  # logP*100 (P dirs = positive logP)
    ap.add_argument("--logp-max", type=int, default=350)
    ap.add_argument("--logp-step", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("data/vls"))
    ap.add_argument("--tag", default="zinc22_instock_leadlike")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tranches = [
        (h, p) for h in args.hac
        for p in range(args.logp_min, args.logp_max + 1, args.logp_step)
    ]
    print(f"discovering shards across {len(tranches)} tranches (HAC={args.hac}, "
          f"logP {args.logp_min/100:.1f}-{args.logp_max/100:.1f}) ...", flush=True)

    shard_urls: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for shards in ex.map(lambda hp: list_shards(*hp), tranches):
            shard_urls.extend(shards)
    print(f"  {len(shard_urls)} non-empty shards found", flush=True)

    smi_path = args.out / f"{args.tag}.smi"
    manifest = args.out / f"{args.tag}.MANIFEST.sha256"
    n_lines = 0
    seen_urls = 0
    with smi_path.open("w", encoding="utf-8") as fout, manifest.open("w", encoding="utf-8") as fman:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_shard, u): u for u in shard_urls}
            for fut in as_completed(futs):
                try:
                    url, raw, lines = fut.result()
                except Exception as exc:  # noqa: BLE001 — a dead shard shouldn't kill the pull
                    print(f"  ! skip {futs[fut]}: {exc}", file=sys.stderr)
                    continue
                for ln in lines:
                    fout.write(ln + "\n")
                n_lines += len(lines)
                sha = hashlib.sha256(raw).hexdigest()
                fman.write(f"{sha}  {len(raw)}  {url.rsplit('/', 1)[1]}  {url}\n")
                seen_urls += 1
                if seen_urls % 50 == 0:
                    print(f"  {seen_urls}/{len(shard_urls)} shards, {n_lines} molecules", flush=True)

    # THE MERGED DECK'S OWN CHECKSUM.
    #
    # The per-shard lines above pin the INPUTS: they say which upstream shards were fetched and what
    # bytes each contained. They do not pin the OUTPUT, and the output is what gets screened. Nothing
    # verifiable connected the two, so a merged deck could be truncated, re-sorted, appended to or
    # replaced with every shard hash still matching.
    #
    # `medchem.vls.stage` verifies the deck against this line before screening, so it is written last
    # and in the same `<sha256>  <size>  <name>` shape the shard lines use, with an explicit MERGED
    # marker in the URL column so a reader can see which line is the deck and which are inputs.
    deck_sha = hashlib.sha256()
    with smi_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            deck_sha.update(chunk)
    deck_hex = deck_sha.hexdigest()
    with manifest.open("a", encoding="utf-8") as fman:
        fman.write(f"{deck_hex}  {smi_path.stat().st_size}  {smi_path.name}  MERGED-DECK\n")

    print(f"\nDONE: {n_lines} molecules -> {smi_path}")
    print(f"pin:  {manifest}  ({seen_urls} shard(s) + the merged deck)")
    print(f"deck: sha256 {deck_hex}")


if __name__ == "__main__":
    main()
