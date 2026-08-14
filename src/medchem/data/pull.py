"""ChEMBL data-pull stage.

Live network stage: fetches bioactivities for the target panel from the public
ChEMBL REST API (via ``chembl_webresource_client``), writes one raw CSV per
target plus a provenance manifest into the stage's key-scoped workdir. The
network is only touched here — CI does not run this stage; the curation logic it
feeds is tested against an in-memory fixture in ``tests/test_curate.py``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from medchem.pipeline.stage import StageContext, StageResult, stage

# Fields requested from the ChEMBL `activity` resource (kept minimal for speed).
_ACTIVITY_FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "pchembl_value",
    "assay_type",
    # Construct/domain identity. Without these a multi-domain target collapses into one label: BRD4
    # IC50 records are 47.7% BD1-explicit, 25.0% biochemical with no domain stated, 15.8% cell-based
    # and 6.3% BD2-explicit, and the domains disagree -- apabetalone reads 5.85 on BD1 against 6.88 on
    # BD2, so a median across them represents neither. Dropping these fields made that mixture
    # invisible. (Figures derived by scripts/derive_composition.py, not asserted.)
    "assay_chembl_id",
    "assay_description",
    "bao_format",
    "target_chembl_id",
    "activity_comment",
    "data_validity_comment",
    "document_year",  # enables the temporal-split evaluation later
]


REST_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def _fetch_paginated(
    resource: str, params: dict, *, page_size: int = 1000, timeout: int = 60
) -> tuple[list[dict], int]:
    """Fetch every page of a ChEMBL REST resource, with an EXPLICIT per-request timeout.

    This replaces ``chembl_webresource_client`` for bulk retrieval, for a measured reason: a pull
    through the client hung for **two days and seventeen hours** on a single dead socket while the
    service itself answered in under a second. Setting ``Settings.TIMEOUT`` did not help — the request
    was still issued with ``read timeout=None``, so there was nothing to interrupt the wait.

    Direct pagination is both faster and interruptible: measured at ~1000 rows per 5 seconds, so a
    15,000-row target takes about 75 seconds rather than never finishing. Each page carries its own
    timeout and retries, and ``page_meta.total_count`` gives the completeness check its expected value
    from the same response that delivers the data.
    """
    import json as _json
    import urllib.error
    import urllib.parse
    import urllib.request

    rows: list[dict] = []
    total = -1
    offset = 0
    # The response key is NOT a naive pluralisation: "activity" -> "activities", not "activitys". An
    # earlier version guessed and silently fetched zero rows for every activity page -- caught only by
    # the completeness check comparing 0 against the 356 ChEMBL reported. So the key is discovered from
    # the payload: whichever field holds the list of records.
    key: str | None = None
    while True:
        q = {**params, "limit": page_size, "offset": offset}
        url = f"{REST_BASE}/{resource}.json?" + urllib.parse.urlencode(q)
        last: Exception | None = None
        payload: dict = {}
        for attempt in range(1, 5):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                    payload = _json.load(resp)
                last = None
                break
            except Exception as exc:  # noqa: BLE001 - transient service failure, retry with backoff
                last = exc
                time.sleep(3 * attempt)
        if last is not None:
            raise RuntimeError(
                f"{resource} page at offset {offset} failed after 4 attempts: "
                f"{type(last).__name__}: {last}"
            ) from last
        if key is None:
            key = next((k for k, v in payload.items() if k != "page_meta" and isinstance(v, list)), None)
            if key is None:
                raise RuntimeError(
                    f"{resource}: response carries no list of records (keys: {sorted(payload)}). "
                    f"Refusing to treat that as an empty result set."
                )
        batch = payload.get(key) or []
        total = payload.get("page_meta", {}).get("total_count", total)
        rows.extend(batch)
        offset += page_size
        if not batch or (total >= 0 and len(rows) >= total):
            break
    return rows, total


def _pull_target(target_chembl_id: str, activity_types: list[str]) -> tuple[pd.DataFrame, int]:
    """Fetch activity rows for one target. Returns ``(frame, count ChEMBL reported)``."""
    params: dict = {"target_chembl_id": target_chembl_id}
    if len(activity_types) == 1:
        params["standard_type"] = activity_types[0]
    rows, total = _fetch_paginated("activity", params)
    df = pd.DataFrame.from_records(rows)
    keep = [c for c in _ACTIVITY_FIELDS if c in df.columns]
    if keep:
        df = df[keep]
    if len(activity_types) > 1 and "standard_type" in df.columns:
        wanted = {t.lower() for t in activity_types}
        df = df[df["standard_type"].astype(str).str.lower().isin(wanted)]
    return df, total


def _pull_assays(target_chembl_id: str) -> pd.DataFrame:
    """Fetch the target's ASSAY records, which carry the description activities do not.

    ``assay_description__icontains`` is NOT a supported filter on the activity endpoint — it returns
    HTTP 400 — and an early attempt of mine caught that exception and reported "0 domain-explicit
    assays" for every target, a false zero that nearly became a finding. The supported route is to
    fetch assays and join on ``assay_chembl_id``.
    """
    rows, _total = _fetch_paginated("assay", {"target_chembl_id": target_chembl_id})
    df = pd.DataFrame.from_records(rows)
    keep = [c for c in ("assay_chembl_id", "description", "assay_type", "bao_format",
                        "confidence_score", "assay_organism", "relationship_type")
            if c in df.columns]
    return df[keep] if keep else df


def _expected_count(target_chembl_id: str, activity_types: list[str]) -> int | None:
    """Ask ChEMBL how many activity rows exist, without downloading them.

    Used only to decide whether an already-present CSV is complete enough to resume from. The pull
    itself takes its expected count from the same response that delivers the data, which is stronger:
    a count fetched separately could describe a different moment than the rows.
    """
    import json as _json
    import urllib.parse
    import urllib.request

    params: dict = {"target_chembl_id": target_chembl_id, "limit": 1}
    if len(activity_types) == 1:
        params["standard_type"] = activity_types[0]
    url = f"{REST_BASE}/activity.json?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            return int(_json.load(resp)["page_meta"]["total_count"])
    except Exception:  # noqa: BLE001 - unavailable count weakens the resume check, never fails the pull
        return None


FROZEN_SNAPSHOT_ENV = "MEDCHEM_FROZEN_SNAPSHOT"


def _restore_from_snapshot(snapshot_dir: Path, name: str, out: Path, assay_out: Path) -> dict:
    """Materialise one target's inputs from a published frozen snapshot, verifying checksums.

    This is the mechanism that makes *exact* reproduction possible rather than merely described.
    ChEMBL is queried live and grows, so a cache-free run reproduces the WORKFLOW but not the frozen
    numbers; the numbers belong to one snapshot. Publishing that snapshot is only useful if the
    pipeline can consume it, so:

        MEDCHEM_FROZEN_SNAPSHOT=data/frozen_snapshots uv run medchem run -p discovery -c configs/brd4.yaml

    reads the archived bytes and never contacts the API. Deliberately fail-closed: a missing file or a
    checksum mismatch raises. Silently falling back to a live fetch would produce numbers that look
    like the frozen ones and are not, which is the failure mode this whole exercise exists to prevent.

    Set by environment rather than by config on purpose — a config field would change the stage's cache
    key and therefore the identity of every already-executed run.
    """
    import gzip
    import shutil

    sums_file = snapshot_dir / "SHA256SUMS"
    if not sums_file.exists():
        raise RuntimeError(
            f"{FROZEN_SNAPSHOT_ENV}={snapshot_dir} has no SHA256SUMS; refusing to restore unverified "
            f"inputs. An unverified snapshot is worse than no snapshot: it cannot be distinguished "
            f"from a corrupted or substituted one."
        )
    want = {}
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, fname = line.partition("  ")
        want[fname.strip()] = digest.strip()

    restored = {}
    for src_name, dest in ((f"{name}_raw.csv", out), (f"{name}_assays.csv", assay_out)):
        gz = snapshot_dir / f"{src_name}.gz"
        plain = snapshot_dir / src_name
        if gz.exists():
            with gzip.open(gz, "rb") as f, dest.open("wb") as g:
                shutil.copyfileobj(f, g)
        elif plain.exists():
            shutil.copy2(plain, dest)
        else:
            raise RuntimeError(
                f"frozen snapshot {snapshot_dir} is missing {src_name}[.gz]; a partial snapshot cannot "
                f"reproduce the frozen metrics and must not be used as if it could"
            )
        got = _sha256_file(dest)
        expect = want.get(src_name)
        if expect is None:
            raise RuntimeError(f"SHA256SUMS does not list {src_name}; refusing an unverifiable input")
        if got != expect:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"{src_name}: checksum mismatch after restore (got {got[:12]}…, expected "
                f"{expect[:12]}…). The archived snapshot does not match its manifest."
            )
        restored[src_name] = got
    return restored


def _csv_rows(path: Path) -> int:
    """Data-row count of an existing CSV, header excluded."""
    import csv as _csv

    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return max(0, sum(1 for _ in _csv.reader(fh)) - 1)


def _target_pref_name(target_chembl_id: str) -> str:
    """Resolve a target's preferred name for a provenance sanity-check."""
    from chembl_webresource_client.new_client import new_client

    try:
        rec = new_client.target.get(target_chembl_id)  # pyright: ignore[reportAttributeAccessIssue]
        return str(rec.get("pref_name", "")) if isinstance(rec, dict) else ""
    except Exception:
        return ""


def _sha256_file(path: Path) -> str:
    """Streaming sha256 of a pulled file. A raw input without a checksum cannot be shown to be the
    same input next time, which is the whole basis of a reproducible rerun."""
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _served_release() -> str:
    """The ChEMBL release the live API is currently serving (best-effort)."""
    from chembl_webresource_client.new_client import new_client

    try:
        st = new_client.status.get()  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(st, dict):
            return str(st.get("chembl_db_version") or st.get("chembl_release") or "")
    except Exception:
        pass
    return ""


@stage("discovery", "data_pull", config_keys=("data",))
def data_pull(ctx: StageContext) -> StageResult:
    """Pull bioactivity for the JAK isoform panel from a public ChEMBL release."""
    data = ctx.config.data
    targets: dict[str, str] = dict(data.targets)
    activity_types = list(data.activity_types)
    workdir = Path(ctx.workdir)

    outputs: dict[str, str] = {}
    counts: dict[str, int] = {}
    provenance: dict = {
        "source": "ChEMBL REST API (chembl_webresource_client)",
        "configured_release": data.chembl_release,
        "activity_types": activity_types,
        "pulled_at_utc": datetime.now(UTC).isoformat(),
        "targets": {},
    }

    # Frozen-snapshot mode short-circuits every network call, so it is resolved before the release
    # probe below (which would otherwise contact the API in an offline reproduction).
    snapshot_env = os.environ.get(FROZEN_SNAPSHOT_ENV, "").strip()
    snapshot_dir = Path(snapshot_env) if snapshot_env else None
    if snapshot_dir is not None:
        if not snapshot_dir.is_absolute():
            snapshot_dir = Path.cwd() / snapshot_dir
        if not snapshot_dir.is_dir():
            raise RuntimeError(
                f"{FROZEN_SNAPSHOT_ENV}={snapshot_env!r} is not a directory. Unset it to pull live, or "
                f"point it at a published snapshot; guessing between the two is how a live pull gets "
                f"mistaken for a frozen one."
            )
        provenance["source"] = f"FROZEN SNAPSHOT restored from {snapshot_dir}"
        provenance["frozen_snapshot"] = {
            "directory": str(snapshot_dir),
            "network_used": False,
            "note": ("exact-reproduction mode: archived bytes, checksum-verified, no API contact. "
                     "Metrics from this mode are comparable to the published frozen results; metrics "
                     "from a live pull are not, because ChEMBL grows."),
        }

    # Verify the live API is actually serving the configured release (it is NOT pinned
    # by the query — record what was served and flag a mismatch rather than trust a label).
    served = None if snapshot_dir is not None else _served_release()
    configured = str(data.chembl_release)
    provenance["served_release"] = served
    provenance["release_match"] = bool(served) and configured in served
    if served and configured not in served:
        import warnings

        warnings.warn(
            f"ChEMBL is serving release {served!r}, not the configured {configured!r}; "
            "recorded in provenance (numbers may differ from the pinned release).",
            stacklevel=2,
        )

    provenance["retrieval"] = {
        "method": "direct paginated REST with an explicit per-request timeout",
        "why": ("the chembl_webresource_client hung for 2d17h on one dead socket while the service "
                "answered in under a second; Settings.TIMEOUT did not reach the request"),
        "page_size": 1000,
        "per_request_timeout_s": 60,
    }

    for name, cid in targets.items():
        out = workdir / f"{name}_raw.csv"
        assay_out = workdir / f"{name}_assays.csv"
        expected = None  # filled from the fetch itself, which is the same response as the data

        # RESUMABLE per target. The workdir is key-scoped, so a re-run after a mid-pull network failure
        # lands here again -- re-fetching targets that already arrived complete wastes a scarce, flaky
        # resource and risks failing on a target that had already succeeded. A cached file is only
        # trusted when it matches ChEMBL's own count; anything else is re-fetched.
        resumed = False
        from_snapshot = False
        if snapshot_dir is not None:
            _restore_from_snapshot(snapshot_dir, name, out, assay_out)
            counts[name] = _csv_rows(out)
            expected = counts[name]
            from_snapshot = True
        elif out.exists() and assay_out.exists():
            probe = _expected_count(cid, list(activity_types))
            if probe is not None and _csv_rows(out) == probe:
                counts[name] = probe
                expected = probe
                resumed = True

        if not (resumed or from_snapshot):
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    df, expected = _pull_target(cid, activity_types)
                    if expected >= 0 and len(df) != expected:
                        raise RuntimeError(
                            f"fetched {len(df)} activity rows but ChEMBL reports {expected}; a "
                            f"truncated snapshot produces a complete-looking run on partial data"
                        )
                    df.to_csv(out, index=False)
                    counts[name] = len(df)
                    # Assay metadata in the SAME attempt, so descriptions and activities are one
                    # snapshot. Fetching them later risks a mismatched pair.
                    _pull_assays(cid).to_csv(assay_out, index=False)
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001 - retry transient service failures
                    last_error = exc
                    time.sleep(15 * attempt)
            if last_error is not None:
                raise RuntimeError(
                    f"{name}: pull failed after 3 attempts -- {type(last_error).__name__}: "
                    f"{last_error}. Re-run to resume; completed targets are not re-fetched, and "
                    f"partial fetches are never stitched together."
                ) from last_error

        outputs[name] = str(out)
        outputs[f"{name}_assays"] = str(assay_out)
        assays = pd.read_csv(assay_out, low_memory=False)

        provenance["targets"][name] = {
            "target_chembl_id": cid,
            "pref_name": None if snapshot_dir is not None else _target_pref_name(cid),
            "n_rows": counts.get(name),
            "n_assays": int(len(assays)),
            "expected_activity_count": expected,
            "completeness_verified": expected is not None and counts.get(name) == expected,
            "resumed_from_cache": resumed,
            "restored_from_frozen_snapshot": from_snapshot,
            # Raw-input hashes, so a rerun can prove it consumed the same bytes rather than asserting it
            "activities_sha256": _sha256_file(out),
            "assays_sha256": _sha256_file(assay_out),
        }

    manifest = workdir / "pull_provenance.json"
    manifest.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    outputs["provenance"] = str(manifest)

    return StageResult(name="data_pull", outputs=outputs, metrics={"rows": counts})
