"""The frozen-snapshot restore path must be fail-closed.

This mode exists so the published metrics can be recomputed from the same bytes rather than
re-derived against a moving database. Its
whole value depends on one property: if the archived bytes are not exactly the archived bytes, it must
STOP. A restore that silently degraded to a live fetch, or accepted an unverified file, would produce
numbers that look like the frozen ones and are not — which is the failure this repository spent most of
its effort learning to detect.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from medchem.data.pull import FROZEN_SNAPSHOT_ENV, _restore_from_snapshot

CONTENT = b"assay_chembl_id,description\nCHEMBL1,Inhibition of BRD4 BD1 by TR-FRET assay\n"


def _snapshot(tmp: Path, *, content: bytes = CONTENT, sums: bytes | None = None,
              gzipped: bool = True) -> Path:
    d = tmp / "snap"
    d.mkdir()
    for name in ("T_raw.csv", "T_assays.csv"):
        if gzipped:
            with gzip.open(d / f"{name}.gz", "wb") as g:
                g.write(content)
        else:
            (d / name).write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    (d / "SHA256SUMS").write_bytes(
        sums if sums is not None
        else f"{digest}  T_raw.csv\n{digest}  T_assays.csv\n".encode()
    )
    return d


def test_restores_bytes_exactly(tmp_path: Path):
    d = _snapshot(tmp_path)
    out, assay_out = tmp_path / "T_raw.csv", tmp_path / "T_assays.csv"
    _restore_from_snapshot(d, "T", out, assay_out)
    assert out.read_bytes() == CONTENT
    assert assay_out.read_bytes() == CONTENT


def test_accepts_uncompressed_snapshot(tmp_path: Path):
    """gzip is a size optimisation, not part of the contract."""
    d = _snapshot(tmp_path, gzipped=False)
    _restore_from_snapshot(d, "T", tmp_path / "a.csv", tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_bytes() == CONTENT


def test_checksum_mismatch_raises_and_removes_the_bad_file(tmp_path: Path):
    """A wrong checksum must not leave a plausible-looking file behind for a later stage to consume."""
    d = _snapshot(tmp_path, sums=b"%s  T_raw.csv\n%s  T_assays.csv\n" % (b"0" * 64, b"0" * 64))
    out = tmp_path / "T_raw.csv"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _restore_from_snapshot(d, "T", out, tmp_path / "T_assays.csv")
    assert not out.exists()


def test_missing_sha256sums_raises(tmp_path: Path):
    d = _snapshot(tmp_path)
    (d / "SHA256SUMS").unlink()
    with pytest.raises(RuntimeError, match="no SHA256SUMS"):
        _restore_from_snapshot(d, "T", tmp_path / "a.csv", tmp_path / "b.csv")


def test_unlisted_file_raises(tmp_path: Path):
    """Present in the directory but absent from the manifest is unverifiable, not acceptable."""
    d = _snapshot(tmp_path, sums=b"deadbeef  something_else.csv\n")
    with pytest.raises(RuntimeError, match="does not list"):
        _restore_from_snapshot(d, "T", tmp_path / "a.csv", tmp_path / "b.csv")


def test_missing_target_file_raises(tmp_path: Path):
    d = _snapshot(tmp_path)
    (d / "T_assays.csv.gz").unlink()
    with pytest.raises(RuntimeError, match="missing T_assays.csv"):
        _restore_from_snapshot(d, "T", tmp_path / "a.csv", tmp_path / "b.csv")


def test_env_var_name_is_stable():
    """Documented in the snapshot README and the public README; renaming it silently breaks both."""
    assert FROZEN_SNAPSHOT_ENV == "MEDCHEM_FROZEN_SNAPSHOT"
