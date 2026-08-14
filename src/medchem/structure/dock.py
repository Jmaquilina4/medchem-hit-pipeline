"""Shard-level docking: prep + dock a slice of a library, with reason-coded outcomes.

Runs identically on a laptop and in a container, so the local CPU baseline and the cluster
run are the same code path — which is what makes a GPU-vs-CPU score comparison meaningful.

**Failures are never treated as "does not bind."** A weak binder receives a weak *score*;
docking engines fail for technical reasons (embedding non-convergence, malformed input,
search timeout on a very flexible ligand). Every outcome is reason-coded and counted so
downstream hit-rate denominators stay honest (ADR 0005: no silent truncation).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# Vina prints a results table; mode 1 is the best-scoring pose.
_SCORE_RE = re.compile(r"^\s*1\s+(-?\d+\.\d+)", re.MULTILINE)

# Outcome codes. Only `ok` carries a score; the rest are technical failures, NOT verdicts.
STATUSES = ("ok", "prep_failed", "engine_error", "timeout", "no_score_parsed")


@dataclass
class ShardResult:
    records: list[dict] = field(default_factory=list)
    prep_seconds: float = 0.0
    dock_seconds: float = 0.0
    wall_seconds: float = 0.0

    @property
    # Part of the shard result contract, for callers that execute docking in batches. Batch execution
    # is optional and external to this repository, so a reader will find no in-tree caller — that is
    # expected rather than dead code.
    def status_counts(self) -> dict[str, int]:
        c = {s: 0 for s in STATUSES}
        for r in self.records:
            c[r["status"]] = c.get(r["status"], 0) + 1
        return c

    @property
    def scores(self) -> list[float]:
        return [r["score"] for r in self.records if r["status"] == "ok"]


def read_smiles(path: str | Path, *, start: int = 0, n: int | None = None) -> list[tuple[str, str]]:
    """Read ``ID<TAB>SMILES`` (or ``SMILES<TAB>ID``) lines, optionally a slice.

    Transparently handles gzip so a container can ship the compressed library.
    """
    import gzip

    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    out: list[tuple[str, str]] = []
    with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for i, line in enumerate(fh):
            if i < start:
                continue
            if n is not None and len(out) >= n:
                break
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            # Whichever field parses as a molecule is the SMILES; ZINC ids start with "ZINC".
            if a.upper().startswith("ZINC"):
                out.append((b, a))
            else:
                out.append((a, b))
    return out


def _dock_one(job: tuple) -> dict:
    """Prep + dock one ligand. Never raises; every failure is reason-coded."""
    smiles, cid, engine, receptor, center, size, exhaustiveness, seed, timeout = job
    from medchem.structure.prep import prepare_ligand

    rec = {"id": cid, "smiles": smiles, "score": None, "status": "ok",
           "prep_s": 0.0, "dock_s": 0.0}
    t0 = time.perf_counter()
    pdbqt = prepare_ligand(smiles, seed=seed)
    rec["prep_s"] = round(time.perf_counter() - t0, 4)
    if pdbqt is None:
        rec["status"] = "prep_failed"
        return rec

    with tempfile.TemporaryDirectory() as td:
        lig = Path(td) / "lig.pdbqt"
        lig.write_text(pdbqt, encoding="utf-8")
        cmd = [
            engine, "--receptor", str(receptor), "--ligand", str(lig),
            "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
            "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
            "--exhaustiveness", str(exhaustiveness), "--cpu", "1", "--seed", str(seed),
            "--out", str(Path(td) / "out.pdbqt"),
        ]
        t1 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            rec["dock_s"] = round(time.perf_counter() - t1, 4)
            rec["status"] = "timeout"          # search budget, NOT a binding verdict
            return rec
        rec["dock_s"] = round(time.perf_counter() - t1, 4)
        if proc.returncode != 0:
            rec["status"] = "engine_error"
            rec["detail"] = (proc.stderr or "")[-200:]
            return rec
        m = _SCORE_RE.search(proc.stdout)
        if not m:
            rec["status"] = "no_score_parsed"
            return rec
        rec["score"] = float(m.group(1))
    return rec


def dock_shard(
    ligands: list[tuple[str, str]],
    *,
    engine: str,
    receptor: str | Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int = 8,
    seed: int = 42,
    workers: int = 8,
    timeout: int = 900,
) -> ShardResult:
    """Dock a shard of ``(smiles, id)`` pairs. Parallel across processes, one core each.

    One core per engine invocation and N workers beats handing the engine N threads: Vina's
    internal threading scales poorly compared with independent ligands, and per-ligand
    isolation means a single pathological ligand cannot stall the shard.
    """
    jobs = [
        (smi, cid, str(engine), str(receptor), center, size, exhaustiveness, seed, timeout)
        for smi, cid in ligands
    ]
    t0 = time.perf_counter()
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(_dock_one, jobs, chunksize=1))
    else:
        recs = [_dock_one(j) for j in jobs]
    wall = time.perf_counter() - t0
    return ShardResult(
        records=recs,
        prep_seconds=round(sum(r["prep_s"] for r in recs), 3),
        dock_seconds=round(sum(r["dock_s"] for r in recs), 3),
        wall_seconds=round(wall, 3),
    )
