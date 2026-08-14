"""VLS Tier-0: library preparation.

Turn a raw purchasable library (``SMILES[<TAB>id]`` per line) into a clean, deduplicated
deck of drug-/lead-like compounds, emitting a reason-coded attrition count at every step
(no silent truncation — ADR 0005). Physchem bounds default to *extended lead-like* (MW
300–460, the fast-follower band around upadacitinib ~380 Da); PAINS/assay-interference
compounds are removed via RDKit's FilterCatalog.

Streams molecule-by-molecule (only canonical SMILES strings are retained, never a list of
RDKit mols) and parallelizes across processes, so a ~10⁶-compound library is tractable in
bounded memory.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LeadLikeBounds:
    mw: tuple[float, float] = (300.0, 460.0)
    logp_max: float = 3.5
    tpsa_max: float = 140.0
    hbd_max: int = 5
    hba_max: int = 10
    rotb_max: int = 10


@dataclass
class LibraryResult:
    records: list[dict]                       # kept: {smiles (canonical), id}
    attrition: list[tuple[str, int]]          # ordered (step, n_remaining)
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count

    @property
    def n_prepared(self) -> int:
        return len(self.records)


def load_library(path: str | Path, *, cap: int | None = None, seed: int = 42) -> list[tuple[str, str]]:
    """Read ``SMILES[<TAB>id]`` lines (whitespace-split; id auto-assigned if absent).

    If ``cap`` is set and the file has more rows, take a deterministic seeded random
    sample of ``cap`` rows. NOTE: a random cap is *not* a scientific selection — prefer
    Tier-0.5 focusing (``medchem.vls.focus``) to choose a dockable deck on purpose.
    """
    import numpy as np

    recs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.split()
            if not parts:
                continue
            recs.append((parts[0], parts[1] if len(parts) > 1 else f"row{i}"))
    if cap is not None and len(recs) > cap:
        idx = np.random.default_rng(seed).choice(len(recs), size=cap, replace=False)
        idx.sort()
        recs = [recs[i] for i in idx]
    return recs


def _prep_chunk(
    payload: tuple[list[tuple[str, str]], LeadLikeBounds, bool],
) -> tuple[list[tuple[str, str]], dict[str, int], int, int]:
    """Worker: parse → physchem → PAINS for one chunk.

    Returns (kept canonical (smiles, id), dropped counts, n_parseable, n_physchem_pass).
    Dedup is global, so it happens in the parent.
    """
    records, b, apply_pains = payload
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    # Silence per-molecule parse warnings on a big deck. rdkit stubs omit DisableLog; it is real.
    RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]
    catalog = None
    if apply_pains:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)

    dropped = {"unparseable": 0, "physchem": 0, "pains": 0}
    n_parseable = 0
    n_physchem = 0
    kept: list[tuple[str, str]] = []
    for smi, cid in records:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            dropped["unparseable"] += 1
            continue
        n_parseable += 1
        # rdkit stubs omit DisableLog; it exists at runtime
        mw = Descriptors.MolWt(mol)  # pyright: ignore[reportAttributeAccessIssue]
        if (
            not (b.mw[0] <= mw <= b.mw[1])
            # rdkit stubs omit DisableLog; it exists at runtime
            or Crippen.MolLogP(mol) > b.logp_max  # pyright: ignore[reportAttributeAccessIssue]
            or rdMolDescriptors.CalcTPSA(mol) > b.tpsa_max
            # rdkit stubs omit DisableLog; it exists at runtime
            or Lipinski.NumHDonors(mol) > b.hbd_max  # pyright: ignore[reportAttributeAccessIssue]
            # rdkit stubs omit DisableLog; it exists at runtime
            or Lipinski.NumHAcceptors(mol) > b.hba_max  # pyright: ignore[reportAttributeAccessIssue]
            # rdkit stubs omit DisableLog; it exists at runtime
            or Lipinski.NumRotatableBonds(mol) > b.rotb_max  # pyright: ignore[reportAttributeAccessIssue]
        ):
            dropped["physchem"] += 1
            continue
        n_physchem += 1
        if catalog is not None and catalog.HasMatch(mol):
            dropped["pains"] += 1
            continue
        kept.append((Chem.MolToSmiles(mol), cid))
    return kept, dropped, n_parseable, n_physchem


def prepare_library(
    records: list[tuple[str, str]],
    *,
    bounds: LeadLikeBounds | None = None,
    apply_pains: bool = True,
    workers: int | None = None,
    chunk_size: int = 20000,
) -> LibraryResult:
    """Tier-0: parse → lead-like physchem → PAINS → dedup, tracking attrition.

    Each filter runs on the survivors of the previous one, so ``attrition`` reads as a
    monotonically-shrinking funnel. Returns canonical-SMILES records ready for Tier-0.5
    focusing / Tier-1 screening.
    """
    b = bounds or LeadLikeBounds()
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    payloads = [(c, b, apply_pains) for c in chunks]

    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(_prep_chunk, payloads))
    else:
        outs = [_prep_chunk(p) for p in payloads]

    dropped = {"unparseable": 0, "physchem": 0, "pains": 0, "duplicate": 0}
    n_parseable = n_physchem = 0
    seen: set[str] = set()
    records_out: list[dict] = []
    for kept, d, npars, nphys in outs:
        for key in ("unparseable", "physchem", "pains"):
            dropped[key] += d[key]
        n_parseable += npars
        n_physchem += nphys
        for canon, cid in kept:
            if canon in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(canon)
            records_out.append({"smiles": canon, "id": cid})

    n_pains_pass = n_physchem - dropped["pains"]
    attrition = [
        ("raw_input", len(records)),
        ("parseable", n_parseable),
        ("lead_like_physchem", n_physchem),
        ("pains_pass", n_pains_pass),
        ("unique_prepared", len(records_out)),
    ]
    return LibraryResult(records=records_out, attrition=attrition, dropped=dropped)
