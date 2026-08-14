"""Structure tier — ligand preparation and docking-box definition.

Ligand prep (SMILES → 3D conformer → PDBQT) is pure CPU and runs anywhere, so its cost is
measured on whatever box is available and **extrapolates directly** to the GPU run — it is
typically 20–40% of a docking campaign's wall clock and is CPU/disk-bound, not GPU-bound.
The docking *engines* are deliberately not imported here: Uni-Dock / Vina-GPU need CUDA and
Glide needs a Schrödinger licence, so the kernel is timed on the GPU box behind the
``DockingEngine`` protocol.

The box is derived from a **co-crystallised ligand** rather than guessed: for JAK1 that is
MI1 in 3EYG (``structure.reference_pdb``), whose centroid defines the ATP-site centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DockingBox:
    """Cubic search box, in Ångström, centred on the ATP site."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @classmethod
    def from_pdb_ligand(
        cls, pdb_path: str | Path, resname: str, *, padding: float = 3.5, min_size: float = 22.0
    ) -> DockingBox:
        """Centre the box on a co-crystal ligand's centroid, sized to its extent + padding.

        Padding is applied per side; the result is clamped to at least ``min_size`` so a small
        reference ligand still yields a box able to hold lead-like candidates.
        """
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        with open(pdb_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("HETATM") and line[17:20].strip() == resname:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
        if not xs:
            raise ValueError(f"ligand {resname!r} not found in {pdb_path}")
        center = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        size = tuple(
            max(min_size, (max(v) - min(v)) + 2 * padding) for v in (xs, ys, zs)
        )
        return cls(center=center, size=size)  # pyright: ignore[reportArgumentType]


@runtime_checkable
class DockingEngine(Protocol):
    """A docking backend. Open default = Uni-Dock/Vina; ``glide`` is a licensed swap.

    Implementations live where they can run (GPU box / licensed workstation); the pipeline
    depends only on this protocol, which is what keeps the open path the default (ADR 0005).
    """

    name: str

    def dock(self, pdbqt_paths: list[str], *, receptor: str, box: DockingBox, mode: str) -> list[dict]:
        """Dock prepared ligands; return one score record per input."""
        ...


def prepare_ligand(smiles: str, *, seed: int = 42, optimize: bool = True) -> str | None:
    """SMILES → 3D conformer → PDBQT text. Returns ``None`` if prep fails.

    ETKDG embedding + MMFF optimisation is the dominant CPU cost of a docking campaign's
    prep phase; failures (embedding non-convergence, unparseable input) are returned as
    ``None`` so the caller can count them rather than crash a 10⁵-ligand run.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]  # present at runtime; rdkit stubs omit it
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()  # pyright: ignore[reportAttributeAccessIssue]  # present at runtime; rdkit stubs omit it
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:  # pyright: ignore[reportAttributeAccessIssue]  # present at runtime; rdkit stubs omit it
        return None
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol)  # pyright: ignore[reportAttributeAccessIssue]  # present at runtime; rdkit stubs omit it
        except Exception:
            pass  # an unoptimised conformer still docks; don't lose the ligand
    try:
        setup = MoleculePreparation().prepare(mol)[0]
        pdbqt, ok, _err = PDBQTWriterLegacy.write_string(setup)
    except Exception:
        return None
    return pdbqt if ok else None
