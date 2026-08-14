"""Structural scorer implementations (behind the ``ComputationalScorer`` protocol).

``MockScorer`` is deterministic + CPU — enough to exercise the active-learning loop. ``Boltz2Scorer``
is the real co-folding + affinity scorer (ADR 0002). It is an **explicit stub**: GPU execution is
required to run it, that backend is optional and external, and none ships here. It sits behind the same
interface, so the loop is unchanged either way.
"""

from __future__ import annotations


class MockScorer:
    """Deterministic stand-in structural score for a molecule. For wiring + tests only."""

    def score(self, smiles: str) -> dict:
        # deterministic, bounded pseudo-affinity (no network, no GPU)
        s = 1.0 / (1.0 + 0.05 * len(smiles))
        return {"structure_score": float(s), "valid": True, "smiles": smiles}


class Boltz2Scorer:
    """Boltz-2 co-folding + affinity scorer (ADR 0002). GPU-only — implement ``score`` to co-fold
    the ligand into the target and return a structural score/affinity.

    Honesty (ADR 0002): Boltz-2 affinity is *data*-correlated with the ChEMBL QSAR, so log
    QSAR/Boltz-2 agreement as ONE opinion — only physics (FEP) is independent confirmation.
    """

    def __init__(self, *, reference_pdb: str, hinge_residue: str | None = None) -> None:
        self.reference_pdb = reference_pdb
        self.hinge_residue = hinge_residue

    def score(self, smiles: str) -> dict:
        raise NotImplementedError(
            "Boltz2Scorer runs on GPU — wire it into the CUDA image / Flyte GPU task (ADR 0002)."
        )
