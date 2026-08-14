"""Plug points for the generative loop.

A ``Sampler`` produces candidate molecules; an ``ComputationalScorer`` scores/validates one molecule with
structure. REINVENT4 (sampler) and Boltz-2 (scorer) implement these on GPU; a ``MockSampler``
lets the whole generate → score → select loop run and be tested on CPU. Keeping these as
Protocols is the composability seam (ADR 0006): the GPU implementations drop in behind them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Sampler(Protocol):
    """Generates candidate SMILES (optionally seeded/constrained on a fixed core)."""

    def sample(self, n: int, *, seed_smiles: str | None = None) -> list[str]:
        ...


@runtime_checkable
class ComputationalScorer(Protocol):
    """Structure-based score/validation for one molecule (e.g. Boltz-2 co-fold + affinity)."""

    def score(self, smiles: str) -> dict:
        ...
