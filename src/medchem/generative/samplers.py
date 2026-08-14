"""Sampler implementations.

``MockSampler`` is deterministic and CPU-only — enough to exercise the generate → score →
select loop and the reward-hacking → recovery comparison without a GPU. ``Reinvent4Sampler``
is the real Mol2Mol → LibInvent generator (ADR 0003). It is an **explicit stub**: it defines the
interface and raises rather than pretending to sample, because running it needs GPU execution that
this repository does not provide. Any such backend is optional and external, and none ships here.
Both implementations sit behind the same ``Sampler`` protocol, so the loop is unchanged either way.
"""

from __future__ import annotations

from collections.abc import Sequence


class MockSampler:
    """Replays a fixed candidate pool. Deterministic; for wiring + tests."""

    def __init__(self, pool: Sequence[str]) -> None:
        self._pool = list(pool)

    def sample(self, n: int, *, seed_smiles: str | None = None) -> list[str]:
        return self._pool[:n]


class Reinvent4Sampler:
    """REINVENT4 constrained generator (Mol2Mol scaffold-hop → LibInvent decoration on a fixed
    hinge core). GPU-only — implement ``sample`` to invoke REINVENT4 in the CUDA image and
    return the generated SMILES. See ADR 0003."""

    def __init__(self, *, seed_core: str, config: dict | None = None) -> None:
        self.seed_core = seed_core
        self.config = config or {}

    def sample(self, n: int, *, seed_smiles: str | None = None) -> list[str]:
        raise NotImplementedError(
            "Reinvent4Sampler runs on GPU — wire it into the CUDA image / Flyte GPU task (ADR 0003)."
        )
