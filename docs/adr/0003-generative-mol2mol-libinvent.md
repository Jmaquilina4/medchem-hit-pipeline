# ADR 0003 — Constrained generation: Mol2Mol → LibInvent

**Status:** Accepted · 2026-07

> **Scope in this release.** The decision stands, but the generator is a **stub**:
> `Reinvent4Sampler` defines the interface and raises, and no generative campaign is included. The
> naive-vs-constrained comparison described here is implemented but was not run for any published
> result.

## Context
v1's de-novo REINVENT run produced >3,000 molecules of which *none* kept hinge binding
or drug-likeness. That was a **mode error, not effort**: unconstrained de-novo
generation has no reason to preserve the hinge pharmacophore, and a *sum*-of-scores
reward let one cheap objective dominate (reward-hacking), yielding an out-of-domain
lead (TPSA 154, logP 0.3, ~0.25 Tanimoto to training).

## Decision
Two-stage **constrained** generation:
1. **Mol2Mol** (similarity/MMP-constrained) to scaffold-hop tractable cores *within* the
   applicability domain.
2. Chemist picks 1–2 cores → **LibInvent** decorates R-groups on the **fixed**
   hinge-binding core, so the pharmacophore is preserved **by construction**.

Reward = transformed weighted **geometric mean** (product) — sigmoid on potency/
selectivity/QED, **double-sigmoid** on MW/SlogP/TPSA (penalize both extremes) — plus a
first-class **applicability-domain penalty**.

## Consequences
- The exact v1 failure (lost hinge) is impossible under LibInvent — state this
  precisely: preserved *by construction*, not because "the model learned it."
- A product reward forces all objectives satisfied simultaneously, killing the
  single-objective reward-hacking that produced the v1 lead.
- Enables an honest, quantified **failure → recovery** experiment (naive vs constrained arms)
  framed as Goodhart's law / reward hacking: the naive arm is expected to exploit the reward, and
  demonstrating the failure alongside its fix is a stronger result than a single clean run.
