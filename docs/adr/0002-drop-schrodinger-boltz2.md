# ADR 0002 — Drop Schrödinger; Boltz-2 as the open structural oracle

**Status:** Accepted · 2026-07

> **Scope in this release.** The decision stands, but the open structural oracle it selects is a
> **stub** here: `Boltz2Scorer` defines the interface and raises. No co-folding ran, and no result in
> this repository depends on one. Read this as a recorded decision, not as a description of a
> capability the release provides.

## Context
The v1 structural arm (Glide docking, Prime MM-GBSA, IFD-MD) is commercial
(Schrödinger/Maestro), which is incompatible with a **public, reproducible** repo that
anyone can clone and run.

## Decision
**Remove the Schrödinger arm from the reproducible pipeline.** Use **Boltz-2** (MIT
license, joint structure + affinity) as the open structural oracle: for consensus
triage (advance only where QSAR and Boltz-2 agree; disagreement → flagged) and as the
expensive oracle for the active-learning loop. In-loop hinge geometry is enforced with
a cheap RDKit pharmacophore/shape constraint. `gnina` is an optional third consensus
leg. No MM-GBSA / IFD-MD rebuild.

## Consequences
- The entire pipeline is reproducible on open tools, so every result here can be independently
  re-derived without a commercial licence. That is the property that makes the rest auditable.
- The v1 "MM-GBSA can lie" insight rebuilds cleanly as a QSAR-vs-Boltz-2 disagreement
  flag.
- **Honesty caveat:** Boltz-2 is *physics*-orthogonal to the ligand-based QSAR, not
  *data*-orthogonal (both lean on ChEMBL-type data). We never call agreement
  "independent confirmation."
- Earlier licensed-software figures, wherever they are reused, are labeled "prior, license-gated, not
  in the reproducible pipeline."
