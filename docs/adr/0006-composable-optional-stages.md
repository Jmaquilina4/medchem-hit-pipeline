# ADR 0006 — Composable, optional pipeline stages

**Status:** Accepted · 2026-07

## Context
The pipeline is a reusable engine, and not every project needs every stage — a
single-target project has no isoform selectivity, for example. We want modularity (run a
subset, turn a feature off, plug in your own data/model at a boundary) **without** drifting
into a general plugin framework, which [ADR 0001](0001-orchestration-minimal-dag.md)
explicitly rejects (right-sized engineering over unnecessary machinery). JAK1 fast-follower stays
the opinionated flagship; modularity is *how it's built*, not the product.

## Decision
Lean on the DAG that already exists (independent registered stages with declared deps +
config_keys, a topo-sort runner, content-addressed caching, `--stage` / `--from`). Add three
small seams — **no plugin registry, no on/off toggle for everything**:

1. **Config-driven optional stages.** `disable_stages: [selectivity]` in the target config
   drops stages from the run; the runner errors if a *kept* stage depends on a disabled one.
   (Required fixing `evaluate`'s over-declared dependency on `selectivity`, which it never read
   — `selectivity` is now a leaf, so it's cleanly optional.)
2. **Clean artifact contracts at each boundary.** Every stage reads/writes named artifacts, so
   any stage runs standalone (`--stage`) or against injected upstream data.
3. **Documented swap points.** BYO data (`data.local_sources`, planned), BYO model
   (`model.potency`, which the QSAR stage now reads), BYO featurizer (`features`). Three seams,
   not a framework.

## Consequences
- A project composes the pipeline from config. The core (curate → featurize → QSAR → eval) is
  target-agnostic — it runs on any target with activity data; selectivity / generative /
  structure / VLS are optional tiers.
- The VLS tiers ([ADR 0005](0005-vls-tiered-funnel.md)) are independently enable-able by the
  same principle.
- We deliberately do **not** build a plugin API, universal on/off toggles, or generality across
  all of small-molecule discovery — that is a product, not a study, and the extra abstraction would
  cost clarity exactly where the scientific argument needs to be sharpest.
