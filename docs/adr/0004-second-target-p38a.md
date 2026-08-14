# ADR 0004 — p38α (MAPK14) as the generalization target

**Status:** SUPERSEDED · 2026-07, reversed 2026-08 — the second target shipped is **BRD4**, the option this ADR explicitly rejected.

> **Why the reversal matters more than the decision.** This ADR ruled out BRD4 on the grounds that
> a shallow water-mediated pocket would force code changes and break "config not code". That
> reasoning was a prediction, and it was the strongest available test of the retargeting claim — so
> BRD4 was run instead of p38α precisely *because* it was predicted to break.
>
> **The prediction was half right.** No pipeline code changed: BRD4 ran through the same generic
> `discovery` stage graph from a config. But the water network was real, and it surfaced as
> *config*: `structure.keep_waters_within` is load-bearing for a bromodomain and inert for a
> kinase, and a JAK1-tuned lead-like window excluded 5 of 7 BET reference compounds. p38α would
> have let those kinase assumptions ride along unexamined.
>
> Kept rather than deleted: a rejected option that later turned out to be the right test is the
> most useful thing in this directory. Current results: [../RESULTS.md](../RESULTS.md).

## Context
The generalization payoff runs the *same* pipeline on a second target by changing only
a config. The target must have (a) enough public ChEMBL data, (b) good co-crystal
coverage, (c) be different enough from JAK1 to prove generality, (d) be
chemically sanity-checkable by a medicinal chemist.

## Decision
Use **p38α / MAPK14 (CHEMBL260)**. Selectivity uses a **cross-MAPK panel (vs JNK/ERK)**,
not the p38 isoforms (p38β/γ/δ are too data-thin). Runner-up: **Factor Xa (CHEMBL244)**
as a bigger class-jump if time allows. **Not BRD4** (shallow water-mediated pocket forces
real code changes, breaking "config not code").

## Consequences
- p38α has ~10k ChEMBL activity records (live-verified) and 300+ diverse co-crystals; it
  is ATP-competitive, so featurization, hinge-constrained docking, and the direct-Δ
  selectivity module transfer 1:1 → target #2 is genuinely a config change.
- Canonical chemotypes (pyridinylimidazoles, diaryl-ureas) let the chemist judge output
  sanity at a glance.
- **Stress test:** the grid-box center and the hinge residue (JAK1 `Leu959` → p38α
  `Met109`) are the two params most likely to be accidentally hardcoded; they must live
  in the config. If any code needs editing for target #2, that finding is itself the
  honest result to report.
