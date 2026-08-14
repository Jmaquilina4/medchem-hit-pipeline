# Architecture Decision Records

Short records of the **judgment calls** behind this project — the "why," not just the "what." The project's thesis is *engineering taste over tool count*, so the reasoning is a first-class artifact.

Format (lightweight [MADR](https://adr.github.io/madr/)): **Context → Decision → Consequences**, plus a one-line status.

| ADR | Decision | Status |
|-----|----------|--------|
| [0001](0001-orchestration-minimal-dag.md) | Minimal custom typed-Python DAG (not a workflow engine) | Accepted |
| [0002](0002-drop-schrodinger-boltz2.md) | Drop Schrödinger; Boltz-2 as the open structural oracle | Accepted |
| [0003](0003-generative-mol2mol-libinvent.md) | Constrained generation: Mol2Mol → LibInvent | Accepted |
| [0004](0004-second-target-p38a.md) | p38α (MAPK14) as the generalization target | **Superseded** — BRD4 shipped instead |
| [0005](0005-vls-tiered-funnel.md) | Virtual library screening: open-by-default tiered funnel | Accepted (design) |
| [0006](0006-composable-optional-stages.md) | Composable optional stages in the DAG | Accepted |

New decision? Copy the shape of an existing ADR, increment the number, link it here.
