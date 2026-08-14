# ADR 0001 — Minimal custom typed-Python DAG, not a workflow engine

**Status:** Accepted · 2026-07

## Context
The pipeline runs on a single node over ~3–5k compounds: QSAR trains in seconds,
docking/structure is hundreds of complexes, REINVENT is the one GPU step. This is a
*function-graph-with-caching* problem, not a cluster-scheduling problem.

## Decision
Implement a **minimal custom DAG**: a typed stage registry (`fn(ctx) -> StageResult`),
a ~150–250 LOC topological runner, and **content-addressed caching**
(`key = hash(stage source + config subtree + upstream keys)`), behind a thin Typer CLI
(`medchem run`). Explicitly **reject** Snakemake/Nextflow/Airflow/Prefect/Dagster/Flyte/
Ray/Dask/Kubernetes/MLflow-server/feature-store.

## Consequences
- Demonstrates the actual CS (topo-sort, content-addressing, idempotency) without
  operational overhead.
- Re-runs execute only what changed; the second target is a config change, not a
  rewrite.
- **Tripwire** that would justify a real workflow engine: fanning docking over
  thousands of poses on a shared cluster. Documented as a non-goal in the README.
- Reaching for an enterprise stack at this scale adds operational surface without adding
  scientific capability, and obscures rather than demonstrates engineering judgment.
