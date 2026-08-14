# Validation report

**Date:** 2026-08-14 · **Cohort spec:** 1.2 · **Panels:** 4 · **Interpreter:** CPython 3.14.6

What was checked, how, and what it establishes. Every figure is recorded in
[`../provenance/`](../provenance/) and asserted against the documentation by
`scripts/verify_docs_against_manifests.py` in CI.

## Frozen panels

| panel | compounds | scaffold-CV R² | temporal R² | y-scramble R² | gate |
|---|---|---|---|---|---|
| JAK1 headline | 6,894 | 0.7597 | -0.3621 | -0.1749 | pass |
| JAK1 sensitivity | 7,912 | 0.7407 | -0.1018 | -0.1290 | pass |
| BRD4 headline | 2,794 | 0.7283 | +0.0190 | -0.1251 | pass |
| BRD4 sensitivity | 7,955 | 0.7523 | +0.1295 | -0.0787 | pass |

Gates are scaffold-CV R² ≥ 0.55 and y-scramble R² ≤ 0.10, **identical across all four panels** — the
property the four published configs demonstrate. No claim is made here about when they were set relative
to the results, because the published artifact cannot evidence that. The temporal split carries no gate by
design: it is reported as a finding, not scored.

## Verification performed

| check | method | result |
|---|---|---|
| static analysis | `ruff`, `pyright` | zero findings |
| unit and invariant tests | `pytest` | full suite green, including architecture layering and CI parity |
| input integrity | SHA-256 of all 16 published snapshots against the run manifests | match |
| snapshot content | decompressed and scanned (~80 MB of text) | no credentials, paths or identifying data |
| documentation | every quoted metric resolved against the manifests; superseded values rejected in active tables | match |
| figures | regenerated from `provenance/` only and compared | reproduce |
| exact reproduction | all four panels re-run cache-free from the published snapshot, offline for the metric-producing stages | agree within float64 rounding |
| history | every blob in every commit scanned | clean |
| links | every relative documentation link resolved | clean |
| publication hygiene | leak scan, binary-metadata scan | clean |

**Reproduction tolerance, per metric family.** Agreement is numerical, not bitwise, because float64
results vary in their last bits with the BLAS kernel, the thread count and the CPU.

**No independent reproduction on other hardware has been performed**, and none is recorded here. Every
figure in the right-hand column below is a SAME-MACHINE cache-free re-run, measured and recorded in
`provenance/REPRODUCTION_RUN.json`. The continuous bound sits about three orders of magnitude above its
measurement (1e-12 against 2.4e-15); the rank bound sits less than one order above its worst
(1e-5 against 1.6e-6), which is the tighter margin of the two and is stated rather than rounded up to
"orders of magnitude". Both carry headroom for cross-machine variation this repository has not itself
measured — an earlier version of this
table attributed that headroom to an outside reproduction that did not happen.

| family | bound | measured on a same-machine cache-free re-run |
|---|---|---|
| every metric this documentation reports | 1 × 10⁻¹² | ≤ 1.8 × 10⁻¹⁵ across all 8 compared records (~600 leaf values) |
| `roc_auc` / `pr_auc` / `pr_auc_lift_over_baseline` in `selectivity_metrics.json` | 1 × 10⁻⁵ | ≤ 4.7 × 10⁻⁷ in this run, and only in the two JAK1 panels; no rank movement in any `eval_report.json`. Four runs measured 4.7 × 10⁻⁷, 1.6 × 10⁻⁶, 1.6 × 10⁻⁶ and 4.7 × 10⁻⁷ — which near-ties swap varies per run, and that factor-of-three spread across source that computes identically is why this bound is called observed rather than derived. All four are recorded, with their source digests, in `provenance/REPRODUCTION_RUN.json` |
| support verdicts, support reasons, paired counts, supported comparators, basis column, production-model decision | exact | identical on all four panels. These are the decisions the two compared artifacts carry; assay-level cohort exclusion reasons live in `run_manifest.json`, which is not compared |

The looser bound is not a hedge, and it applies to three keys rather than to the results. Those three are
**rank** statistics computed on `RandomForestRegressor(n_jobs=-1)` predictions, and scikit-learn's
parallel prediction accumulates tree outputs in thread-completion order, so repeated `predict()` calls on
one fitted model differ by up to 1.3 × 10⁻¹⁵ (measured; exactly reproducible at `n_jobs=1`). A shift that
small cannot move a continuous metric, but it can swap two near-tied predictions, and a rank statistic
then moves by one discrete quantum. The JAK1 cohorts contain such near-ties and the BRD4 cohorts do not,
which is why only the former differ — a property of the data, not of the code.

The 1 × 10⁻⁵ figure is an **observed upper bound with headroom over the largest difference seen**, not a
derived limit: the quantum's size depends on how many near-ties a dataset contains, so a different cohort
could in principle move further. The movement is confined to the two JAK1 panels' `selectivity_metrics.json`; no `eval_report.json` moved at all, and no affected value appears in any document. (An earlier version said "the nine affected values". That count came from an earlier run's analysis and is not recorded anywhere, so it is replaced by what the published record does state: which artifacts moved.)

## Defect classes found and closed during development

Recorded as classes rather than as a change log; the detection mechanism is the transferable part.
Detail in [`PITFALLS.md`](PITFALLS.md).

| class | how it was detected |
|---|---|
| label leakage across an artifact boundary | a `train_label_source` field in the evaluation report |
| provenance pointing at the wrong input file | resolving artifacts by stage cache key instead of by filename |
| documentation drift | asserting documented numbers against manifests, and rejecting superseded values |
| a checker reporting success over zero inputs | making every check refuse to pass on an empty scope |
| assay-cohort impurity | auditing the description-based label against ChEMBL's structured fields |
| environment-dependent identity | recording the interpreter beside the digest and checking it first |

## Limitations

Stated in full in [`RESULTS.md`](RESULTS.md) and [`MODEL_CARD.md`](MODEL_CARD.md). In brief: chronological
generalisation is poor to negligible; in the **headline BD1-explicit, structured single-protein binding IC50 cohort** no BET
selectivity pair is supported and no BRD4 selectivity model is written, while the pooled sensitivity
cohort supports 2 of 3 pairs and does write one; bootstrap intervals ignore scaffold-group dependence; the
y-scramble null is a single permutation with no empirical p-value; and **no candidate this pipeline
nominates has received prospective experimental validation** — the ChEMBL inputs are themselves
experimental assay measurements.
