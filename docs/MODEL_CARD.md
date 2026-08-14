# Model Card — potency and selectivity models

Covers the models behind the four frozen panels. Every figure is recorded in
[`provenance/`](../provenance/) and asserted against this documentation by
`scripts/verify_docs_against_manifests.py` in CI. Full tables: [`RESULTS.md`](RESULTS.md).

## Model details

| | |
|---|---|
| **Task** | regression of pIC50 (potency) and of Δ pIC50 between two proteins (selectivity) |
| **Architecture** | RandomForest (400 trees) and XGBoost, both trained; RF is the reported model |
| **Features** | chiral ECFP4, radius 2, 2048 bits, plus 9 RDKit descriptors (MolWt, MolLogP, TPSA, HBD, HBA, QED, rotatable bonds, aromatic rings, FractionCSP3) |
| **Featurisation** | identical across both targets, deliberately — retuning it per target would weaken the controlled cross-target comparison |
| **Targets** | JAK1 (kinase) and BRD4 (bromodomain), each with a headline and a pooled sensitivity cohort |
| **Cohort spec** | 1.2 — BRD4 headline is BD1-explicit (from the assay DESCRIPTION) with a structured single-protein binding FORMAT (from ChEMBL's structured fields) -- the structured fields confirm the assay format, not BD1 identity; fail-closed on ambiguity |
| **Environment** | Python 3.14.6, scikit-learn 1.9.0, RDKit 2026.3.4, XGBoost 3.3.0 — recorded per run |

## Intended use

**Computational prioritisation within represented chemical space.** Ranking candidate compounds that
resemble the training chemistry, to decide what to look at next.

## Out of scope

- **Prospective activity prediction.** The chronological evaluation is the reason: JAK1 temporal
  R² **−0.362**, BRD4 **+0.019**. See the limitation below; it is the headline finding, not a caveat.
- **BET selectivity on the headline cohort.** On the BD1-explicit, structured-binding cohort **none** of
  the three pairs is distinguishable from noise (n = 80, 56, 54; every interval spans zero), so no
  selectivity model was written for the BRD4 **headline** cohort. The pooled sensitivity cohort supports
  2 of 3 pairs and does write a model — which is why this bullet names the cohort rather than the target
  family. An earlier version read "BET selectivity of any kind", contradicting its own next sentence.
- **Any decision that would be costly if wrong.** No candidate this pipeline nominates has received
  prospective experimental validation. The model *inputs* are public experimental ChEMBL assay
  measurements; its *outputs* are computational priorities.

## Evaluation

| | JAK1 headline | JAK1 sensitivity | BRD4 headline | BRD4 sensitivity |
|---|---|---|---|---|
| compounds | 6,894 | 7,912 | 2,794 | 7,955 |
| scaffold-CV R² *(gate ≥ 0.55)* | 0.7597 | 0.7407 | 0.7283 | 0.7523 |
| temporal R² *(2022 cutoff)* | −0.3621 | −0.1018 | +0.0190 | +0.1295 |
| y-scramble R² *(gate ≤ 0.10)* | −0.1749 | −0.1290 | −0.1251 | −0.0787 |

**Splits.** Bemis–Murcko scaffold-grouped 5-fold CV is the reported metric. Random splits are computed
but not used as a headline: test sets share 66.0–76.8% of their scaffolds with training. The temporal
split trains only on pre-cutoff labels, so no post-cutoff measurement informs a training label.

**Gates** are **identical across all four panels**, which is the property the published configs prove
and the one the cross-target comparison rests on: a cohort change could not be rewarded with a looser
bar. The temporal split has **no** gate deliberately — it is reported as a
finding, not scored as a pass.

**Negative control.** The y-scramble null passed on all four runs. A value near zero is the expected
outcome; a less-negative value is not weaker evidence than a more-negative one. Its limitation is design: a **single permutation**,
so no empirical p-value was estimated.

## Limitations

1. **Retrospective ≠ chronological.** Strong scaffold generalisation, poor-to-negligible chronological
   generalisation, on both target classes.
2. **Scaffold overlap belongs to the random split, not the temporal one.** The 66.0–76.8% figure is
   computed on the random 80/20 split, which is why that split is not a headline. The temporal test sets
   share only **5.0–12.1%** of their scaffolds with training, so the negative temporal R² cannot be
   attributed to shared scaffolds and the chronological split is the harder test. An earlier version of
   this line said overlap "flatters even the temporal figures", which the measurement contradicts and
   which contradicted RESULTS.md.
3. **Assay heterogeneity.** A single-protein target is a mixture of assays that disagree by up to
   1.03 log units on one molecule. Cohorts make the choice explicit; spec 1.2 requires a structured
   single-protein binding format, which removed 197 assays a text-only rule had admitted.
4. **BET selectivity is unsupported on the headline cohort.** BD1-explicit structured-binding matching
   costs the comparators 63–88% of their compounds, and no pair survives *in that cohort*. The pooled
   `target_associated` sensitivity cohort supports 2 of 3 pairs and writes a production model; which of
   the two answers the question depends on whether BD1-resolved measurements are required, and that is
   the point of running both.
5. **Intervals are optimistic.** Bootstrap R² intervals resample fixed out-of-fold predictions without
   refitting, and ignore scaffold-group dependence.
6. **No applicability-domain gate at inference.** Coverage is measured and reported, not enforced.

## Ethical and safety considerations

Outputs are computational hypotheses about binding affinity. They carry no assessment of toxicity,
selectivity beyond the pairs evaluated, pharmacokinetics, or synthetic accessibility beyond a heuristic.
Nothing here should be read as a recommendation to synthesise or administer any compound.
