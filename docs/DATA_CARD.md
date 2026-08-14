# Data Card — frozen ChEMBL datasets

Covers the four frozen evaluation panels. Every figure here is recorded in
[`provenance/`](../provenance/) and asserted against the documentation by
`scripts/verify_docs_against_manifests.py`.

## Source and licence

**ChEMBL**, EMBL-EBI, <https://www.ebi.ac.uk/chembl/>. Stated precisely, because the honest version is
weaker than "release 34":

* **Requested** release: 34, as configured. The REST query does **not** pin a release, so this is a label
  on the request, not a property of the response.
* **Served** release: **unconfirmed.** The pipeline probes for it and records whatever it gets, but the
  API did not confirm it for these pulls, and the frozen-snapshot restore path contacts no network at
  all. No claim is made about which release answered.
* **What IS pinned:** the exact bytes. Every one of the 16 raw inputs is fixed by SHA256 in the run
  manifests and republished in [`../data/frozen_snapshots/`](../data/frozen_snapshots/) with those
  checksums. A retrieval timestamp is written into the run tree, which is not published, so what a
  reader can verify is the bytes and their checksums — not when they were fetched. The snapshot —
  not a release label — is what
  makes the metrics exactly reproducible.

ChEMBL data is **CC BY-SA 3.0**. The raw records are **redistributed in this repository** — 16 gzipped
CSVs in [`data/frozen_snapshots/`](../data/frozen_snapshots/) (2.7 MB), with checksums and attribution in
that directory's README. Redistribution is permitted with attribution and share-alike, so those files
carry CC BY-SA 3.0 rather than the repository's code licence.

## Targets

| panel | primary | comparators |
|---|---|---|
| JAK1 (headline, sensitivity) | JAK1 `CHEMBL2835` | JAK2 `CHEMBL2971`, JAK3 `CHEMBL2148`, TYK2 `CHEMBL3553` |
| BRD4 (headline, sensitivity) | BRD4 `CHEMBL1163125` | BRD2 `CHEMBL1293289`, BRD3 `CHEMBL1795186`, BRDT `CHEMBL1795185` |

Target identifiers were resolved against the live API rather than from memory: ChEMBL's preferred name
for the BET proteins is "Bromodomain-containing protein N", so a gene-symbol search for BRD4 returns an
unrelated protein.

## Composition, and why it required a specification

A single-protein ChEMBL target is a **mixture of assays that disagree**. BRD4's IC50 records are 47.7%
first-bromodomain, 25.0% biochemical with no domain stated, 15.8% cell-based, 6.3% second-bromodomain,
5.1% unmatched. Counted per *assay* rather than per *record*, cell-based is 58.8%. One phase-3 reference
compound reads 5.85 on one domain against 6.88 on the other.

Cohorts are therefore versioned (`medchem/cohorts.py`, **spec 1.2**) and each run records the exact assay
IDs it admitted and excluded, each with its own reason. The BRD4 headline cohort is fail-closed on
ChEMBL's structured fields for the assay FORMAT — single-protein BAO format, binding assay type — plus a
description-level exclusion of tandem-domain constructs. That removed 197 assays a description-only rule
had admitted. Note the split: the structured fields confirm the format; BD1 identity is still read from
the description, because ChEMBL does not encode the domain. Full tables in
[`RESULTS.md`](RESULTS.md).

## Curation

IC50 only, enforced on both the pChEMBL and the unit-conversion path. ChEMBL validity flags and
explicit inactive/not-determined comments dropped. Units converted from `standard_value` rather than
trusted from a label. One median pIC50 per canonical structure. When a temporal cutoff is set, labels are
split by era so training never sees a post-cutoff measurement.

| panel | curated rows | compounds | drug-like subset |
|---|---|---|---|
| JAK1 headline | 23,081 | 6,894 | 5,097 |
| JAK1 sensitivity | 27,512 | 7,912 | 5,952 |
| BRD4 headline | 3,036 | 2,794 | 2,150 |
| BRD4 sensitivity | 9,805 | 7,955 | 5,767 |

## Known limitations

- **Heterogeneous by nature.** Assay format and protein construct vary within a single target; that is
  what the cohort specification exists to make explicit rather than to hide.
- **Not chronologically representative.** A chronological split is not a controlled prospective test:
  the post-cutoff records are whatever the literature happened to publish, not a designed hold-out. The
  66.0–76.8% scaffold overlap quoted elsewhere is the **random** split's; the temporal test sets share
  only 5.0–12.1% of their scaffolds with training, so they are close to scaffold-disjoint already.
- **Censored values excluded.** Relations other than `=` are dropped rather than imputed, which biases
  the retained set toward measurable potencies.
- **Comparator thinness.** BRDT contributes 356 IC50 records against TYK2's 6,381, and BD1-explicit,
  structured single-protein binding IC50 matching leaves the BET comparator panels at 80, 56 and 54 paired measurements — which is why no BET
  selectivity pair is supported.
- **Live source.** ChEMBL grows, so a fresh pull will not reproduce these counts. That is what the
  published snapshots are for.
