\
# Frozen ChEMBL snapshots

These files are the exact bioactivity records the frozen results were computed from. They are published
so the metrics in this repository can be **recomputed from the same bytes**, rather than re-derived
against a moving database. Agreement is expected within the numerical tolerance the README documents,
not to the last bit: float64 rounding varies with BLAS kernel, thread count and CPU.

## Source and licence

Data source: **ChEMBL**, European Molecular Biology Laboratory — European Bioinformatics Institute
(EMBL-EBI), <https://www.ebi.ac.uk/chembl/>. Release **34 as requested** in the run configs. The REST
query cannot pin a release and the served release was never confirmed, so what is pinned is these
**bytes**, by the SHA-256 values in `SHA256SUMS`.

ChEMBL data is made available under the **Creative Commons Attribution-ShareAlike 3.0 Unported licence**
(CC BY-SA 3.0): <https://creativecommons.org/licenses/by-sa/3.0/>. Redistribution is permitted with
attribution under the same terms, so **these data files are distributed under CC BY-SA 3.0**, not under
the repository's code licence. Anyone redistributing them further must preserve this notice and the
share-alike term.

The files are unmodified subsets: the records ChEMBL returned for the configured targets and activity
types, with a fixed column selection. No values were altered, imputed, or filtered — all curation
happens downstream in the pipeline, from these bytes.

## Contents

One gzipped CSV per target, plus assay metadata per target. Cohort selection depends on assay
*descriptions*, so a snapshot of activities alone would not be sufficient to reproduce a cohort.

`SHA256SUMS` lists the checksum of each file **after decompression** — that is what the pipeline
verifies and what the run manifests record. Only the `.csv.gz` files ship, so a plain
`sha256sum -c SHA256SUMS` will not find them. Verify like this instead:

```bash
cd data/frozen_snapshots
while read -r want name; do
  case "$want" in \#*) continue ;; esac
  got=$(gzip -dc "$name.gz" | shasum -a 256 | cut -d' ' -f1)
  [ "$got" = "$want" ] && echo "OK   $name" || echo "FAIL $name"
done < SHA256SUMS
```

## Reproducing the frozen metrics

```bash
MEDCHEM_FROZEN_SNAPSHOT=data/frozen_snapshots \
  uv run medchem run -p discovery -c configs/brd4.yaml \
  --stage data_pull --stage curate --stage featurize \
  --stage qsar --stage selectivity --stage evaluate
```

The stage list matters: it restricts the run to the stages that produce the metrics, all of which are
offline once the snapshot is restored. The full graph also runs `receptor`, which **fetches a structure
from the RCSB PDB** — so "no network" is true of the stages above and false of the whole graph.

Without the environment variable the pipeline pulls live from ChEMBL, which reproduces the workflow
against current data and will *not* match the frozen numbers.
