"""medchem — a reproducible, configuration-driven pipeline for computational hit finding.

TARGET-AGNOSTIC by construction: stages register under one generic ``discovery`` pipeline, so a new
protein is a config change rather than a new code path, within the two limits the README states (the data
must be reachable as ChEMBL identifiers or an equivalent frozen snapshot, and one of the versioned cohort
specifications must express the assay policy). The frozen release exercises a kinase (JAK1) and a
bromodomain (BRD4).

Layout: a small dependency-ordered DAG in ``medchem.pipeline``; stage logic in the topic subpackages
(``data``, ``features``, ``models``, ``generative``, ``structure``, ``vls``, ``eval``); and
``medchem.stages``, whose only job is to import them so the registry is populated. ``medchem.cli`` is
presentation over the stages and is deliberately outside the results-determining digest.

Implementation status is not uniform, and the README's table is the authority. In brief: acquisition
through gated evaluation is reproduced for both targets; virtual screening is implemented and
implementation-tested; receptor preparation is implemented as an independent stage with no artifact
evidenced for the frozen panels; the generative path runs on CPU with a replay sampler, while the
REINVENT4 sampler and the Boltz-2 scorer are INTERFACES WHOSE SHIPPED METHODS RAISE; and no docking stage
is registered.
"""

__version__ = "0.2.0"
