"""Acquisition and curation: ChEMBL retrieval or frozen-snapshot restore, then labels.

Curation applies a versioned assay cohort, standardises units from ``standard_units`` rather than from any
label, aggregates replicates by median, and builds both all-years and era-split labels. Cohort selection
reads assay metadata, which is why an alternative acquisition source must supply that table too.
"""
