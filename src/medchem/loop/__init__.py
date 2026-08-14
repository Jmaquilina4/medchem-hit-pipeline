"""Placeholder for the active-learning outer loop. NOTHING HERE RUNS.

The interface it would implement lives in ``medchem.generative.active_learning`` and is documented there
as interface-only. The corresponding ``loop`` config section was DELETED rather than left as decoration --
an audit found none of its keys read anywhere -- and the root config now rejects it, so a config cannot
imply this loop executes. This module is retained only as the namespace that interface refers to.
"""
