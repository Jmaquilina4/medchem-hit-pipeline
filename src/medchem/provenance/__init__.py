"""Provenance tooling: answering "which artifacts did this run actually produce?".

Deliberately the OUTERMOST layer (see ``tests/test_architecture.py``). Answering that question requires
composing the whole stage graph — the registry, the runner's key computation, and the config — so this
package necessarily imports from every layer below it. It sat under ``pipeline/`` for one commit and the
architecture test caught the inversion immediately: a layer-2 module cannot import layer-6 stages.

Nothing in the pipeline imports this, which is the point twice over: provenance must not be able to
influence the thing it describes, and it stays out of the scientific-source digest's closure.
"""

from medchem.provenance.resolve import resolve_stage_keys, resolve_stage_outputs

__all__ = ["resolve_stage_keys", "resolve_stage_outputs"]
