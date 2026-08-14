"""The reward components the generative scorer emits — the names, and where each comes from.

A leaf module for the same reason :mod:`medchem.transforms` is one. Two layers need this and they
sit at opposite ends of the graph: :mod:`medchem.generative.scorer` produces the components, and
:mod:`medchem.config` validates that a configured scoring spec only asks for components that exist.
Putting the registry in the scorer made ``config`` import from ``generative``, which is the innermost
layer reaching into the outermost — the one layering violation this package has already fixed once.

Why the check is worth the module. ``medchem.generative.scoring.score_components`` scores a component
it cannot find as ``0.0``, and ``aggregate`` then floors every component at ``1e-9`` before taking the
weighted geometric mean, precisely so one zero cannot annihilate the product. The consequence of a
misspelled name — ``qsar_pic_50`` for ``qsar_pic50`` — is therefore not a zero score. It is worse to
notice: that objective becomes INERT. Every candidate receives the same floored constant for it, which
is a monotone-neutral factor, so the ranking is decided entirely by the remaining components while the
report still lists the objective as scored.

Measured on the shipped seven-component JAK1 spec: misspelling ``qsar_pic50`` leaves scores non-zero
(0.026, 7.9e-05, 0.023 on three test molecules) and produces exactly the ranking obtained by DELETING
that component from the spec. The reward magnitude drops, which looks like a hard problem rather than a
typo, and the potency objective the campaign exists to optimise is silently not optimised.

(An earlier version of this note claimed the misspelling zeroed every candidate and made the selection
arbitrary. It does not; ``aggregate`` prevents that. The floor is the reason the failure is quiet.)

Nothing here imports from the rest of the package, and nothing here should.
"""

from __future__ import annotations

# Components read straight out of the RDKit descriptor block, mapped to the descriptor each needs.
# `features.descriptors` decides which of those are computed, so a spec asking for `qed` under a
# descriptor list without `QED` is a config error the scorer reports by name.
DESCRIPTOR_COMPONENTS: dict[str, str] = {
    "mw": "MolWt",
    "slogp": "MolLogP",
    "tpsa": "TPSA",
    "qed": "QED",
}

# Everything the scorer can emit. `selectivity_delta` is here because a spec may legitimately request
# it; it is produced only when a selectivity model is supplied, and ``score_molecules`` raises with
# its own message when the spec asks for it and the model is absent.
SCORED_COMPONENTS: frozenset[str] = frozenset(
    {"qsar_pic50", "selectivity_delta", "applicability_domain", *DESCRIPTOR_COMPONENTS}
)
