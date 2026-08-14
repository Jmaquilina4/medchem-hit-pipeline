"""Receptor preparation, docking inputs, and a docking harness nothing in the graph invokes.

WHAT RUNS: ``receptor`` prepares a docking-ready receptor and box from a configured PDB entry, asserting
that the named hinge/anchor residue exists and lies inside the box. It is an independent stage; no
receptor artifact is evidenced or published for the frozen panels, and nothing downstream consumes one.

WHAT DOES NOT: ``dock`` is a callable subprocess harness for an engine with a Vina-compatible command line
and score table, with reason-coded outcomes. No registered pipeline stage invokes it, no engine is
bundled, and no published result uses it. ``cofold`` prepares co-folding inputs; the Boltz-2 scorer itself
is an interface that raises.
"""
