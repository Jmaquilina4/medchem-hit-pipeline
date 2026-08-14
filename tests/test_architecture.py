"""Structural invariants: dependencies point inward, and nothing imports in a circle.

These are cheap to check and expensive to discover late. The package had exactly one layering
violation — ``medchem.config`` (innermost) importing ``medchem.generative.scoring`` (outermost) to
introspect transform signatures — found by an architecture pass, not by any test. Fixing it meant
moving the transforms to a leaf module, which is the correct home anyway: two unrelated layers need
them, so they belong below both.

Layer numbers encode intent, not a physical constraint. Raising one is a design decision that should
be made deliberately by editing this file, which is the point.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "medchem"

# Lower may not import higher. Modules absent from this map are unconstrained (layer 9).
LAYERS = {
    "transforms": 0,   # pure math, imports nothing from the package
    "cohorts": 0,      # frozen assay-cohort spec: pure classification, needed by config AND curate
    "reward_components": 0,  # reward-component registry: needed by config AND the generative scorer
    "config": 1,       # typed config; validates against transforms
    "pipeline": 2,     # stage registry, DAG runner, cache
    "features": 3,     # featurisation
    "data": 3,         # pull + curate
    "models": 4,       # qsar, selectivity
    "eval": 4,         # evaluation harness
    "structure": 4,    # ligand prep, docking wrappers
    "generative": 5,   # samplers, scorer, design
    "vls": 5,          # virtual screen
    "stages": 6,       # registration side-effect module
    "cli": 7,          # entry point
}
UNCONSTRAINED = 9


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent)
    return str(rel).replace("/", ".")[:-3].removesuffix(".__init__")


def _layer(module: str) -> int:
    parts = module.split(".")
    return LAYERS.get(parts[1], UNCONSTRAINED) if len(parts) > 1 else 0


def _first_party_imports() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for f in sorted(SRC.rglob("*.py")):
        name = _module_name(f)
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("medchem"):
                graph[name].add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("medchem"):
                        graph[name].add(alias.name)
    return graph


def test_dependencies_point_inward():
    """An inner layer importing an outer one inverts the architecture and tends to create cycles
    later. Lazy imports inside functions still count: the dependency is real either way."""
    violations = [
        f"{mod} (L{_layer(mod)}) -> {dep} (L{_layer(dep)})"
        for mod, deps in _first_party_imports().items()
        for dep in deps
        if _layer(mod) < UNCONSTRAINED and _layer(dep) > _layer(mod)
    ]
    assert not violations, "layering violations:\n  " + "\n  ".join(sorted(violations))


def test_no_import_cycles():
    graph = _first_party_imports()
    cycles: set[tuple[str, ...]] = set()

    def walk(node: str, path: list[str]) -> None:
        for nxt in graph.get(node, ()):
            if nxt in path:
                cycles.add(tuple(path[path.index(nxt):] + [nxt]))
            elif nxt in graph and len(path) < 20:
                walk(nxt, path + [nxt])

    for mod in list(graph):
        walk(mod, [mod])
    assert not cycles, "import cycles:\n  " + "\n  ".join(" -> ".join(c) for c in sorted(cycles))


def test_transforms_module_stays_a_leaf():
    """Its whole purpose is being importable from any layer. One package import and it stops being
    usable by config, and the violation returns."""
    assert not _first_party_imports().get("medchem.transforms"), (
        "medchem.transforms must not import from the package — config depends on it being a leaf"
    )


def test_config_does_not_import_from_outer_layers():
    """Guards the specific regression: config reached into generative/ for the transform registry."""
    for dep in _first_party_imports().get("medchem.config", set()):
        assert _layer(dep) <= _layer("medchem.config"), (
            f"medchem.config imports {dep}, which sits at a higher layer"
        )
