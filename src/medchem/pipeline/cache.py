"""Content-addressed hashing for stage caching + provenance.

A stage's cache key is ``sha256(medchem version + stage source + resolved config
subtree + upstream output hashes)``. Changing the medchem version, the stage
function's own source, the config sections the stage declares, or any upstream
artifact invalidates the key.

Scope: ``hash_source`` hashes the transitive closure of first-party modules a stage can
reach, each from its parsed AST with docstrings stripped. So a real change to a shared
helper like ``features.compute_features`` invalidates every stage that uses it, while a
comment or docstring edit invalidates nothing. Both halves matter: the old text-based,
single-module hash under-invalidated across modules (a correctness risk) and
over-invalidated within one (a pure cost).
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object (dict order-independent)."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256(canonical.encode())


def hash_file(path: str | Path) -> str:
    """Streaming sha256 of a file's contents."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstring literals so prose edits do not invalidate a cache key."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


@functools.lru_cache(maxsize=256)
def _normalised_module_hash(path: str) -> str:
    """Hash a module's PARSED form, not its text.

    Comments never reach the AST and docstrings are stripped, so reformatting or rewriting a comment
    leaves the key unchanged. Only a change in what the code *does* moves it.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return _sha256(path.encode())
    return _sha256(ast.dump(_strip_docstrings(tree), annotate_fields=False).encode())


@functools.lru_cache(maxsize=256)
def _static_imports(path: str, root: str) -> tuple[str, ...]:
    """First-party modules imported anywhere in a file, INCLUDING inside function bodies.

    Runtime introspection of ``vars(module)`` sees only module-level imports. This package defers many
    imports into function bodies so optional dependencies stay optional, and a deferred first-party
    import is just as load-bearing as a top-level one — so the file is parsed rather than inspected.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == root:
            found.add(node.module)
            # `from medchem.features import featurize` also depends on the submodule
            found.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] == root)
    return tuple(sorted(found))


def _module_file(name: str) -> str | None:
    mod = sys.modules.get(name)
    path = getattr(mod, "__file__", None)
    if path:
        return path
    try:  # not yet imported (a deferred import that has not run) -- resolve it statically
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    return spec.origin if spec and spec.origin else None


@functools.lru_cache(maxsize=256)
def _first_party_closure(module_name: str) -> tuple[str, ...]:
    """Every ``medchem`` module reachable from ``module_name``, including itself, sorted.

    Two discovery methods, unioned, because neither alone is sufficient: runtime attribute
    inspection catches re-exports that no import statement names, and static AST parsing catches
    imports deferred into function bodies that never appear in ``vars()``.
    """
    root = module_name.split(".")[0]
    seen: set[str] = set()
    stack = [module_name]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        deps: set[str] = set()
        mod = sys.modules.get(name)
        if mod is not None:
            for attr in vars(mod).values():
                if isinstance(attr, ModuleType):
                    dep = getattr(attr, "__name__", None)
                else:
                    dep = getattr(attr, "__module__", None)  # function/class -> its home module
                if dep:
                    deps.add(dep)
        path = _module_file(name)
        if path:
            deps.update(_static_imports(path, root))
        # `from x import y` yields both the module and the SYMBOL name; only keep what resolves to
        # a real module file, or the closure fills with unresolvable entries that hash as no-source.
        stack.extend(
            d for d in deps
            if d.split(".")[0] == root and d not in seen and _module_file(d) is not None
        )
    return tuple(sorted(seen))


def hash_source(fn: Callable[..., Any]) -> str:
    """Hash everything a stage's behaviour can depend on: its module AND its first-party imports.

    The previous version hashed only the stage function's own module text, which was wrong in both
    directions and for the same reason — it tracked *file text* rather than semantic dependency:

    * **Under-invalidated across modules.** ``features.compute_features`` is used by the qsar,
      selectivity, generative and vls stages but lives elsewhere, so editing featurisation
      invalidated none of them. Stale features would have produced a coherent, wrong result set, and
      the documented remedy was to remember to bump ``medchem.__version__`` by hand.
    * **Over-invalidated within a module.** Any edit to the stage's module — a comment, a docstring, a
      constant extraction — invalidated that stage and everything downstream. Extracting one constant
      once cost ~25 minutes of recomputation for a change that could not alter an output value.

    Now: the transitive closure of first-party modules, each hashed from its parsed AST with
    docstrings stripped. Prose edits are free; a real change anywhere in the closure invalidates.

    Falls back to a deterministic ``module.qualname`` when source is unavailable (frozen/exec'd/C
    code) — never the object ``repr``, whose memory address would make keys vary between runs.
    """
    module = inspect.getmodule(fn)
    name = getattr(module, "__name__", None) or getattr(fn, "__module__", "")
    if not name:
        return _sha256(f"{getattr(fn, '__qualname__', 'unknown')}".encode())

    parts: list[str] = []
    for dep in _first_party_closure(name):
        dep_file = _module_file(dep)
        if dep_file:
            parts.append(f"{dep}:{_normalised_module_hash(dep_file)}")
        else:
            parts.append(f"{dep}:no-source")
    if not parts:  # no source anywhere: stay deterministic rather than falling back to repr
        return _sha256(f"{name}.{getattr(fn, '__qualname__', 'unknown')}".encode())
    return _sha256("\n".join(parts).encode())


def stage_cache_key(
    *,
    code_version: str,
    source_hash: str,
    config_subtree: Any,
    upstream_hashes: dict[str, str],
) -> str:
    """Combine the four inputs into a single stage cache key."""
    return hash_obj(
        {
            "code_version": code_version,
            "source": source_hash,
            "config": config_subtree,
            "upstream": upstream_hashes,
        }
    )
