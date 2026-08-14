"""A cache key must move when behaviour changes, and stay put when only prose changes.

The original key hashed the stage function's own module TEXT, which was wrong in both directions:

* it under-invalidated across modules -- `models.selectivity` imports `features.compute_features` but
  its DAG dependency is `curate`, so editing featurisation invalidated nothing and a stale-feature run
  would have looked entirely normal;
* it over-invalidated within a module -- extracting one constant in `curate.py` forced ~25 minutes of
  recomputation for a change that could not alter an output value.

Note the blast radius was ONE stage, not four: `qsar`, `evaluate`, `generative` and `vls` all have
`featurize` as a DAG upstream, so its output hash already fed their keys.
"""

from __future__ import annotations

import medchem.stages  # noqa: F401  (registers stages, populating sys.modules)
from medchem.pipeline.cache import (
    _first_party_closure,
    _module_file,
    _normalised_module_hash,
    hash_source,
)


def test_comment_and_docstring_edits_do_not_move_the_hash(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text(
        '"""Original docstring."""\n\n'
        "# a comment\n"
        "X = 1\n\n\n"
        "def f(y):\n"
        '    """Doc."""\n'
        "    return y + X\n"
    )
    b.write_text(
        '"""Completely rewritten prose, much longer than it was before."""\n\n'
        "X = 1\n\n\n"
        "def f(y):\n"
        '    """Different doc entirely."""\n'
        "    # a new comment in a new place\n"
        "    return y + X\n"
    )
    assert _normalised_module_hash(str(a)) == _normalised_module_hash(str(b))


def test_a_real_code_change_moves_the_hash(tmp_path):
    a = tmp_path / "c.py"
    b = tmp_path / "d.py"
    a.write_text("X = 1\n\n\ndef f(y):\n    return y + X\n")
    b.write_text("X = 2\n\n\ndef f(y):\n    return y + X\n")   # a constant that changes behaviour
    assert _normalised_module_hash(str(a)) != _normalised_module_hash(str(b))


def test_whitespace_and_formatting_do_not_move_the_hash(tmp_path):
    a = tmp_path / "e.py"
    b = tmp_path / "f.py"
    a.write_text("def f(y):\n    return y+1\n")
    b.write_text("def f(y):\n\n    return y + 1\n")
    assert _normalised_module_hash(str(a)) == _normalised_module_hash(str(b))


def test_closure_captures_a_cross_module_helper_reached_only_by_import():
    """The exact regression: selectivity's DAG dependency is curate, but its behaviour depends on
    features.compute_features. Before the fix, editing featurisation invalidated nothing here."""
    closure = _first_party_closure("medchem.models.selectivity")
    assert "medchem.features.featurize" in closure
    assert "medchem.models.selectivity" in closure


def test_closure_captures_imports_deferred_into_function_bodies():
    """This package defers imports to keep optional dependencies optional. A deferred first-party
    import is just as load-bearing, and runtime `vars()` inspection cannot see it -- so the closure
    parses the file as well. `config` reaches `transforms` only from inside a validator."""
    assert "medchem.transforms" in _first_party_closure("medchem.config")


def test_closure_contains_only_resolvable_modules():
    """`from x import y` yields the symbol name too; unresolvable entries would hash as no-source and
    could mask a real dependency."""
    for mod in ("medchem.models.selectivity", "medchem.generative.stage", "medchem.vls.stage"):
        for dep in _first_party_closure(mod):
            assert _module_file(dep) is not None, f"{dep} in {mod}'s closure does not resolve"


def test_closure_does_not_pull_in_the_whole_package():
    """If every stage's closure were the entire package, the fix would trade one wrong behaviour for
    blanket invalidation -- correct but useless on a project where a run costs CPU-hours."""
    pull = _first_party_closure("medchem.data.pull")
    assert "medchem.vls.screen" not in pull
    assert "medchem.generative.design" not in pull


def test_hash_source_is_deterministic_and_stage_specific():
    from medchem.data.pull import data_pull
    from medchem.models.selectivity import selectivity

    assert hash_source(selectivity) == hash_source(selectivity)
    assert hash_source(selectivity) != hash_source(data_pull)
