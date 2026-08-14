"""Invariants for the documentation verifier itself.

It is the check that makes documented numbers derived rather than transcribed, and its weak point is
asserting only that the CORRECT value appears somewhere in the docs -- which is compatible with a WRONG
value also appearing, in a table, three sections away. A stale gate table passed exactly that way.

The fix was a superseded-value denylist. These tests guard the two ways that denylist can rot: going
empty, or listing a value that is currently true.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "_verify_docs", REPO / "scripts" / "verify_docs_against_manifests.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_superseded_list_is_populated():
    """An empty denylist would make the check vacuous while still reporting success."""
    v = _verifier()
    assert len(v.SUPERSEDED_METRIC_VALUES) >= 3
    for value, why in v.SUPERSEDED_METRIC_VALUES:
        assert value and why, "every entry needs the value AND why it is superseded"


def test_no_superseded_value_is_also_a_current_value():
    """The denylist must not contain a number that is currently true.

    If it did, the verifier would demand that value appear (the presence check) and forbid it (the
    staleness check) — permanently red, or worse, quietly reconciled by weakening one of them.
    """
    v = _verifier()
    truth = v.resolve()
    current: set[str] = set()
    for t in truth.values():
        current |= {
            f"{t['scaffold_cv_r2']:.4f}", f"{t['y_scramble_r2']:.4f}", f"{t['temporal_r2']:.4f}",
            f"{t['scaffold_cv_r2']:.3f}", f"{t['y_scramble_r2']:.3f}", f"{t['temporal_r2']:.3f}",
            f"{t['compounds']:,}", f"{t['scaffold_overlap']:.1%}",
        }
    for value, _why in v.SUPERSEDED_METRIC_VALUES:
        assert value.lstrip("-−+") not in {c.lstrip("-−+") for c in current}, (
            f"{value!r} is listed as superseded but is a CURRENT value — the verifier would both "
            f"require and forbid it"
        )


def test_historical_context_admits_a_comparison_and_rejects_a_bare_table():
    """The exemption must be narrow enough to still catch a stale row.

    A spec-to-spec comparison legitimately quotes the old number; a gate table does not.
    """
    v = _verifier()
    assert v.HISTORICAL_CONTEXT.search("| scaffold-CV R² | 0.7502 | **0.7283** |\nunder spec 1.1")
    assert not v.HISTORICAL_CONTEXT.search(
        "| scaffold-CV R² | ≥ 0.55 | 0.7597 ✓ | 0.7407 ✓ | 0.7502 ✓ | 0.7523 ✓ |"
    )


@pytest.mark.parametrize("panel", ["jak1", "jak1_sensitivity", "brd4", "brd4_sensitivity"])
def test_every_panel_shares_one_release_identity(panel: str):
    """All four manifests must record one clean revision and the same cohort spec, or 'the frozen
    results' describes four different things.

    Asserted through ``code.single_clean_revision`` rather than by comparing a revision literal. The
    identifier is redacted at publication: a SHA for a workspace that is not published cannot be
    resolved or checked by a reader, so what the records carry is the property it was evidence for.
    """
    v = _verifier()
    t = v.resolve()[panel]
    assert v.FROZEN_SHA is None, "a revision literal must not be reintroduced into the verifier"
    assert t["single_clean_revision"] is True, (
        f"{panel}: published manifest does not record code.single_clean_revision=true"
    )
    assert t["dirty"] is False
    assert t["spec_version"] == v.FROZEN_COHORT_SPEC


def test_no_published_record_or_document_carries_a_revision_identifier():
    """The redaction has to hold across provenance AND prose, or it only moved the disclosure."""
    import re

    offenders = []
    for f in sorted(PROV.rglob("*.json")) if (PROV := REPO / "provenance").is_dir() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        code = d.get("code") if isinstance(d, dict) else None
        if isinstance(code, dict):
            for k in ("git_sha", "git_describe", "manifest_tool_sha"):
                if k in code:
                    offenders.append(f"{f.relative_to(REPO)}: code.{k} is present")
    for doc in [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]:
        if not doc.is_file():
            continue
        for m in re.finditer(r"`([0-9a-f]{7,12})`", doc.read_text(encoding="utf-8")):
            offenders.append(f"{doc.relative_to(REPO)}: cites `{m.group(1)}`")
    assert not offenders, "revision identifiers survive publication:\n  " + "\n  ".join(offenders)


# --- gates that silently stopped checking ------------------------------------------------------------

def _shipped_scripts() -> list[Path]:
    """The scripts the sanitized export publishes.

    In the working tree the exporter's allowlist is authoritative; legacy one-offs under scripts/ are
    excluded from lint and type-checking and at least one does not even parse, so checking all of them
    would fail for reasons that cannot reach a reader. In the EXPORT the exporter is not shipped, and
    every script present is by definition published — so there the scope is everything, which is
    stricter.
    """
    import importlib.util as _iu

    exporter = REPO / "scripts" / "export_public.py"
    if not exporter.is_file():
        return sorted((REPO / "scripts").glob("*.py"))
    spec = _iu.spec_from_file_location("_ep", exporter)
    assert spec is not None and spec.loader is not None
    ep = _iu.module_from_spec(spec)
    spec.loader.exec_module(ep)
    return [REPO / "scripts" / n for n in ep.KEEP_SCRIPTS
            if n.endswith(".py") and (REPO / "scripts" / n).is_file()]


def test_no_published_gate_script_has_unreachable_statements():
    """A `return` mid-function had been disabling half of the documentation verifier.

    `verify_docs_against_manifests.check()` returned at what looked like the end of its last loop, and
    the seven statements after it never ran: the composition-figure assertions, the withdrawn-claim
    regression guards, and two required-disclosure checks. The block's own comment says it exists
    because three of four BRD4 percentages and the JAK1 median gap had once been wrong -- so the guard
    written to stop exactly that recurring was itself inert.

    Nothing had regressed behind it, which is the point: the gate reported success while not performing
    the checks its output implied, and no lint rule enabled here catches unreachable code.
    """
    import ast

    scripts = _shipped_scripts()
    assert len(scripts) >= 8, f"scope too small ({len(scripts)}); the discovery is broken"
    offenders = []
    for path in scripts:
        tree = ast.parse(path.read_text(encoding="utf-8"))       # a shipped script MUST parse
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = node.body
            for i, stmt in enumerate(body[:-1]):
                if isinstance(stmt, ast.Return | ast.Raise | ast.Continue | ast.Break):
                    dead = body[i + 1:]
                    offenders.append(
                        f"{path.name}:{stmt.lineno} {node.name}(): {len(dead)} statement(s) after a "
                        f"{type(stmt).__name__} are unreachable (lines "
                        f"{dead[0].lineno}-{dead[-1].end_lineno})"
                    )
                    break
    assert not offenders, "unreachable code in a published script:\n  " + "\n  ".join(offenders)


def test_frozen_figure_captions_are_derived_not_hardcoded():
    """A published figure asserted a conclusion the results had withdrawn.

    frozen_attrition.png's caption named a supported-pair count that the frozen result contradicts (no
    BET pair is supported), and carried attrition percentages that disagreed with the figure's own bar
    labels and with every document. The documentation verifier reads prose, so a string baked into a
    PNG is invisible to it -- which is why the caption has to come from the same records the bars do.

    Comments are stripped before matching: this test previously failed on its own explanation, which is
    the same mistake as documenting a leak by quoting it.
    """
    import io
    import tokenize

    src = (REPO / "scripts" / "make_frozen_figures.py").read_text(encoding="utf-8")
    code = "".join(
        tok.string if tok.type != tokenize.COMMENT else ""
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
    )
    banned = ("only one pair", "one pair is supported", "\u221272%", "\u221284%")
    hits = [b for b in banned if b in code]
    assert not hits, f"hardcoded in make_frozen_figures.py: {hits}; derive from provenance instead"
    # The caption must actually consult the support decisions and the bar data.
    assert "supported" in code and "kept_a" in code and "kept_b" in code


# ---------------------------------------------------------------------------------------------
# Generated text must be checked as GENERATED. A grammar error split across a source line break
# ("...in a " + "exported tree...") is invisible to every grep over the source, because neither
# fragment contains it -- only the concatenated output does.
# ---------------------------------------------------------------------------------------------

def _generated_identity_text() -> str:
    """Every string the digest tool actually emits, flattened. Not its source."""
    import json
    import subprocess

    repo = Path(__file__).resolve().parent.parent
    out = subprocess.run(["uv", "run", "python", "scripts/scientific_source_digest.py", "--json"],
                         cwd=repo, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    return json.dumps(json.loads(out.stdout))


def test_generated_identity_text_has_no_article_agreement_errors():
    """The concatenation defect, asserted where it is observable."""
    text = _generated_identity_text().lower()
    for bad in ("in a exported", "a exported tree", "a export ", "a analysis", "a identity",
                "a earlier", "a interpreter", "a artifact", "a empty", "a exact"):
        assert bad not in text, f"generated identity text contains {bad!r}"
    assert "in an exported tree" in text, (
        "the corrected phrasing is absent, so this test would pass vacuously if the note were removed"
    )


def test_generated_identity_text_does_not_claim_the_analysis_revision_is_recorded():
    """`analysis_run_sha` is withheld, so its note must not describe it as recorded or pending."""
    text = _generated_identity_text().lower()
    assert "deliberately not recorded" in text
    for bad in ("recorded for completeness", "not yet assigned: the source revision"):
        assert bad not in text, f"generated text still says {bad!r}"
