"""Config-driven optional-stage composition (ADR 0006)."""

from __future__ import annotations

from pathlib import Path

import pytest

import medchem.stages  # noqa: F401  (import registers all stages)
from medchem.pipeline import runner


def test_full_pipeline_has_all_stages():
    names = [s.name for s in runner.plan("jak1")]
    assert {"curate", "featurize", "qsar", "selectivity", "evaluate", "generative"} <= set(names)


def test_disable_optional_stages():
    # a single-target project drops selectivity — and its dependents (generation + vls,
    # both of which consume the selectivity model) must be dropped alongside it.
    reduced = [s.name for s in runner.plan("jak1", disable=["selectivity", "generative", "vls"])]
    assert "selectivity" not in reduced and "generative" not in reduced and "vls" not in reduced
    # the potency core survives
    assert {"curate", "featurize", "qsar", "evaluate"} <= set(reduced)


def test_selectivity_is_genuinely_optional():
    """Disabling selectivity alone must work, and must not silently drop its consumers.

    This test previously asserted the OPPOSITE -- that dropping selectivity while keeping the
    screening stage was a hard error -- because both consumers hard-required it. The README meanwhile
    advertised single-target composition. The dependency is now declared optional, so the capability
    and the documentation agree.
    """
    kept = {st.name for st in runner.plan("jak1", disable=["selectivity"])}
    assert "selectivity" not in kept
    # the consumers survive rather than being pruned or erroring
    assert {"vls", "generative"} <= kept, (
        "screening and generative must still plan without selectivity"
    )
    assert {"curate", "featurize", "qsar"} <= kept


def test_optional_dep_is_ordered_before_its_consumer_when_present():
    order = [st.name for st in runner.plan("jak1")]
    assert order.index("selectivity") < order.index("vls")
    assert order.index("selectivity") < order.index("generative")


def test_required_dep_still_blocks_disabling():
    """Optional deps must not weaken the guarantee for REQUIRED ones."""
    import pytest

    with pytest.raises(ValueError, match="requires it"):
        runner.plan("jak1", disable=["qsar"])


def test_disable_depended_on_stage_errors():
    # qsar/evaluate depend on featurize -> disabling it is an error, not silent breakage
    with pytest.raises(ValueError):
        runner.plan("jak1", disable=["featurize"])


# ---------------------------------------------------------------------------------------------
# Target-neutrality. These are the tests that make "retargeting is a config change" checkable
# rather than asserted: they run the front half against a SYNTHETIC non-kinase target and assert
# that no artifact key, filename, or metric key carries terminology from the flagship case study.
# ---------------------------------------------------------------------------------------------

JAK_TERMS = ("jak", "tyk2", "alljak")


def test_no_stage_is_registered_under_a_target_specific_pipeline():
    from medchem.pipeline.stage import _REGISTRY

    for name in _REGISTRY:
        assert not any(t in name.lower() for t in JAK_TERMS), (
            f"pipeline {name!r} is named after a specific target; a pipeline is a shape, "
            f"the target is data"
        )


def test_curate_artifact_contract_is_target_neutral(tmp_path):
    """Artifact keys and filenames are the contract. If they name a target, retargeting is a rename."""
    import pandas as pd

    from medchem.data.curate import curate
    from medchem.pipeline.stage import StageContext, StageResult

    # a synthetic two-target panel with no kinase involved
    raw = tmp_path / "raw_ACME1.csv"
    pd.DataFrame({
        "canonical_smiles": ["CCO", "CCN", "c1ccccc1", "CCC"],
        "standard_value": [100.0, 250.0, 900.0, 40.0],
        "standard_units": ["nM"] * 4,
        "standard_type": ["IC50"] * 4,
        "standard_relation": ["="] * 4,
        "document_year": [2020, 2021, 2019, 2022],
        "molecule_chembl_id": ["C1", "C2", "C3", "C4"],
    }).to_csv(raw, index=False)

    class _Data:
        activity_types = ["IC50"]
        primary = "ACME1"
        targets = {"ACME1": "T1", "ACME2": "T2"}

    class _Cfg:
        data = _Data()

    ctx = StageContext(
        config=_Cfg(),
        workdir=str(tmp_path / "work"),
        upstream={"data_pull": StageResult(name="data_pull", outputs={"ACME1": str(raw)})},
    )
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)
    res = curate(ctx)

    for key, path in res.outputs.items():
        assert not any(t in key.lower() for t in JAK_TERMS), f"artifact key {key!r} names a target"
        fname = Path(path).name.lower()
        assert not any(t in fname for t in JAK_TERMS), f"artifact filename {fname!r} names a target"
    for key in res.metrics:
        assert not any(t in key.lower() for t in JAK_TERMS), f"metric key {key!r} names a target"

    # and it actually produced the primary training set for the configured target
    assert "potency_training" in res.outputs
    assert res.metrics.get("primary_target") == "ACME1"


def test_build_selectivity_uses_configured_targets_not_constants():
    """Delta column names must derive from the configured primary, not a hardcoded panel."""
    import pandas as pd

    from medchem.data.curate import build_selectivity

    df = pd.DataFrame({
        "canonical_smiles": ["CCO", "CCO", "CCN", "CCN"],
        "target": ["ACME1", "ACME2", "ACME1", "ACME2"],
        "pIC50": [8.0, 6.0, 7.5, 7.0],
    })
    out = build_selectivity(df, primary="ACME1", comparators=["ACME2"])
    assert "delta_ACME1_ACME2" in out.columns
    assert "delta_min_vs_comparators" in out.columns
    assert not any("JAK" in c for c in out.columns)


def test_legacy_target_slug_resolves_to_the_generic_pipeline():
    """Documented commands say `-p jak1`; stages register under `discovery`. The alias must keep
    both working and produce the IDENTICAL graph, or published instructions silently break."""
    from medchem.pipeline.stage import PIPELINE_ALIASES, resolve_pipeline

    assert resolve_pipeline("jak1") == "discovery"
    assert resolve_pipeline("discovery") == "discovery"      # not double-resolved
    assert PIPELINE_ALIASES["jak1"] == "discovery"

    assert [s.name for s in runner.plan("jak1")] == [s.name for s in runner.plan("discovery")]


def test_unknown_pipeline_name_is_an_error_not_an_empty_graph():
    """An empty plan would look like a successful no-op run."""
    with pytest.raises((KeyError, ValueError)):
        runner.plan("no-such-pipeline")


def test_documented_cli_invocation_accepts_the_legacy_slug():
    """Covered at the CLI, not just the runner. ``runner.plan`` resolved the alias while the CLI
    validated the raw name against the registry first, so every documented ``-p jak1`` command
    exited 2 -- a break the runner-level test could not see."""
    from typer.testing import CliRunner

    from medchem.cli import app

    cli = CliRunner()
    cfg = str(Path(__file__).resolve().parent.parent / "configs" / "jak1.yaml")

    legacy = cli.invoke(app, ["run", "-p", "jak1", "-c", cfg, "--list"])
    generic = cli.invoke(app, ["run", "-p", "discovery", "-c", cfg, "--list"])
    assert legacy.exit_code == 0, legacy.output
    assert generic.exit_code == 0, generic.output
    assert legacy.output == generic.output          # identical graph, not merely both accepted
    assert "curate" in legacy.output

    bogus = cli.invoke(app, ["run", "-p", "nope", "-c", cfg, "--list"])
    assert bogus.exit_code != 0                     # unknown names still rejected
