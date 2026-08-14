"""End-to-end smoke test of the demo pipeline (also run in CI)."""

from __future__ import annotations

import json
from pathlib import Path

import medchem.stages  # noqa: F401  (registers the demo pipeline)
from medchem.config import Config, DemoConfig
from medchem.pipeline import runner
from medchem.pipeline.stage import list_pipelines


def test_pipelines_registered():
    pipelines = list_pipelines()
    assert "demo" in pipelines
    assert "discovery" in pipelines


def test_demo_pipeline_runs(tmp_path):
    cfg = Config(demo=DemoConfig(n=4))
    results = runner.run(
        "demo",
        cfg,
        workdir=str(tmp_path / "work"),
        cache_dir=str(tmp_path / "cache"),
    )
    assert [r.name for r in results] == ["seed", "double"]

    double = next(r for r in results if r.name == "double")
    values = json.loads(Path(double.outputs["values"]).read_text(encoding="utf-8"))
    assert values == [0, 2, 4, 6]
    assert double.metrics["sum"] == 12
