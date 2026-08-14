"""A tiny, dependency-only demo pipeline used by tests and the CI smoke run.

It exercises the runner, the stage contract, and content-addressed caching with
zero heavy dependencies, so ``medchem run --pipeline demo`` and CI stay fast.
"""

from __future__ import annotations

import json
from pathlib import Path

from medchem.pipeline.stage import StageContext, StageResult, stage


@stage("demo", "seed", config_keys=("demo", "seed"))
def seed(ctx: StageContext) -> StageResult:
    """Emit a deterministic list of integers [0, n). Records the global seed."""
    n = ctx.config.demo.n
    values = list(range(n))
    out = Path(ctx.workdir) / "seed.json"
    out.write_text(json.dumps(values), encoding="utf-8")
    # reads ctx.config.seed so the declared "seed" config_key is genuine
    return StageResult(name="seed", outputs={"values": str(out)}, metrics={"count": n, "seed": ctx.config.seed})


@stage("demo", "double", deps=("seed",), config_keys=("demo",))
def double(ctx: StageContext) -> StageResult:
    """Double each seeded integer."""
    seed_path = ctx.upstream["seed"].outputs["values"]
    values = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    doubled = [v * 2 for v in values]
    out = Path(ctx.workdir) / "double.json"
    out.write_text(json.dumps(doubled), encoding="utf-8")
    return StageResult(name="double", outputs={"values": str(out)}, metrics={"sum": sum(doubled)})
