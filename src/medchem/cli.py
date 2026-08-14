"""medchem command-line interface — the single entrypoint the skills layer orchestrates.

Design rule: the CLI (and everything under ``src/``) is deterministic and makes no
LLM calls. An optional skills layer may call these commands, parse the
``--json`` output, and add procedure + judgment on top.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Annotated

import typer

from medchem import __version__
from medchem.config import load_config
from medchem.pipeline import runner
from medchem.pipeline.stage import PIPELINE_ALIASES, list_pipelines, resolve_pipeline

app = typer.Typer(
    add_completion=False,
    help="medchem — reproducible, config-driven pipeline for bioactivity modelling and compound "
         "prioritisation. Evaluated on a kinase (JAK1) and a bromodomain (BRD4).",
)


@app.command()
def version() -> None:
    """Print the installed medchem version."""
    typer.echo(__version__)


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Target config YAML.")],
    pipeline: Annotated[str, typer.Option("--pipeline", "-p", help="Pipeline name.")] = "discovery",
    stage: Annotated[list[str] | None, typer.Option("--stage", help="Run only these stage(s).")] = None,
    from_stage: Annotated[str | None, typer.Option("--from", help="Run from this stage onward.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Ignore cache and re-run.")] = False,
    workdir: Annotated[Path, typer.Option("--workdir", help="Base output directory.")] = Path("runs"),
    list_dag: Annotated[bool, typer.Option("--list", help="Print the topo-sorted stage graph and exit.")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output for skills.")] = False,
) -> None:
    """Run a pipeline end-to-end with content-addressed caching.

    Cached stages are loaded from cache and skipped unless ``--force`` is given
    (this applies to targeted ``--stage`` / ``--from`` runs too).
    """
    import medchem.stages  # noqa: F401  (registers all stages as a side effect)

    cfg = load_config(config)
    # Resolve legacy target-named slugs (`-p jak1`) to the generic pipeline BEFORE validating, or
    # every published command line breaks the moment stages are registered generically.
    pipeline = resolve_pipeline(pipeline)
    if pipeline not in list_pipelines():
        available = sorted({*list_pipelines(), *PIPELINE_ALIASES})
        raise typer.BadParameter(f"unknown pipeline {pipeline!r}; available: {available}")

    disable = list(cfg.disable_stages)  # typed field; the runner rejects names it cannot honour

    if list_dag:
        for i, st in enumerate(runner.plan(pipeline, disable), 1):
            deps = f"  <- {', '.join(st.deps)}" if st.deps else ""
            typer.echo(f"  {i}. {st.name}{deps}")
        return

    try:
        results = runner.run(
            pipeline,
            cfg,
            workdir=str(workdir / cfg.target),
            only=stage or None,
            from_stage=from_stage,
            force=force,
            disable=disable,
        )
    except NotImplementedError as exc:
        typer.secho(f"[not-implemented] {exc}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=3) from exc  # 3 = Phase-2 stub (distinct from 2 = usage error)

    if json_out:
        payload = [
            {
                "stage": r.name,
                "from_cache": r.from_cache,
                "metrics": r.metrics,
                "gate_status": r.gate_status,
                "cache_key": r.cache_key,
            }
            for r in results
        ]
        typer.echo(_json.dumps(payload, indent=2))
    else:
        for r in results:
            tag = "cache" if r.from_cache else "ran"
            typer.echo(f"  [{tag:>5}] {r.name}  {r.metrics}")

    # CI enforcement: a stage that failed its eval gate makes the run fail.
    failed = [r.name for r in results if r.gate_status == "fail"]
    if failed:
        typer.secho(f"[gate-fail] stages failed their gates: {failed}", fg=typer.colors.RED)
        raise typer.Exit(code=4)


@app.command("eval")
def eval_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Target config YAML.")],
) -> None:
    """Deprecated: evaluation runs as a stage of `medchem run`, not as a separate command."""
    typer.secho("eval: not implemented as a standalone command; use `medchem run`.",
                fg=typer.colors.YELLOW)
    raise typer.Exit(code=3)


@app.command()
def generalize(
    target: Annotated[str, typer.Argument(help="Second-target name, e.g. p38a.")],
) -> None:
    """Deprecated: retargeting is a config change, so there is nothing for a command to do."""
    typer.secho(
        f"generalize {target!r}: retargeting is a config change — see configs/ for the four "
        f"frozen panels.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=3)


if __name__ == "__main__":
    app()
