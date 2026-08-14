"""An optional dependency must actually be DELIVERED, not merely ordered.

Two commits made `selectivity` an optional dependency: the graph composed without it, consumers
dropped its scoring component rather than scoring zero, and tests covered all of that. None of it
worked. The runner built each stage's context from `st.deps` alone, so
`ctx.upstream.get("selectivity")` was **always** None and every consumer took the selectivity-absent
path even when the stage had run and produced a model. The same omission kept the optional upstream out
of the cache key, so a changed selectivity model would not invalidate its consumers.

It surfaced as a crash in the screening stage -- the fifth instance in this project of code calling an
optional predictor unconditionally -- and only because that stage was enabled for one target.

These tests assert delivery and invalidation, which the composition tests could not see.
"""

from __future__ import annotations

import pytest

from medchem.config import Config
from medchem.pipeline import runner
from medchem.pipeline.stage import _REGISTRY, StageContext, StageResult, stage

PIPE = "optdep_probe"


@pytest.fixture
def probe_pipeline(tmp_path):
    """A throwaway pipeline whose consumer records exactly what it was handed."""
    seen: dict[str, list[str]] = {}

    @stage(PIPE, "producer", config_keys=("demo",))
    def producer(ctx: StageContext) -> StageResult:
        p = f"{ctx.workdir}/produced.txt"
        with open(p, "w") as fh:
            fh.write("payload")
        return StageResult(name="producer", outputs={"model": p})

    @stage(PIPE, "consumer", deps=(), optional_deps=("producer",), config_keys=("demo",))
    def consumer(ctx: StageContext) -> StageResult:
        seen["upstream"] = sorted(ctx.upstream)
        p = f"{ctx.workdir}/consumed.txt"
        with open(p, "w") as fh:
            fh.write(",".join(sorted(ctx.upstream)))
        return StageResult(name="consumer", outputs={"out": p})

    yield seen, tmp_path
    _REGISTRY.pop(PIPE, None)


def test_optional_dependency_is_delivered_to_the_consumer(probe_pipeline):
    seen, tmp_path = probe_pipeline
    runner.run(PIPE, Config(), workdir=str(tmp_path / "w"), cache_dir=str(tmp_path / "c"))
    assert seen["upstream"] == ["producer"], (
        "the optional upstream never reached ctx.upstream -- consumers cannot use what they are not given"
    )


def test_disabled_optional_dependency_is_absent_rather_than_empty(probe_pipeline):
    seen, tmp_path = probe_pipeline
    runner.run(PIPE, Config(), workdir=str(tmp_path / "w"), cache_dir=str(tmp_path / "c"),
               disable=["producer"])
    assert seen["upstream"] == []


def test_optional_dependency_participates_in_the_consumer_cache_key(probe_pipeline):
    """Otherwise a consumer keeps a cached result computed without the optional upstream, even after
    that upstream appears."""
    seen, tmp_path = probe_pipeline
    without = runner.run(PIPE, Config(), workdir=str(tmp_path / "w1"), cache_dir=str(tmp_path / "c1"),
                         disable=["producer"])
    with_it = runner.run(PIPE, Config(), workdir=str(tmp_path / "w2"), cache_dir=str(tmp_path / "c2"))
    key_without = next(r.cache_key for r in without if r.name == "consumer")
    key_with = next(r.cache_key for r in with_it if r.name == "consumer")
    assert key_without != key_with


def test_screen_library_tolerates_a_missing_selectivity_model():
    """The crash site. An absent predictor must leave the annotation NaN -- not 0.0, which would be a
    fabricated measurement that also feeds the priority rules."""
    pytest.importorskip("rdkit")
    import numpy as np

    from medchem.vls.screen import Thresholds, screen_library

    records = [{"smiles": s, "id": f"z{i}"} for i, s in enumerate(
        ["c1ccccc1C(=O)O", "c1ccncc1", "CCOc1ccccc1", "CC(=O)Nc1ccccc1"])]
    train_fp = np.zeros((3, 2048), dtype=np.float32)
    train_fp[:, :30] = 1
    res = screen_library(
        records,
        potency_predict=lambda x: np.full(len(x), 8.5),
        selectivity_predict=None,
        train_fp=train_fp,
        active_fp=None, known_ref_fp=None, known_ref_names=[],
        thresholds=Thresholds(),
    )
    deltas = np.asarray(res.columns["pred_selectivity_delta"], dtype=float)
    parsed = np.asarray(res.columns["parsed"], dtype=bool) if "parsed" in res.columns else None
    subject = deltas[parsed] if parsed is not None else deltas
    assert subject.size and np.all(np.isnan(subject)), (
        "absent selectivity must read as NaN, never as a real 0.0"
    )
    # the selectivity-gated priorities must be unreachable when selectivity is unknown
    assert res.priority.get("fep_first", 0) == 0, (
        "a NaN selectivity must not satisfy a selectivity threshold"
    )


def test_vls_stage_receives_and_uses_the_selectivity_model(tmp_path):
    """The end-to-end check that a narrower version of this test cannot make.

    Hand-loading the selectivity model and calling the scorer
    directly, which proves only that the scorer works. It says nothing about what the STAGE is handed,
    and the stage was still being handed None. This exercises the stage function itself, both with the
    optional upstream present and absent, and asserts the annotation differs between the two.
    """
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    from medchem.features.featurize import compute_features
    from medchem.vls.stage import vls

    smiles = ["c1ccccc1C(=O)O", "c1ccncc1", "CCOc1ccccc1", "CC(=O)Nc1ccccc1", "c1ccc2[nH]ccc2c1"]
    fp, desc, _s, keep = compute_features(smiles)
    x = np.hstack([fp, desc])
    kept = [s for s, k in zip(smiles, keep, strict=True) if k]

    lib = tmp_path / "lib.smi"
    lib.write_text("\n".join(f"{s} z{i}" for i, s in enumerate(kept)) + "\n")

    train_csv = tmp_path / "potency_training.csv"
    pd.DataFrame({"canonical_smiles": kept, "pIC50": np.linspace(6, 9, len(kept))}).to_csv(
        train_csv, index=False)
    feats = tmp_path / "features.npz"
    np.savez_compressed(feats, X=x)
    pot_path = tmp_path / "potency_model.joblib"
    joblib.dump(RandomForestRegressor(n_estimators=6, random_state=42).fit(x, np.linspace(6, 9, len(kept))),
                pot_path)
    qsar_metrics = tmp_path / "qsar_metrics.json"
    qsar_metrics.write_text(json.dumps({"conformal_rf": {"interval_halfwidth_90": 0.9}}))
    sel_path = tmp_path / "selectivity_model.joblib"
    joblib.dump(RandomForestRegressor(n_estimators=6, random_state=42).fit(x, np.linspace(0, 2, len(kept))),
                sel_path)

    cfg = Config.model_validate({
        "features": {"n_bits": 2048, "radius": 2},
        # Bounds widened to admit these small fixture molecules. The shipped defaults start at
        # MW 300 and would prepare ZERO records from this library, leaving an empty table that
        # trivially "passes" a NaN assertion -- which is exactly what happened on the first attempt.
        "vls": {
            "enabled": True,
            "actives": {"pic50_min": 6.0},
            "library": {
                "path": str(lib),
                "lead_like": {"mw_min": 50, "mw_max": 500, "logp_max": 6.0, "tpsa_max": 200,
                              "hbd_max": 6, "hba_max": 12, "rotb_max": 12},
            },
        },
    })
    base = {
        "curate": StageResult(name="curate", outputs={"potency_training": str(train_csv)}),
        "featurize": StageResult(name="featurize", outputs={"features": str(feats)}),
        "qsar": StageResult(name="qsar", outputs={"model": str(pot_path),
                                                  "metrics": str(qsar_metrics)}),
    }

    def run(upstream, tag):
        work = tmp_path / tag
        work.mkdir()
        res = vls(StageContext(config=cfg, workdir=str(work), upstream=upstream))
        assert res.metrics.get("status") != "skipped", f"{tag}: stage skipped, nothing was tested"
        return res

    with_sel = run({**base, "selectivity": StageResult(
        name="selectivity", outputs={"model": str(sel_path)})}, "with_sel")
    without = run(dict(base), "without_sel")

    def deltas(res):
        p = [v for k, v in res.outputs.items() if str(v).endswith(".csv")]
        assert p, f"no table emitted: {res.outputs}"
        col = pd.read_csv(p[0])["pred_selectivity_delta"]
        return np.asarray(col, dtype=float)

    d_with, d_without = deltas(with_sel), deltas(without)
    assert not np.all(np.isnan(d_with)), (
        "the stage did not use the selectivity model it was given -- the optional upstream is not "
        "reaching the stage body"
    )
    assert np.all(np.isnan(d_without)), "absent selectivity must stay NaN, never a fabricated 0.0"
