"""QSAR stage: RF + XGBoost potency models with honest generalization metrics.

Reports three things, deliberately: a random-split test (optimistic), a
**Bemis-Murcko scaffold cross-validation** (the honest generalization number),
and a split-conformal 90% prediction interval with its empirical coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medchem.pipeline.stage import StageContext, StageResult, stage


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _rf(seed: int, n_estimators: int = 400) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)


@stage("discovery", "qsar", deps=("featurize",), config_keys=("model", "seed"))
def qsar(ctx: StageContext) -> StageResult:
    """Train RF + XGB, report random-split, scaffold-CV, and conformal metrics."""
    from sklearn.model_selection import GroupKFold, train_test_split
    from xgboost import XGBRegressor

    seed = int(getattr(ctx.config, "seed", 42))
    # Typed reads. Dict-punning a modeled section worked only by accident -- `dict(model)` yields
    # submodel INSTANCES, so `.get("potency", {})` returned a PotencyModelConfig that the next
    # `dict()` happened to flatten -- and it silently accepted whatever `xgb` held. Every parameter
    # below is now a declared field of medchem.config.XgbConfig, so a key that is not forwarded here
    # is rejected at load instead of being dropped after reaching the cache key.
    pot = ctx.config.model.potency
    rf_trees = pot.rf_n_estimators
    xp = pot.xgb
    data = np.load(ctx.upstream["featurize"].outputs["features"])
    x, y = data["X"], data["y"]
    meta = pd.read_csv(ctx.upstream["featurize"].outputs["meta"])
    scaffolds = meta["scaffold"].fillna("").to_numpy()

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)

    models: dict[str, Any] = {
        "random_forest": _rf(seed, rf_trees),
        "xgboost": XGBRegressor(
            n_estimators=xp.n_estimators,
            max_depth=xp.max_depth,
            learning_rate=xp.learning_rate,
            subsample=xp.subsample,
            colsample_bytree=xp.colsample_bytree,
            n_jobs=-1, random_state=seed,
        ),
    }
    random_split: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x[tr], y[tr])
        random_split[name] = _reg_metrics(y[te], model.predict(x[te]))

    # Scaffold cross-validation (the honest generalization number) — RF.
    n_groups = len(set(scaffolds))
    scaffold_cv: dict[str, float] = {}
    if n_groups >= 2:
        n_splits = min(5, n_groups)
        oof = np.full(len(y), np.nan)
        fold_r2: list[float] = []
        for f_tr, f_te in GroupKFold(n_splits=n_splits).split(x, y, groups=scaffolds):
            model = _rf(seed, rf_trees)
            model.fit(x[f_tr], y[f_tr])
            pred = model.predict(x[f_te])
            oof[f_te] = pred
            fold_r2.append(_reg_metrics(y[f_te], pred)["r2"])
        scaffold_cv = {
            **_reg_metrics(y, oof),
            "fold_r2_mean": float(np.mean(fold_r2)),
            "fold_r2_std": float(np.std(fold_r2)),
            "n_splits": n_splits,
        }

    # Split-conformal 90% interval (RF): calibrate on a held-out slice of train.
    tr2, cal = train_test_split(tr, test_size=0.25, random_state=seed)
    rf = _rf(seed, rf_trees)
    rf.fit(x[tr2], y[tr2])
    resid = np.abs(y[cal] - rf.predict(x[cal]))
    n_cal = len(resid)
    # Finite-sample-corrected conformal level ceil((n+1)(1-alpha))/n (Vovk et al.);
    # the plain 0.9 quantile undercovers on a finite calibration set.
    q_level = min(1.0, float(np.ceil((n_cal + 1) * 0.9) / n_cal))
    q90 = float(np.quantile(resid, q_level, method="higher"))
    te_pred = rf.predict(x[te])
    conformal = {
        "interval_halfwidth_90": q90,
        "empirical_coverage": float(np.mean(np.abs(y[te] - te_pred) <= q90)),
        "n_calibration": int(n_cal),
        "conformal_level": q_level,
    }

    metrics: dict[str, Any] = {
        "n_compounds": int(len(y)),
        "n_features": int(x.shape[1]),
        "random_split": random_split,
        "scaffold_cv_rf": scaffold_cv,
        "conformal_rf": conformal,
    }
    out = Path(ctx.workdir)
    (out / "qsar_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "y_true": y[te],
            "rf_pred": models["random_forest"].predict(x[te]),
            "xgb_pred": models["xgboost"].predict(x[te]),
        }
    ).to_csv(out / "qsar_test_predictions.csv", index=False)

    # Persist a production potency model (RF fit on ALL data) for downstream scoring/generation.
    import joblib

    joblib.dump(_rf(seed, rf_trees).fit(x, y), out / "potency_model.joblib")

    return StageResult(
        name="qsar",
        outputs={
            "metrics": str(out / "qsar_metrics.json"),
            "predictions": str(out / "qsar_test_predictions.csv"),
            "model": str(out / "potency_model.joblib"),
        },
        metrics=metrics,
    )
