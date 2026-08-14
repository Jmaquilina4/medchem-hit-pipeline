"""Evaluation harness: a gated robustness report for the primary-target potency model.

Turns the one-off v1 robustness check into a *system* that runs on every
pipeline invocation and emits a report (``eval_report.md`` + ``.json``):

- Baseline RF vs XGB on a random split (R²/MAE) — RF matches the shipped qsar model
- Y-scramble null (shuffled labels -> R² should collapse to ~0)
- Scaffold-CV vs random split (the honest generalization drop)
- **Temporal / near-prospective split** (train pre-cutoff-year -> predict post-cutoff) —
  the honest replacement for what v1 mislabelled "external validation"
- Class balance: inverse-scaffold-frequency weighting vs one-per-scaffold balancing
- Applicability domain: test error binned by max NN-Tanimoto to train
- Leakage: NN-Tanimoto distribution + exact/scaffold overlap between train and test
- Baselines: mean predictor and Morgan+Ridge

Each configured gate (``config.eval.gates``) is checked; ``gate_status`` is ``fail`` if
any hard gate fails, which the CLI turns into a non-zero exit for CI.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medchem.pipeline.stage import StageContext, StageResult, stage


def _reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _rf(seed: int, n_estimators: int = 400) -> Any:
    """RF matching the shipped qsar model (400 trees by default)."""
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)


def _nn_tanimoto(fp_test: np.ndarray, fp_train: np.ndarray) -> np.ndarray:
    """Max Tanimoto of each test fingerprint to any train fingerprint (bit vectors)."""
    inter = fp_test @ fp_train.T
    a = fp_test.sum(axis=1, keepdims=True)
    b = fp_train.sum(axis=1)[None, :]
    union = a + b - inter
    tan = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return tan.max(axis=1)


@stage("discovery", "evaluate", deps=("featurize", "qsar"), config_keys=("eval", "model", "seed"))
def evaluate(ctx: StageContext) -> StageResult:
    """Run the gated robustness suite on the primary-target potency model."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, train_test_split
    from xgboost import XGBRegressor

    seed = int(getattr(ctx.config, "seed", 42))
    n_bits = ctx.config.features.n_bits
    gates = ctx.config.eval.gates
    pot_cfg = ctx.config.model.potency
    rf_trees = pot_cfg.rf_n_estimators
    xp = pot_cfg.xgb

    data = np.load(ctx.upstream["featurize"].outputs["features"])
    x, y = data["X"], data["y"]
    fp = x[:, :n_bits]
    meta = pd.read_csv(ctx.upstream["featurize"].outputs["meta"])
    scaf = meta["scaffold"].fillna("").to_numpy()

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)

    # 1. Baseline RF vs XGB (random split)
    rf = _rf(seed, rf_trees).fit(x[tr], y[tr])
    # The CONFIGURED potency XGB, not a second copy of its defaults. These five values were repeated
    # here as literals, so a config that tuned `model.potency.xgb` trained one estimator in `qsar`
    # and evaluated a differently-parameterised one here -- and the report named it "xgb" either way.
    xgb = XGBRegressor(
        n_estimators=xp.n_estimators, max_depth=xp.max_depth, learning_rate=xp.learning_rate,
        subsample=xp.subsample, colsample_bytree=xp.colsample_bytree,
        n_jobs=-1, random_state=seed,
    ).fit(x[tr], y[tr])
    random_split = {"rf": _reg(y[te], rf.predict(x[te])), "xgb": _reg(y[te], xgb.predict(x[te]))}

    # 2. Y-scramble null
    rng = np.random.default_rng(seed)
    y_scr = y[tr].copy()
    rng.shuffle(y_scr)
    y_scramble_r2 = _reg(y[te], _rf(seed, rf_trees).fit(x[tr], y_scr).predict(x[te]))["r2"]

    # 3. Scaffold-CV (RF)
    n_groups = len(set(scaf.tolist()))
    n_splits = min(5, n_groups)
    scaffold_cv: dict[str, Any] = {"n_splits": n_splits}
    if n_groups >= 2:
        oof = np.full(len(y), np.nan)
        fold_r2: list[float] = []
        for f_tr, f_te in GroupKFold(n_splits=n_splits).split(x, y, groups=scaf):
            p = _rf(seed, rf_trees).fit(x[f_tr], y[f_tr]).predict(x[f_te])
            oof[f_te] = p
            fold_r2.append(_reg(y[f_te], p)["r2"])
        scaffold_cv = {**_reg(y, oof), "fold_r2_mean": float(np.mean(fold_r2)),
                       "fold_r2_std": float(np.std(fold_r2)), "n_splits": n_splits}

    # 3b. Temporal / near-prospective split (replaces the cut "external validation").
    year = (
        pd.to_numeric(meta["document_year"], errors="coerce").to_numpy()
        if "document_year" in meta.columns else np.full(len(y), np.nan)
    )
    temporal: dict[str, Any] | None = None
    cutoff = ctx.config.eval.temporal_cutoff_year
    if cutoff is not None and np.isfinite(year).any():
        cutoff = int(cutoff)
        tr_t = np.where(year < cutoff)[0]
        te_t = np.where(year >= cutoff)[0]

        # LABELS, not just rows, must respect the cutoff. `y` is the median over ALL years, so a
        # compound first reported in 2015 and re-measured in 2023 would train on a label informed by
        # 2023. Measured before this fix: 8.2% of JAK1 and 5.5% of BRD4 training compounds. When
        # curation supplies era-split labels, training uses the PRE-cutoff median instead.
        y_train = y.copy()
        label_source = "all-years median (LEAKY: no era labels supplied by curation)"
        if "pIC50_pre" in meta.columns:
            pre = pd.to_numeric(meta["pIC50_pre"], errors="coerce").to_numpy()
            usable = np.isfinite(pre)
            y_train = np.where(usable, pre, np.nan)
            # A training compound with no pre-cutoff measurement cannot be used: its only evidence is
            # post-cutoff, so including it would be the leak in a different shape.
            tr_t = np.array([i for i in tr_t if usable[i]], dtype=int)
            label_source = "pre-cutoff median (era-split labels from curation)"

        if len(tr_t) >= 50 and len(te_t) >= 20:
            pred_t = _rf(seed, rf_trees).fit(x[tr_t], y_train[tr_t]).predict(x[te_t])
            temporal = {**_reg(y[te_t], pred_t), "cutoff_year": cutoff,
                        "n_train": int(len(tr_t)), "n_test": int(len(te_t)),
                        "train_label_source": label_source,
                        "n_train_dropped_no_pre_label": int(
                            len(np.where(year < cutoff)[0]) - len(tr_t)),
                        }

    # 4. Class balance: inverse-scaffold-frequency weighting vs one-per-scaffold
    scaf_tr = [str(s) for s in scaf[tr]]
    counts = Counter(scaf_tr)
    weights = np.array([1.0 / counts[s] for s in scaf_tr])
    weighted = _reg(y[te], _rf(seed, rf_trees).fit(x[tr], y[tr], sample_weight=weights).predict(x[te]))
    seen: set[str] = set()
    bal_idx = [int(i) for i, s in zip(tr, scaf_tr, strict=True) if not (s in seen or seen.add(s))]
    balanced = _reg(y[te], _rf(seed, rf_trees).fit(x[bal_idx], y[bal_idx]).predict(x[te]))
    class_balance = {"weighted": weighted, "balanced_one_per_scaffold": balanced,
                     "n_weighted": len(tr), "n_balanced": len(bal_idx)}

    # 5. Applicability domain + 6. leakage (test vs train NN-Tanimoto)
    nn = _nn_tanimoto(fp[te], fp[tr])
    abs_err = np.abs(y[te] - rf.predict(x[te]))
    bins = [(0.9, 1.01, ">0.9"), (0.7, 0.9, "0.7-0.9"), (0.5, 0.7, "0.5-0.7"),
            (0.3, 0.5, "0.3-0.5"), (-0.01, 0.3, "<0.3")]
    ad_curve = []
    for lo, hi, label in bins:
        m = (nn >= lo) & (nn < hi)
        ad_curve.append({"bin": label, "n": int(m.sum()),
                         "mae": float(abs_err[m].mean()) if m.any() else None})
    train_scaf = set(scaf_tr)
    leakage = {
        "nn_tanimoto_median": float(np.median(nn)),
        "nn_tanimoto_p95": float(np.quantile(nn, 0.95)),
        "exact_dupes": int((nn >= 0.999).sum()),
        "test_scaffold_overlap_frac": float(np.mean([str(s) in train_scaf for s in scaf[te]])),
    }

    # 7. Baselines
    baselines = {
        "mean_predictor": _reg(y[te], np.full(len(te), y[tr].mean())),
        "morgan_ridge": _reg(y[te], Ridge(alpha=1.0).fit(fp[tr], y[tr]).predict(fp[te])),
    }

    # Hard gates: protect the metrics we trust (leakage is reported, not gated —
    # analog-dense data makes random-split near-dups unavoidable and uninformative).
    gate_results: dict[str, dict[str, Any]] = {}
    if "r2" in scaffold_cv:
        thr = gates.scaffold_cv_r2_min
        gate_results["scaffold_cv_r2_min"] = {
            "threshold": thr, "value": scaffold_cv["r2"], "pass": scaffold_cv["r2"] >= thr}
    gate_results["y_scramble_r2_max"] = {
        "threshold": gates.y_scramble_r2_max, "value": y_scramble_r2,
        "pass": y_scramble_r2 <= gates.y_scramble_r2_max}
    # Optional, and only now actually reachable: this branch tested membership in a dict, but
    # `eval.gates` is a strict model that had no such field, so a config could not set it and the
    # gate could never fire. Declared in EvalGates as `None`-by-default, which is what "off" means.
    if gates.leakage_max_exact_dupes is not None:
        thr = gates.leakage_max_exact_dupes
        gate_results["leakage_max_exact_dupes"] = {
            "threshold": thr, "value": leakage["exact_dupes"], "pass": leakage["exact_dupes"] <= thr}
    gate_status = "pass" if all(g["pass"] for g in gate_results.values()) else "fail"

    metrics: dict[str, Any] = {
        "n_compounds": int(len(y)),
        "random_split": random_split,
        "y_scramble_r2": y_scramble_r2,
        "scaffold_cv": scaffold_cv,
        "temporal_split": temporal,
        "class_balance": class_balance,
        "applicability_domain": ad_curve,
        "leakage": leakage,
        "baselines": baselines,
        "gates": gate_results,
        "gate_status": gate_status,
    }
    out = Path(ctx.workdir)
    (out / "eval_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # A HEADING for the markdown report, not a result. Written as an explicit cascade rather than
    # `primary or target or "..."`: that is the same shape as a config value silently overriding
    # another, the regression guard in tests/test_config.py cannot tell the two apart structurally, and
    # an exemption for "this one is only a label" is exactly the kind of exemption that later covers
    # something that is not.
    if ctx.config.data.primary:
        target_label = str(ctx.config.data.primary)
    elif ctx.config.target:
        target_label = str(ctx.config.target)
    else:
        target_label = "Primary target"
    (out / "eval_report.md").write_text(
        _render_report(metrics, target_label=target_label), encoding="utf-8")
    return StageResult(
        name="evaluate",
        outputs={"report_json": str(out / "eval_report.json"), "report_md": str(out / "eval_report.md")},
        metrics={"gate_status": gate_status, "y_scramble_r2": y_scramble_r2},
        gate_status=gate_status,
    )


def _render_report(m: dict[str, Any], target_label: str = "Primary target") -> str:
    """Render the markdown report. The title takes the TARGET from config: it was hardcoded to the
    flagship target, so a second target's report was titled with the first target's name."""
    rs = m["random_split"]
    sc = m["scaffold_cv"]
    cb = m["class_balance"]
    wt = cb["weighted"]
    bal = cb["balanced_one_per_scaffold"]
    lk = m["leakage"]
    sc_row = (
        f"| RF (scaffold-CV) | {sc['r2']:.3f} | {sc['mae']:.3f} | {sc['rmse']:.3f} |"
        if "r2" in sc else "| RF (scaffold-CV) | n/a (single scaffold group) | — | — |"
    )
    lines = [
        f"# {target_label} potency QSAR — Evaluation Report",
        "",
        f"Compounds: **{m['n_compounds']}** · Gate status: **{m['gate_status'].upper()}**",
        "",
        "## Random split vs scaffold-CV",
        "| Model / split | R² | MAE | RMSE |",
        "|---|---|---|---|",
        f"| RF (random) | {rs['rf']['r2']:.3f} | {rs['rf']['mae']:.3f} | {rs['rf']['rmse']:.3f} |",
        f"| XGB (random) | {rs['xgb']['r2']:.3f} | {rs['xgb']['mae']:.3f} | {rs['xgb']['rmse']:.3f} |",
        sc_row,
        f"| **Y-scramble null** | {m['y_scramble_r2']:.3f} | — | — |",
        "",
    ]
    tsp = m.get("temporal_split")
    if tsp:
        lines += [
            f"## Temporal / near-prospective split (train < {tsp['cutoff_year']} → predict ≥ {tsp['cutoff_year']})",
            f"- n_train **{tsp['n_train']}** → n_test **{tsp['n_test']}** · "
            f"**R² {tsp['r2']:.3f}**, MAE {tsp['mae']:.3f}, RMSE {tsp['rmse']:.3f}",
            "- _Honest replacement for what v1 mislabelled 'external validation': "
            "a genuine forward-in-time hold-out._",
            "",
        ]
    lines += [
        "## Class balance",
        "| Training | R² | MAE |",
        "|---|---|---|",
        f"| Weighted (inv-scaffold-freq) | {wt['r2']:.3f} | {wt['mae']:.3f} |",
        f"| Balanced (1/scaffold, n={cb['n_balanced']}) | {bal['r2']:.3f} | {bal['mae']:.3f} |",
        "",
        "## Applicability domain (test error by NN-Tanimoto to train)",
        "| Similarity bin | n | MAE |",
        "|---|---|---|",
    ]
    for b in m["applicability_domain"]:
        mae = f"{b['mae']:.3f}" if b["mae"] is not None else "—"
        lines.append(f"| {b['bin']} | {b['n']} | {mae} |")
    lines += [
        "",
        "## Leakage check (train vs test)",
        f"- NN-Tanimoto median: **{lk['nn_tanimoto_median']:.3f}**, 95th pct: **{lk['nn_tanimoto_p95']:.3f}**",
        f"- Exact near-duplicates (train-test): **{lk['exact_dupes']}**",
        f"- Test scaffolds also present in train: **{lk['test_scaffold_overlap_frac']:.1%}**",
        "- _Analog-dense dataset: random-split R² is an optimistic upper bound; "
        "the scaffold-CV R² is the leakage-free generalization number._",
        "",
        "## Baselines",
        f"- Mean predictor R²: {m['baselines']['mean_predictor']['r2']:.3f}",
        f"- Morgan+Ridge R²: {m['baselines']['morgan_ridge']['r2']:.3f}",
        "",
        "## Gates",
    ]
    for name, g in m["gates"].items():
        lines.append(f"- {name}: value={g['value']}, threshold={g['threshold']} → {'PASS' if g['pass'] else 'FAIL'}")
    return "\n".join(lines) + "\n"
