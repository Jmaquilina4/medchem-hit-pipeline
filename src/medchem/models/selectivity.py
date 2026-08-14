"""Selectivity stage: direct-Δ isoform-selectivity models + direct-vs-potency comparison.

Selectivity analyses, as pipeline output rather than one-off scripts:

- **Per-pair model** — per isoform pair, 5-fold Bemis-Murcko scaffold-CV of a direct-Δ RF
  (chiral ECFP4 + descriptors): R² / MAE / RMSE, PR-AUC for the selective class, top-10%
  enrichment.
- **Direct vs potency-derived** — the de-confounded comparison. The **headline** is
  apples-to-apples: `direct_delta_rf` vs `pot_delta_rf` use the SAME features (fp+desc),
  the SAME model (RF), and the SAME folds — only the target formulation differs
  (predict Δ directly vs predict two potencies and subtract). `pot_delta_xgb_desc` is a
  secondary "even a tuned XGB potency model loses" variant, clearly a different algorithm.

Config-driven: isoform pairs, the selective-Δ threshold, and the RF size all come from
`model.selectivity` / `model.potency`. Δ = pIC50(primary) − pIC50(comparator).

**"Selective" is ONE-SIDED throughout this module**: Δ >= threshold, i.e. more potent on the primary
target. Not |Δ| >= threshold. Selectivity *for* the primary is the design goal, so a compound a log
unit more potent on a comparator is not a weaker success — it is a different programme's compound.
Stated here because the metric is reported far from where it is defined, and a doc table once labelled
the column |Δ|.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medchem.data.curate import DELTA_MIN_COLUMN
from medchem.features.featurize import compute_features
from medchem.pipeline.stage import StageContext, StageResult, stage


def _reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _bootstrap_r2_ci(
    y_true: np.ndarray, y_pred: np.ndarray, *, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for out-of-fold R².

    Resamples the held-out (truth, prediction) pairs — it does NOT refit, so it captures sampling
    variability of the evaluation set and not model-fitting variability. That is the standard way to
    put an interval on a held-out metric, and it is enough to answer the question that matters here:
    could this R² have arisen from noise at this sample size?
    """
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(y_true[idx]) == 0:  # degenerate resample: R² undefined
            continue
        vals.append(r2_score(y_true[idx], y_pred[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _support(delta: np.ndarray, oof: np.ndarray, threshold: float, seed: int) -> dict[str, Any]:
    """Is this pair's selectivity result interpretable, and if not, why not?

    The BRD4 run made the need explicit: BRD4–BRDT returned R² = −0.009 from 177 paired compounds
    with 13 positives and the pipeline reported it with no caveat, because the evaluation gates cover
    potency only. A model cannot learn a distinction the data barely contains, and the failure looks
    identical to a modelling failure unless prevalence is reported alongside.

    Two things are recorded rather than one number:

    * a bootstrap CI on R². **Its limitation matters**: it resamples FIXED out-of-fold predictions
      without refitting, and ignores scaffold-group dependence. It therefore measures row-resampling
      stability of the held-out estimate — NOT that the result could not arise from model-fitting
      noise. A CI excluding zero licenses "positive but uncertain cross-validated estimate", not
      "real signal". Establishing the stronger claim needs repeated grouped CV or a scaffold-aware
      permutation test, which is not yet implemented;
    * the positive-class prevalence, which is also the PR-AUC baseline. PR-AUC without its baseline
      is uninterpretable: 0.148 looks poor next to 0.854, but against a 2.2% prevalence it is 6.8×
      baseline while the 0.854 is 2.8×.

    This does NOT fail the run. A target where selectivity is unanswerable can still have a perfectly
    good potency model, and discarding it would be the wrong trade.
    """
    n = int(len(delta))
    # ONE-SIDED, deliberately: counts compounds selective *for* the primary target, not |Δ| >= t.
    # Selectivity for the primary is the design goal, so a compound that is a log unit MORE potent on
    # a comparator is not a success at a lower magnitude -- it is a different molecule's opportunity.
    # An earlier version of docs/RESULTS.md labelled this column as |Δ|, which was wrong.
    n_sel = int((delta >= threshold).sum())
    prevalence = n_sel / n if n else 0.0
    lo, hi = _bootstrap_r2_ci(delta, oof, seed=seed)
    distinguishable = bool(lo > 0.0)

    reasons: list[str] = []
    if hi < 0.0:
        # Strictly worse than predicting the mean. A different and stronger statement than
        # "indistinguishable from noise", so it must not share that wording.
        reasons.append(
            f"R² 95% CI [{lo:.3f}, {hi:.3f}] lies entirely below zero — the model is worse than "
            "predicting the mean Δ"
        )
    elif not distinguishable:
        reasons.append(f"R² 95% CI [{lo:.3f}, {hi:.3f}] spans zero — not distinguishable from noise")
    if n_sel == 0:
        reasons.append("no compound exceeds the selectivity threshold; the label is constant")
    elif prevalence < 0.05:
        reasons.append(
            f"positive class is {prevalence:.1%} ({n_sel}/{n}) — classification metrics are unstable "
            "and PR-AUC must be read against this baseline, not against another target's PR-AUC"
        )
    return {
        "supported": distinguishable and n_sel > 0,
        "reasons": reasons,
        "n_paired": n,
        "n_selective": n_sel,
        "prevalence": float(prevalence),
        "r2_ci95": [lo, hi],
        "r2_distinguishable_from_zero": distinguishable,
    }


def _classification(delta_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    # One-sided, matching _support: selective FOR the primary target. See the note there.
    sel = (delta_true >= threshold).astype(int)
    out: dict[str, float] = {}
    if 0 < sel.sum() < len(sel):
        out["pr_auc"] = float(average_precision_score(sel, score))
        out["roc_auc"] = float(roc_auc_score(sel, score))
        k = max(1, int(0.10 * len(score)))
        top = np.argsort(score)[::-1][:k]
        base = float(sel.mean())
        out["top10_enrichment"] = float(sel[top].mean() / base) if base > 0 else float("nan")
    else:
        out["pr_auc"] = out["roc_auc"] = out["top10_enrichment"] = float("nan")
    return out


def _rf(seed: int, n_estimators: int = 400) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)


# The potency-subtract COMPARATOR's estimator. Fixed on purpose and not read from
# `model.potency.xgb`: this model exists to be the thing the direct-Δ model is measured against, so
# it has to be the same estimator in every run and every panel. A configurable comparator would let a
# config improve the direct-Δ result by weakening what it is compared to. Stated here because the two
# XGBoost specifications in this package differ (the potency model's n_estimators is 600) and a
# reader finding that difference deserves to find the reason next to it.
_XGB_BASELINE = {
    "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
}


def _xgb(seed: int) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**_XGB_BASELINE, n_jobs=-1, random_state=seed)


@stage("discovery", "selectivity", deps=("curate",), config_keys=("model", "features", "seed"))
def selectivity(ctx: StageContext) -> StageResult:
    """Direct-Δ selectivity per pair + an apples-to-apples direct-vs-potency comparison."""
    from sklearn.model_selection import GroupKFold

    seed = int(getattr(ctx.config, "seed", 42))
    feat = ctx.config.features
    n_bits, radius, chir = feat.n_bits, feat.radius, feat.use_chirality
    # None (omitted) reaches compute_features as None and selects the standard block there; [] must
    # survive as [] and mean no descriptor columns. `if desc_cfg else None` collapsed the two.
    descriptors = None if feat.descriptors is None else [str(d) for d in feat.descriptors]

    sel_cfg = ctx.config.model.selectivity
    threshold = sel_cfg.delta_threshold
    # NO fallback pair list. A hardcoded JAK panel would silently model the wrong targets for any
    # other project; an unconfigured selectivity stage should produce nothing, loudly.
    #
    # `is None`, not `or`: omitting `pairs` asks for the derived panel, `pairs: []` asks for no
    # selectivity at all. Read with `or`, the second request became the first.
    if sel_cfg.pairs is None:
        primary = str(ctx.config.data.primary or "")
        comparators = [t for t in ctx.config.data.targets if t != primary]
        pairs_cfg = [f"{primary}-{c}" for c in comparators] if primary and comparators else []
    else:
        pairs_cfg = list(sel_cfg.pairs)
    pairs = [tuple(str(p).split("-")) for p in pairs_cfg]
    # The SELECTIVITY model's own tree count, not the potency model's. Reading potency's meant
    # model.selectivity.rf_n_estimators was configurable and ignored.
    rf_trees = sel_cfg.rf_n_estimators

    matrix = pd.read_csv(ctx.upstream["curate"].outputs["selectivity"])

    per_pair: dict[str, Any] = {}   # direct-Δ per pair
    direct_vs_potency_cmp: dict[str, Any] = {}   # direct vs potency-derived (apples-to-apples RF + XGB variant)
    # A requested pair that produces no result must say so. Dropping it from the output makes an
    # absent pair indistinguishable from one that was never configured -- the reader cannot tell
    # whether selectivity was unanswerable or simply not asked.
    not_evaluated: dict[str, str] = {}
    for a, b in pairs:
        key = f"{a}-{b}"
        if a not in matrix.columns or b not in matrix.columns:
            missing = [t for t in (a, b) if t not in matrix.columns]
            not_evaluated[key] = f"no curated activity column for {missing}"
            continue
        sub = matrix[matrix[a].notna() & matrix[b].notna()].copy()
        x_fp, x_desc, scaf_list, keep = compute_features(
            sub["canonical_smiles"].tolist(),
            n_bits=n_bits, radius=radius, use_chirality=chir, descriptors=descriptors,
        )
        sub = sub[keep].reset_index(drop=True)
        scaf = np.asarray(scaf_list)
        y_a = sub[a].to_numpy(dtype=float)
        y_b = sub[b].to_numpy(dtype=float)
        delta = y_a - y_b
        x_full = np.hstack([x_fp, x_desc])  # same feature set as the QSAR stage

        n_groups = len(set(scaf.tolist()))
        if n_groups < 2 or len(sub) < 25:
            not_evaluated[key] = (
                f"only {len(sub)} paired compounds across {n_groups} scaffold group(s); "
                "below the minimum for grouped cross-validation"
            )
            continue
        n_splits = min(5, n_groups)

        oof_direct = np.full(len(delta), np.nan)
        oof_pot_rf = np.full(len(delta), np.nan)
        oof_pot_xgb = np.full(len(delta), np.nan)
        for tr, te in GroupKFold(n_splits=n_splits).split(x_full, delta, groups=scaf):
            # Headline pair: SAME features (fp+desc), SAME model (RF), SAME folds.
            oof_direct[te] = _rf(seed, rf_trees).fit(x_full[tr], delta[tr]).predict(x_full[te])
            pa = _rf(seed, rf_trees).fit(x_full[tr], y_a[tr])
            pb = _rf(seed, rf_trees).fit(x_full[tr], y_b[tr])
            oof_pot_rf[te] = pa.predict(x_full[te]) - pb.predict(x_full[te])
            # Secondary: tuned XGB potency-subtract (different algorithm, same features).
            xa = _xgb(seed).fit(x_full[tr], y_a[tr])
            xb = _xgb(seed).fit(x_full[tr], y_b[tr])
            oof_pot_xgb[te] = xa.predict(x_full[te]) - xb.predict(x_full[te])

        support = _support(delta, oof_direct, threshold, seed)
        cls = _classification(delta, oof_direct, threshold)
        # PR-AUC is meaningless without its baseline, which IS the prevalence.
        if "pr_auc" in cls and support["prevalence"] > 0:
            cls["pr_auc_baseline"] = support["prevalence"]
            cls["pr_auc_lift_over_baseline"] = cls["pr_auc"] / support["prevalence"]
        per_pair[key] = {
            "n": int(len(delta)),
            "n_selective": int((delta >= threshold).sum()),
            **_reg(delta, oof_direct),
            **cls,
            "n_splits": n_splits,
            "support": support,
        }
        direct_vs_potency_cmp[key] = {
            "direct_delta_rf": {  # RF on fp+desc, predict Δ directly
                "r2": _reg(delta, oof_direct)["r2"],
                "pr_auc": _classification(delta, oof_direct, threshold)["pr_auc"],
            },
            "pot_delta_rf": {  # RF on fp+desc, predict two potencies & subtract — APPLES-TO-APPLES
                "r2": _reg(delta, oof_pot_rf)["r2"],
                "pr_auc": _classification(delta, oof_pot_rf, threshold)["pr_auc"],
            },
            "pot_delta_xgb_desc": {  # secondary: tuned XGB potency-subtract (different algorithm)
                "r2": _reg(delta, oof_pot_xgb)["r2"],
                "pr_auc": _classification(delta, oof_pot_xgb, threshold)["pr_auc"],
            },
        }

    supported = sorted(k for k, v in per_pair.items() if v["support"]["supported"])
    unsupported = sorted(k for k, v in per_pair.items() if not v["support"]["supported"])
    metrics = {
        "pairs_requested": [f"{a}-{b}" for a, b in pairs],
        "pairs_supported": supported,
        "pairs_unsupported": unsupported,
        "pairs_not_evaluated": not_evaluated,
        "support_note": (
            "An unsupported pair is NOT a modelling failure to be hidden: it means the public data "
            "cannot answer the selectivity question for that pair, usually because almost no "
            "compound exceeds the threshold. The run continues and the potency model stands on its "
            "own. Read every pr_auc against its pr_auc_baseline, never against another target's."
        ),
        "direct_delta_scaffold_cv": per_pair,
        "direct_vs_potency": direct_vs_potency_cmp,
        "comparison_note": (
            "Headline = direct_delta_rf vs pot_delta_rf: identical features (chiral ECFP4 "
            "+ desc), identical RF, identical folds — only the target formulation differs. "
            "pot_delta_xgb_desc is a secondary tuned-potency variant (different algorithm)."
        ),
        "delta_threshold": threshold,
    }
    out = Path(ctx.workdir)
    (out / "selectivity_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Persist a production selectivity model: RF on ALL compounds' min-vs-other selectivity Δ,
    # on the same chiral ECFP4 + desc features the scorer uses (feeds generation / VLS tier-1).
    outputs = {"metrics": str(out / "selectivity_metrics.json")}
    # ------------------------------------------------------------------------------------------
    # A production model is only emitted from SUPPORTED pairs. An unsupported pair's Δ is not a
    # measurement of selectivity -- it is noise at that sample size -- and folding it into an
    # aggregate would launder it into every downstream score through a model file that looks
    # identical to a good one. If nothing is supported, NO model is written and consumers take the
    # optional-selectivity path, which they now genuinely support.
    # ------------------------------------------------------------------------------------------
    supported_comparators = [k.split("-", 1)[1] for k in supported]
    model_basis = None
    if not supported:
        model_note = (
            "no selectivity model written: no pair is supported. Downstream stages take the "
            "optional-selectivity path rather than scoring against noise."
        )
    elif len(supported) < len(per_pair):
        # Recompute the aggregate over SUPPORTED comparators only, rather than reusing curate's
        # delta_min over all of them.
        cols = [f"delta_{a}_{c}" for a, c in [tuple(k.split("-", 1)) for k in supported]]
        present = [c for c in cols if c in matrix.columns]
        if present:
            matrix = matrix.copy()
            matrix["delta_min_supported"] = matrix[present].min(axis=1, skipna=True)
            model_basis = "delta_min_supported"
            model_note = (
                f"aggregate built from SUPPORTED comparators only ({supported_comparators}); "
                f"unsupported pairs excluded rather than folded into the aggregate"
            )
        else:
            model_note = "supported pairs have no delta columns in the matrix; no model written"
    else:
        model_basis = DELTA_MIN_COLUMN
        model_note = "every requested pair is supported; aggregate spans all comparators"

    metrics["production_model"] = {
        "written": False,
        "basis_column": model_basis,
        "supported_comparators": supported_comparators,
        "note": model_note,
    }

    if model_basis and model_basis in matrix.columns:
        full = matrix[matrix[model_basis].notna()]
        if len(full) >= 25:
            import joblib

            # `descriptors` MUST be passed: omitting it silently used the default list, so a model
            # trained here did not match the features every consumer computes from config.
            fx_fp, fx_desc, _fs, fkeep = compute_features(
                full["canonical_smiles"].tolist(), n_bits=n_bits, radius=radius,
                use_chirality=chir, descriptors=descriptors,
            )
            kept = full[fkeep]
            model = _rf(seed, rf_trees).fit(
                np.hstack([fx_fp, fx_desc]), kept[model_basis].to_numpy(dtype=float)
            )
            joblib.dump(model, out / "selectivity_model.joblib")
            outputs["model"] = str(out / "selectivity_model.joblib")
            metrics["production_model"]["written"] = True
            metrics["production_model"]["n_training_compounds"] = int(len(kept))
    # metrics were already serialised above; rewrite so the production-model verdict is recorded
    (out / "selectivity_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return StageResult(name="selectivity", outputs=outputs, metrics=metrics)
