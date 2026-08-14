"""Empirically derive the QSAR applicability-domain (AD) cutoff.

Instead of picking a folklore Tanimoto cutoff (0.3-0.4), we measure where the
model's own error degrades. Using Bemis-Murcko scaffold GroupKFold, for every
compound we pair its OUT-OF-FOLD absolute residual with its OUT-OF-FOLD nearest-
neighbour Tanimoto (max ECFP4 similarity to the compounds the fold was trained
on). Binning residual/coverage vs that similarity gives the reliability curve;
the in-AD cutoff is the similarity below which coverage drops under the 90%
conformal target (equivalently MAE approaches the conformal half-width).

The two numbers that make this target-specific are DERIVED, not hardcoded:

* the fingerprint width comes from the config, because the feature matrix is fingerprint bits followed
  by descriptor columns and this script must slice at the right boundary. A hardcoded 2048 silently
  mixes descriptor columns into the similarity term for any other width.
* the conformal half-width -- the coverage target the AD cutoff is defined against -- comes from THIS
  target's own qsar metrics. It was hardcoded to 0.9549490673740708, which is JAK1's measured value, so
  every other target's applicability domain was judged against a kinase's uncertainty. That is precisely
  the defect ``vls.tier1.conformal_halfwidth`` documents and refuses to default; a script whose whole
  purpose is deriving a per-target threshold should not have carried it either.

Usage:
    uv run python scripts/derive_ad_threshold.py <featurize_cache_dir> --config configs/<panel>.yaml
    uv run python scripts/derive_ad_threshold.py <featurize_cache_dir> --config <cfg> \
        --conformal-halfwidth 0.95      # only if the qsar metrics are unavailable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent

N_TREES = 400
SEED = 42
BIN_EDGES = [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.01]


def nn_tanimoto(query_fp: np.ndarray, ref_fp: np.ndarray) -> np.ndarray:
    """Max Tanimoto of each query bit-vector to any reference bit-vector."""
    q = query_fp.astype(np.float32)
    r = ref_fp.astype(np.float32)
    inter = q @ r.T
    a = q.sum(1, keepdims=True)
    b = r.sum(1)[None, :]
    union = a + b - inter
    tan = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return tan.max(1)


def _resolve_halfwidth(config_path: Path, explicit: float | None) -> tuple[float, str]:
    """This target's own 90% conformal half-width, from its qsar metrics. Fail closed, never inherit.

    An AD threshold is defined as the similarity below which coverage falls under the conformal target,
    so the half-width IS the criterion. Borrowing another target's makes the derived cutoff describe that
    other target.
    """
    if explicit is not None:
        return explicit, "given on the command line"
    sys.path.insert(0, str(REPO / "src"))
    from medchem.config import load_config
    from medchem.provenance import resolve_stage_outputs

    cfg = load_config(config_path)
    outs = resolve_stage_outputs(REPO, config_path, "qsar")
    for path in outs.values():
        if path.name == "qsar_metrics.json":
            m = json.loads(path.read_text())
            hw = ((m.get("conformal_rf") or {}).get("interval_halfwidth_90"))
            if hw is None:
                break
            return float(hw), f"measured for {cfg.target} by its own qsar stage"
    raise SystemExit(
        f"no qsar conformal half-width available for {config_path.name}. Run the qsar stage for this "
        f"target first, or pass --conformal-halfwidth explicitly. Refusing to fall back to another "
        f"target's value: that is what made this script's output describe JAK1 whatever it was run on."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("feat_dir", help="a featurize stage output directory (features.npz + meta.csv)")
    ap.add_argument("--config", required=True,
                    help="the panel config, for the fingerprint width and the target's own half-width")
    ap.add_argument("--conformal-halfwidth", type=float, default=None,
                    help="override, only when this target's qsar metrics are unavailable")
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO / config_path
    sys.path.insert(0, str(REPO / "src"))
    from medchem.config import load_config

    n_bits = load_config(config_path).features.n_bits
    conf_halfwidth, hw_source = _resolve_halfwidth(config_path, args.conformal_halfwidth)

    data = np.load(feat_dir / "features.npz")
    x, y = data["X"], data["y"]
    fp = x[:, :n_bits]
    scaffolds = pd.read_csv(feat_dir / "meta.csv")["scaffold"].fillna("").to_numpy()

    n_groups = len(set(scaffolds.tolist()))
    n_splits = min(5, n_groups)
    oof_resid = np.full(len(y), np.nan)
    oof_nnsim = np.full(len(y), np.nan)

    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups=scaffolds):
        model = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=SEED)
        model.fit(x[tr], y[tr])
        oof_resid[te] = np.abs(y[te] - model.predict(x[te]))
        oof_nnsim[te] = nn_tanimoto(fp[te], fp[tr])  # similarity to TRAIN fold only (no leakage)

    ok = ~np.isnan(oof_resid)
    resid, sim = oof_resid[ok], oof_nnsim[ok]
    r = np.corrcoef(sim, resid)[0, 1]

    print(f"n={ok.sum()}  scaffold-CV folds={n_splits}  n_bits={n_bits}")
    print(f"conformal half-width={conf_halfwidth:.4f}  ({hw_source})")
    print(f"corr(nn_similarity, |residual|) = {r:+.3f}   (negative = closer -> more accurate)\n")

    hdr = (f"{'sim bin':>12} | {'n':>5} | {'%':>5} | {'MAE':>6} | {'RMSE':>6} | "
           f"{'coverage@90':>11} | {'median|res|':>11}")
    print(hdr)
    print("-" * len(hdr))
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:], strict=False):
        m = (sim >= lo) & (sim < hi)
        if not m.any():
            print(f"{lo:.2f}-{hi:<6.2f} |     0 |")
            continue
        rr = resid[m]
        cov = float(np.mean(rr <= conf_halfwidth))
        print(
            f"  {lo:.2f}-{hi:<5.2f} | {m.sum():>5} | {100*m.sum()/len(sim):>4.1f} | "
            f"{rr.mean():>6.3f} | {np.sqrt((rr**2).mean()):>6.3f} | {cov:>11.3f} | {np.median(rr):>11.3f}"
        )

    print("\nCumulative: compounds with nn_similarity BELOW threshold t (the 'out-of-AD' tail)")
    print(f"{'t':>6} | {'n_below':>8} | {'%_below':>8} | {'MAE_below':>9} | "
          f"{'cov_below':>9} | {'MAE_above':>9} | {'cov_above':>9}")
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        below, above = sim < t, sim >= t
        mae_b = resid[below].mean() if below.any() else float("nan")
        cov_b = np.mean(resid[below] <= conf_halfwidth) if below.any() else float("nan")
        mae_a = resid[above].mean() if above.any() else float("nan")
        cov_a = np.mean(resid[above] <= conf_halfwidth) if above.any() else float("nan")
        print(f"{t:>6.2f} | {below.sum():>8} | {100 * below.mean():>7.1f}% | {mae_b:>9.3f} | "
              f"{cov_b:>9.3f} | {mae_a:>9.3f} | {cov_a:>9.3f}")


if __name__ == "__main__":
    main()
