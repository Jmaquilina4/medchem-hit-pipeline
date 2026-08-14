"""Scoring bridge: exposes this project's own models to REINVENT4's `ExternalProcess` component.

Contract (from reinvent_plugins/components/comp_external_process.py):
    stdin   newline-separated SMILES
    stdout  {"payload": {"<property>": [float, ...], ...}} with one value per input SMILES, in order

Properties provided:
    qsar_pic50      RF potency prediction (chiral ECFP4 + descriptors)
    sim_to_train    max Tanimoto to the QSAR training set -- drives the applicability-domain BAND
    sim_to_known_reference   max Tanimoto to marketed reference compounds -- known-reference similarity alert

Why these three and not more: everything else REINVENT needs (MW, logP, TPSA, QED, Csp3, ring counts,
alerts) already exists as a native component and runs in microseconds. This process exists only for the
signals native components cannot supply.

IMPORTANT -- how these are meant to be TRANSFORMED in the TOML, and why:

  qsar_pic50 is a FLOOR, not a target. Use a steep sigmoid or a step at ~7.0-7.5 so nothing is gained
  above it. Measured on this project's data, the RF's *highest* predictions land where it has the least
  support: prediction correlates with sim_to_train at Spearman -0.48, and its MAE in the operating band
  is 1.28 log units. A random forest cannot extrapolate -- predictions are bounded by training values --
  so a high score in a sparse region is more likely trees averaging over unrelated leaves than signal.
  Maximising it walks uphill into the model's blind spot.

  sim_to_train is a BAND (double_sigmoid ~0.35-0.55), not a floor. A conventional applicability-domain
  penalty rewards being *in* domain, which for a fast-follower rewards being derivative. Too far and
  predictions are fantasy; too close and there is no novelty. The useful place is the EDGE.

  sim_to_known_reference is a one-sided penalty (reverse_sigmoid, knee ~0.5). Optimising into
  chemistry already represented by clinical or marketed compounds is the commercially fatal version of
  "drifted toward known chemistry". The reference set comes from the CONFIG, per target -- it is not a
  fixed list, because one target's reference compounds are meaningless for another.

Speed: ExternalProcess spawns a fresh subprocess per batch, so model load cost is paid per call. The
training fingerprints are cached to a .npy on first run, which takes the load from ~4 s to ~0.3 s.

Usage (normally invoked by REINVENT, but testable directly):
    printf 'CCO\\nc1ccccc1\\n' | uv run python scripts/reinvent_oracle.py --config configs/brd4.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / ".medchem_cache" / "scorer"


# ------------------------------------------------------------------------------------------------
# TARGET RESOLUTION — everything below used to be hardcoded to the flagship target.
#
# This was a cross-target contamination waiting to happen, and it was live: the model was chosen by
# `next(Path("runs").rglob("potency_model.joblib"))`, which with three targets on disk silently picked
# JAK1's. Scoring a BRD4 campaign would have used JAK1's QSAR model, JAK1's training set for the
# applicability-domain term, and the five marketed JAK inhibitors as the "known reference" set — and
# nothing in the output would have looked wrong.
#
# Now a config is REQUIRED. It supplies the target, and every artifact is resolved inside that
# target's own run directory. There is no fallback: an ambiguous or missing model is an error, because
# guessing is exactly how the wrong model gets used.
# ------------------------------------------------------------------------------------------------
def _resolve(config_path: str) -> dict:
    """Resolve target, model, training set and reference compounds from ONE config. No globbing."""
    from medchem.config import load_config

    cfg = load_config(config_path)
    target = cfg.target
    run_dir = REPO / "runs" / target
    if not run_dir.is_dir():
        raise SystemExit(
            f"no run directory for target {target!r} at {run_dir.relative_to(REPO)} — "
            f"run the pipeline for this target before scoring against it"
        )

    def _newest(pattern: str) -> Path:
        hits = sorted(run_dir.rglob(pattern), key=lambda p: p.stat().st_mtime)
        if not hits:
            raise SystemExit(
                f"no {pattern} under runs/{target}/ — the scorer will NOT fall back to another "
                f"target's artifacts. Run the pipeline for {target!r} first."
            )
        return hits[-1]

    model = _newest("potency_model.joblib")
    train = _newest("potency_training.csv")
    refs = list(cfg.vls.known_reference.values())
    if not refs:
        raise SystemExit(
            f"config {config_path} defines no vls.known_reference compounds; the "
            f"sim_to_known_reference term would compare against nothing"
        )
    # The FEATURISATION this config trained under, carried explicitly. The scoring code below loads a
    # model that `featurize` built from these values and must reproduce them exactly; it previously
    # called compute_features() with no arguments, taking the function's defaults (2048 chiral bits and
    # the nine standard descriptors) whatever the config said -- directly under a comment asserting
    # "featurisation must match training exactly". With the shipped configs the defaults happen to
    # agree, which is why nothing broke; with any other featurisation the oracle scores a REINVENT
    # campaign against a model fed a matrix it was never trained on.
    feat = cfg.features
    return {
        "target": target,
        "model": model,
        "train_csv": train,
        "known_reference": refs,
        "n_bits": feat.n_bits,
        "radius": feat.radius,
        "use_chirality": feat.use_chirality,
        # None means "the project's standard descriptor block"; [] means no descriptor columns. Both
        # are passed through unchanged -- collapsing them is the bug this mirrors elsewhere.
        "descriptors": None if feat.descriptors is None else [str(d) for d in feat.descriptors],
        # per-target fingerprint cache, so one target's cached bits can never serve another
        "cache": CACHE / target,
    }


def _fp(mol, *, n_bits: int, radius: int, use_chirality: bool):
    """The similarity fingerprint, built with the run's OWN geometry rather than three literals."""
    from rdkit.Chem import AllChem

    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, useChirality=use_chirality)


def _load_reference_bits(ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """Training-set and reference fingerprints as packed bit matrices, cached per TARGET.

    Every similarity term must live in the space the model was trained in, so the geometry comes from
    the config rather than from literals.

    The cache filename carries the geometry. Keyed by target alone, a config that changed the
    fingerprint geometry would silently reuse bits built under the old one -- a stale-cache version of
    the same mistake, and one nobody would look for.
    """
    from rdkit import Chem

    geom = {"n_bits": ctx["n_bits"], "radius": ctx["radius"],
            "use_chirality": ctx["use_chirality"]}
    tag = f"{geom['n_bits']}b{geom['radius']}r{'c' if geom['use_chirality'] else 'a'}"
    cache = ctx["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    train_npy = cache / f"train_bits.{tag}.npy"
    if train_npy.exists():
        train = np.load(train_npy, mmap_mode="r")
    else:
        import pandas as pd

        smis = pd.read_csv(ctx["train_csv"])["canonical_smiles"].tolist()
        rows = [np.frombuffer(bytes(_fp(m, **geom).ToBitString(), "ascii"), dtype=np.uint8) - ord("0")
                for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
        train = np.packbits(np.asarray(rows, dtype=bool), axis=1)
        np.save(train_npy, train)
    pat_rows = [np.frombuffer(bytes(_fp(Chem.MolFromSmiles(s), **geom).ToBitString(), "ascii"),
                              dtype=np.uint8) - ord("0") for s in ctx["known_reference"]]
    patent = np.packbits(np.asarray(pat_rows, dtype=bool), axis=1)
    return np.asarray(train), patent


def _max_tanimoto(query_bits: np.ndarray, ref_packed: np.ndarray) -> np.ndarray:
    """Max Tanimoto of each query row against every reference row, on packed bits."""
    q = np.packbits(query_bits, axis=1)
    # popcount via lookup on uint8
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    qc = lut[q].sum(axis=1).astype(np.float32)
    rc = lut[ref_packed].sum(axis=1).astype(np.float32)
    out = np.zeros(len(q), dtype=np.float32)
    # chunk over references to bound memory
    for start in range(0, len(ref_packed), 512):
        chunk = ref_packed[start:start + 512]
        inter = lut[np.bitwise_and(q[:, None, :], chunk[None, :, :])].sum(axis=2).astype(np.float32)
        union = qc[:, None] + rc[None, start:start + 512] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(union > 0, inter / union, 0.0)
        out = np.maximum(out, t.max(axis=1))
    return out


def main() -> int:
    import argparse

    from rdkit import Chem, RDLogger

    ap = argparse.ArgumentParser(description="REINVENT4 ExternalProcess scoring bridge")
    ap.add_argument("--config", required=True,
                    help="target config (e.g. configs/brd4.yaml). REQUIRED: there is no default, "
                         "because defaulting is how another target's model gets used silently.")
    args = ap.parse_args()
    ctx = _resolve(args.config)
    # provenance on stderr so it lands in the REINVENT log without corrupting the stdout payload
    print(f"[scorer] target={ctx['target']} model={ctx['model'].relative_to(REPO)} "
          f"train={ctx['train_csv'].relative_to(REPO)} refs={len(ctx['known_reference'])}",
          file=sys.stderr)

    RDLogger.DisableLog("rdApp.*")

    smiles = [s.strip() for s in sys.stdin.read().splitlines() if s.strip()]
    n = len(smiles)
    qsar = np.zeros(n, dtype=np.float32)
    sim_train = np.zeros(n, dtype=np.float32)
    sim_known_ref = np.zeros(n, dtype=np.float32)

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ok = [i for i, m in enumerate(mols) if m is not None]

    if ok:
        # --- QSAR. Featurisation must match training exactly, so reuse the project's own function.
        import joblib

        from medchem.features.featurize import compute_features

        fp, desc, _, keep = compute_features(
            [smiles[i] for i in ok],
            n_bits=ctx["n_bits"], radius=ctx["radius"],
            use_chirality=ctx["use_chirality"], descriptors=ctx["descriptors"],
        )
        model = joblib.load(ctx["model"])
        est = model["model"] if isinstance(model, dict) and "model" in model else model
        preds = est.predict(np.hstack([fp, desc]))
        # compute_features can drop rows; map predictions back onto the surviving indices
        kept = [ok[j] for j, k in enumerate(keep) if k]
        for idx, p in zip(kept, preds, strict=False):
            qsar[idx] = float(p)

        # --- similarity signals
        train_bits, known_ref_bits = _load_reference_bits(ctx)
        qbits = np.zeros((len(ok), ctx["n_bits"]), dtype=bool)
        for j, i in enumerate(ok):
            bits = _fp(mols[i], n_bits=ctx["n_bits"], radius=ctx["radius"],
                       use_chirality=ctx["use_chirality"])
            qbits[j] = np.frombuffer(bytes(bits.ToBitString(), "ascii"),
                                     dtype=np.uint8) - ord("0")
        st = _max_tanimoto(qbits, train_bits)
        sp = _max_tanimoto(qbits, known_ref_bits)
        for j, i in enumerate(ok):
            sim_train[i] = st[j]
            sim_known_ref[i] = sp[j]

    # Invalid SMILES score 0 on every property. REINVENT requires one value per input, in order --
    # silently dropping rows would misalign the whole batch against its molecules.
    print(json.dumps({"payload": {
        "qsar_pic50": [round(float(x), 4) for x in qsar],
        "sim_to_train": [round(float(x), 4) for x in sim_train],
        "sim_to_known_reference": [round(float(x), 4) for x in sim_known_ref],
    }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
