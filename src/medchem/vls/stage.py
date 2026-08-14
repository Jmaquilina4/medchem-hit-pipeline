"""VLS stage: Tier-0 library prep + Tier-1 annotation/stratification, wired into the DAG.

Optional / composable (ADR 0006): runs only when ``vls.enabled`` and the pulled library exists;
otherwise it records a ``skipped`` status so a fresh clone's front-half ``medchem run`` is unaffected.

This stage **selects nothing** (see ADR 0005 §Tier-2-selection-retired). It prepares the whole
pinned library, annotates every compound with the three similarity terms + model predictions,
and emits the applicability-domain strata census — the scientific payload of the ligand tier —
plus the full docking manifest. The structure tiers (Uni-Dock → gnina → Boltz-2 → OpenFE) are
GPU work that consumes that manifest, with the scarce-budget quotas in ``vls.budget_allocation``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from medchem.pipeline.stage import StageContext, StageResult, stage
from medchem.vls.envelope import (
    check_admits,
    derive_envelope,
    derive_reference_ceiling,
    intersect_bounds,
    resolve_potency_cut,
)
from medchem.vls.library import LeadLikeBounds, load_library, prepare_library
from medchem.vls.screen import STRATA, Thresholds, _fp_only, screen_library


def _verify_library_pin(lib_path: str, manifest_path: str) -> dict:
    """Verify the library file against the SHA-256 recorded in its manifest, or say why not.

    ``LibraryConfig`` documents the manifest as the pin. A recorded-but-unchecked checksum is not a pin:
    the path names a location, and the file there can be replaced without any record noticing. So this
    runs BEFORE screening and raises on a mismatch -- screening a library that is not the one the
    provenance record names would produce a correct-looking result about the wrong compounds.

    A manifest is optional. When none is configured the result says so, which is honest and is not the
    same as verifying nothing while claiming a pin.
    """
    import hashlib

    if not manifest_path:
        return {"verified": False, "why": "no manifest configured; the library is identified by path only"}
    mf = Path(manifest_path)
    if not mf.is_file():
        raise FileNotFoundError(
            f"vls.library.manifest points at {manifest_path!r}, which does not exist. The manifest is "
            f"the library pin; refusing to screen an unverified library."
        )
    # The manifest written by scripts/pull_zinc22.py is SHARD-LEVEL: one line per upstream shard, plus
    # one final line for the merged deck. The shard lines pin the INPUTS; only the deck line pins the
    # file that is actually screened, and an earlier version of this check looked only at each line's
    # LAST field -- the shard URL -- so it matched nothing and rejected the real configured library.
    #
    # So: find the line that mentions this file in ANY field, and take the 64-hex token from it.
    lib_name = Path(lib_path).name
    want = None
    n_shard_lines = 0
    for line in mf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        hexes = [p for p in parts if len(p) == 64 and all(c in "0123456789abcdef" for c in p.lower())]
        if any(Path(p).name == lib_name for p in parts) and hexes:
            want = hexes[0]
            break
        n_shard_lines += 1
    if want is None:
        raise ValueError(
            f"vls.library.manifest {manifest_path!r} records no SHA-256 for {lib_name!r} "
            f"({n_shard_lines} other line(s) present, which pin upstream inputs rather than the merged "
            f"deck). The deck's own checksum is the pin on what is screened; rebuild the library with "
            f"scripts/pull_zinc22.py, which appends it."
        )
    h = hashlib.sha256()
    with open(lib_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != want:
        raise ValueError(
            f"library {lib_path!r} does not match its manifest pin: manifest records {want[:16]}…, file "
            f"hashes to {got[:16]}…. The file at this path is not the library the record names."
        )
    return {"verified": True, "manifest": manifest_path, "sha256": got, "file": lib_name}



def _descriptor_table(smiles: list[str]) -> dict[str, list[float]]:
    """MW / cLogP / TPSA across a compound list, skipping anything RDKit cannot parse."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]
    out: dict[str, list[float]] = {"MolWt": [], "MolLogP": [], "TPSA": []}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out["MolWt"].append(float(Descriptors.MolWt(mol)))  # pyright: ignore[reportAttributeAccessIssue]
        out["MolLogP"].append(float(Crippen.MolLogP(mol)))  # pyright: ignore[reportAttributeAccessIssue]
        out["TPSA"].append(float(rdMolDescriptors.CalcTPSA(mol)))
    return out


def _reference_descriptors(refs: dict[str, str]) -> dict[str, dict[str, float]]:
    """Per-compound descriptors for the reference panel, keyed by name."""
    out: dict[str, dict[str, float]] = {}
    for name, smi in refs.items():
        table = _descriptor_table([smi])
        if table["MolWt"]:
            out[name] = {k: v[0] for k, v in table.items()}
    return out


@stage("discovery", "vls", deps=("curate", "featurize", "qsar"), optional_deps=("selectivity",),
       config_keys=("vls", "features", "seed"))
def vls(ctx: StageContext) -> StageResult:
    """Tier-0 (prepare the full purchasable library) → Tier-1 (annotate + stratify)."""
    import joblib

    out = Path(ctx.workdir)
    # Typed reads throughout this stage, which used to mix them with raw-dict lookups of the same
    # sections.
    #
    # To be accurate about what that cost, because an earlier version of this comment overstated it:
    # the two did NOT disagree. `dict(cfg.vls)` returns the validated Tier1Config itself, so a
    # subsequent `dict(...)` of it yielded exactly the validated values -- verified field by field. The
    # real defects were narrower and both live:
    #   * every `.get(key, <literal>)` duplicated a default that config.py already owns, so a default
    #     changed there would silently not reach the screen -- two sources of truth, one unchecked;
    #   * `conformal_halfwidth` was resolved with `or`, which is a genuine behaviour bug (see below).
    # Typed reads remove the first by construction. The second is fixed on its own terms.
    vls_cfg = ctx.config.vls
    lib_cfg = vls_cfg.library
    lib_path = lib_cfg.path
    ready = vls_cfg.enabled and bool(lib_path) and Path(lib_path).exists()
    if not ready:
        metrics = {
            "status": "skipped",
            "reason": f"enabled={vls_cfg.enabled}, "
                      f"library={'present' if lib_path and Path(lib_path).exists() else 'absent'}",
            "hint": "pull a library (scripts/pull_zinc22.py) and set vls.enabled: true",
        }
        (out / "vls_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return StageResult(name="vls", outputs={"metrics": str(out / "vls_metrics.json")}, metrics=metrics)

    feat = ctx.config.features
    n_bits, radius, chir = feat.n_bits, feat.radius, feat.use_chirality
    # None (omitted) selects the standard descriptor block downstream; [] must stay [] and mean no
    # descriptor columns. The screen has to featurize exactly as `featurize` did or the model it
    # loads is fed a different matrix than it was trained on.
    descriptors = None if feat.descriptors is None else [str(d) for d in feat.descriptors]

    # Curated data is read FIRST: the physchem window can be derived from this target's own actives,
    # which means it has to exist before Tier 0 filters anything.
    primary_df = pd.read_csv(ctx.upstream["curate"].outputs["potency_training"])
    train_fp = np.load(ctx.upstream["featurize"].outputs["features"])["X"][:, :n_bits]

    acfg_t = vls_cfg.actives
    # "Potent" is target-relative: an absolute 8.0 discards JQ1 (~7.1) on a bromodomain. A configured
    # quantile wins, and the resolved absolute value is recorded so a reader still sees what applied.
    pic50_min, pic50_how = resolve_potency_cut(
        primary_df["pIC50"].dropna().tolist(),
        quantile=acfg_t.pic50_quantile, absolute=acfg_t.pic50_min,
    )

    # ---- Tier 0: VERIFY THE PIN, then prepare the ENTIRE library (no cap, no sample, no seed) -----
    #
    # `manifest` is documented as the pin, and a pin that is recorded but never checked is not a pin: a
    # path identifies a location, and the file at a location can be replaced. So the library bytes are
    # verified against the manifest's SHA-256 before anything is screened, and a mismatch RAISES rather
    # than screening a library that is not the one the record names.
    manifest_record = _verify_library_pin(lib_path, lib_cfg.manifest)
    n_rows = sum(1 for _ in open(lib_path, encoding="utf-8"))
    ll = lib_cfg.lead_like
    explicit_bounds = {
        "mw_min": ll.mw_min, "mw_max": ll.mw_max,
        "logp_max": ll.logp_max, "tpsa_max": ll.tpsa_max,
    }
    window = dict(explicit_bounds)
    envelope_record: dict = {"mode": "explicit", "bounds": explicit_bounds}

    derive_cfg = vls_cfg.library.lead_like.derive
    if derive_cfg is not None:
        # Three sources, each supplying what the others cannot: potent actives give chemotype
        # plausibility (wide, because optimised compounds drift large), the target's own reference
        # compounds give a lead-likeness ceiling that SLIDES with the target, and explicit config
        # supplies any hard cap the data cannot express.
        sel = primary_df[primary_df["pIC50"] >= pic50_min]
        env = derive_envelope(
            _descriptor_table(sel["canonical_smiles"].dropna().tolist()),
            quantiles=(derive_cfg.envelope_quantiles[0], derive_cfg.envelope_quantiles[1]),
            margin=derive_cfg.margin, n_actives=len(sel),
            potency_cut=pic50_min, potency_cut_how=pic50_how,
        )
        ref_desc = _reference_descriptors(dict(vls_cfg.known_reference))
        ceiling = derive_reference_ceiling(ref_desc, margin=derive_cfg.margin)
        window = intersect_bounds(env.bounds, ceiling["bounds"])
        if derive_cfg.intersect_with_explicit:
            window = intersect_bounds(window, explicit_bounds)
        admits = check_admits(window, ref_desc)
        envelope_record = {
            "mode": "derived",
            "bounds": window,
            "actives_envelope": env.as_dict(),
            "reference_ceiling": ceiling,
            "explicit_ceiling": explicit_bounds if derive_cfg.intersect_with_explicit else None,
            "admits_references": admits,
        }
        if derive_cfg.must_admit_references and not admits["passed"]:
            # A physchem filter that rejects the compounds you benchmark against is wrong. Refuse
            # rather than screen a deck that cannot contain this target's known chemistry.
            raise ValueError(
                "derived screening window rejects this target's own reference compounds "
                f"({admits['n_admitted']}/{admits['n_references']} admitted):\n  - "
                + "\n  - ".join(f"{k}: {', '.join(v)}" for k, v in admits["rejected"].items())
            )

    bounds = LeadLikeBounds(
        mw=(window["mw_min"], window["mw_max"]),
        logp_max=window["logp_max"], tpsa_max=window["tpsa_max"],
        hbd_max=ll.hbd_max, hba_max=ll.hba_max, rotb_max=ll.rotb_max,
    )
    # Only "pains" is honoured here; the physicochemical window comes from `lead_like` regardless, and
    # no de-duplication step reads this list. Recorded in the report so the gap is visible.
    prefilters = list(lib_cfg.prefilters)
    records = load_library(lib_path, cap=None)
    lib = prepare_library(records, bounds=bounds, apply_pains=("pains" in prefilters))

    anchors = primary_df[primary_df["pIC50"] >= pic50_min].nlargest(acfg_t.max_anchors, "pIC50")
    active_fp, _ = _fp_only(
        anchors["canonical_smiles"].tolist(), n_bits=n_bits, radius=radius, use_chirality=chir
    )

    known_reference = dict(vls_cfg.known_reference)
    known_ref_names = list(known_reference.keys())
    known_ref_fp, pkeep = _fp_only(
        [known_reference[k] for k in known_ref_names], n_bits=n_bits, radius=radius, use_chirality=chir
    )
    known_ref_names = [n for n, k in zip(known_ref_names, pkeep, strict=True) if k]

    # ---- Tier 1: annotate + stratify (selects nothing) ------------------------------------
    potency = joblib.load(ctx.upstream["qsar"].outputs["model"])
    # Selectivity is OPTIONAL: a single-target project has no comparator isoforms, and disabling the
    # stage must not break this one. Without it, every compound simply carries no selectivity
    # annotation -- the stratification and similarity signals are unaffected.
    sel_up = ctx.upstream.get("selectivity")
    selectivity_predict = None
    if sel_up is not None and "model" in sel_up.outputs:
        selectivity_predict = joblib.load(sel_up.outputs["model"]).predict
    qm = json.loads(Path(ctx.upstream["qsar"].outputs["metrics"]).read_text(encoding="utf-8"))
    t1 = vls_cfg.tier1
    # An AD threshold measured on one target is not transferable: the fingerprint-similarity
    # distribution differs. Refuse rather than relabel a deck against another target's curve.
    if t1.derived_for is not None and t1.derived_for != ctx.config.target:
        raise ValueError(
            f"vls.tier1.derived_for={t1.derived_for!r} but this run is target "
            f"{ctx.config.target!r}. Applicability-domain thresholds are measured per target from its "
            f"own out-of-fold coverage curve (scripts/derive_ad_threshold.py); using another target's "
            f"would relabel the deck while the run reported success. Re-derive, or clear derived_for "
            f"to record these as unprovenanced."
        )
    threshold_provenance = {
        "derived_for": t1.derived_for,
        "matches_this_target": t1.derived_for == ctx.config.target,
        "warning": None if t1.derived_for else (
            "vls.tier1 carries no derived_for: these thresholds are UNPROVENANCED. Any stratification "
            "they produce is arithmetic, not evidence, until they are measured for this target."
        ),
    }
    # Potency thresholds resolve from this target's own distribution when a quantile is configured.
    pool = primary_df["pIC50"].dropna().tolist()
    hit_floor, hit_floor_how = resolve_potency_cut(
        pool, quantile=t1.hit_floor_quantile, absolute=t1.hit_floor)
    prio_point, prio_point_how = resolve_potency_cut(
        pool, quantile=t1.prio_point_quantile, absolute=t1.prio_point)
    # Typed throughout. Half of these were re-read out of a raw dict beside the typed `t1` above, which
    # DUPLICATED config.py's defaults rather than disagreeing with them -- see the note at the top of
    # this function, which corrects the same overstatement. The values matched; the second source of
    # truth is the defect, because a default changed in config.py would not have reached here.
    #
    # `is None`, not `or`: unset means "use this run's measured half-width", and 0.0 means "no
    # uncertainty band" -- a real request, and the way to see what the band does to the strata.
    # Under `or`, 0.0 was replaced by the measured value and the ablation ran unablated.
    t = Thresholds(
        in_domain=t1.in_domain,
        borderline=t1.borderline,
        high_conf=t1.high_conf,
        conformal_halfwidth=(qm["conformal_rf"]["interval_halfwidth_90"]
                             if t1.conformal_halfwidth is None else t1.conformal_halfwidth),
        hit_floor=hit_floor,
        prio_point=prio_point,
        prio_delta=t1.prio_delta,
        fep_lower=t1.fep_lower,
        fep_delta=t1.fep_delta,
        known_reference_flag=t1.known_reference_flag,
    )
    res = screen_library(
        lib.records,
        potency_predict=potency.predict, selectivity_predict=selectivity_predict,
        train_fp=train_fp, active_fp=active_fp, known_ref_fp=known_ref_fp, known_ref_names=known_ref_names,
        thresholds=t, n_bits=n_bits, radius=radius, use_chirality=chir, descriptors=descriptors,
    )

    # ---- Outputs: full annotated manifest (= the docking queue) + strata census -----------
    table = pd.DataFrame({k: v for k, v in res.columns.items()})
    table = table[table["parsed"]].drop(columns=["parsed"])
    for col in ("pred_pic50", "pred_pic50_lower", "pred_pic50_upper", "pred_selectivity_delta",
                "sim_to_train", "sim_to_actives", "sim_to_known_reference"):
        if col in table.columns:          # pred_selectivity_delta is absent without the optional stage
            table[col] = table[col].round(3)
    table.to_csv(out / "vls_annotated.csv", index=False)

    # Per-stratum QSAR-label breakdown — the observational comparison docking-everything buys.
    per_stratum = {
        s: {
            "n": int((table["stratum"] == s).sum()),
            "qsar_labels": table.loc[table["stratum"] == s, "qsar_label"].value_counts().to_dict(),
            "median_sim_to_train": round(float(table.loc[table["stratum"] == s, "sim_to_train"].median()), 3)
            if (table["stratum"] == s).any() else None,
            "median_pred_pic50": round(float(table.loc[table["stratum"] == s, "pred_pic50"].median()), 3)
            if (table["stratum"] == s).any() else None,
        }
        for s in STRATA
    }
    known_reference_hits = table[table["sim_to_known_reference"] >= t.known_reference_flag]
    known_reference_hits.nlargest(min(len(known_reference_hits), 200), "sim_to_known_reference").to_csv(
        out / "vls_known_reference_review.csv", index=False
    )

    funnel = {
        # source/snapshot are RECORD-ONLY labels: they select nothing. The bytes are pinned by the
        # manifest, whose verification result is recorded beside them.
        "library_source": lib_cfg.source,
        "library_snapshot": lib_cfg.snapshot,
        "library_manifest": lib_cfg.manifest,
        "library_pin": manifest_record,
        "prefilters_declared": prefilters,
        "prefilters_honoured": [f for f in prefilters if f == "pains"],
        "selection": "NONE — the entire pinned Tier-0-prepared library is the docking queue "
                     "(no subsample, no seed, no cap). See ADR 0005 §Tier-2-selection-retired.",
        "rows_in_file": n_rows,
        "tier0_attrition": [list(x) for x in lib.attrition],
        "tier0_dropped": lib.dropped,
        "tier1_strata": res.strata,
        "tier1_qsar_labels": res.priority,
        "tier1_n_screened": res.n_screened,
        "tier1_n_unparseable": res.n_unparseable,
        "per_stratum": per_stratum,
        "similarity_summary": res.similarity_summary,
        "known_reference": {
            "references": known_ref_names,
            "flag_threshold": t.known_reference_flag,
            "n_flagged": int(res.known_reference_flagged),
            "note": "TRIAGE ONLY — Tanimoto has no legal standing; Markush claims are defined "
                    "by substitution patterns, not similarity.",
        },
        "docking_queue": {
            "n": res.n_docking_queue,
            "keep_rule": "all",
            "note": "Tier 2 performs no selection; scarce-budget quotas live at Tiers 3-5 "
                    "(vls.budget_allocation).",
        },
        "thresholds": t.__dict__,
        "actives_anchor": {"pic50_min": pic50_min, "n": int(active_fp.shape[0])},
    }
    (out / "vls_funnel.json").write_text(json.dumps(funnel, indent=2), encoding="utf-8")

    metrics = {
        "rows_in_file": n_rows,
        "tier0_prepared": lib.n_prepared,
        "screened": res.n_screened,
        "strata": res.strata,
        "qsar_labels": res.priority,
        "docking_queue": res.n_docking_queue,
        "known_reference_flagged": int(res.known_reference_flagged),
        "max_sim_to_actives": res.similarity_summary.get("sim_to_actives", {}).get("max"),
        "max_sim_to_train": res.similarity_summary.get("sim_to_train", {}).get("max"),
        "screening_window": envelope_record,
        "potency_cut": {"value": pic50_min, "how": pic50_how},
        "tier1_thresholds": {
            "provenance": threshold_provenance,
            "hit_floor": {"value": hit_floor, "how": hit_floor_how},
            "prio_point": {"value": prio_point, "how": prio_point_how},
        },
    }
    return StageResult(
        name="vls",
        outputs={
            "funnel": str(out / "vls_funnel.json"),
            "annotated": str(out / "vls_annotated.csv"),
            "known_reference_review": str(out / "vls_known_reference_review.csv"),
        },
        metrics=metrics,
    )
