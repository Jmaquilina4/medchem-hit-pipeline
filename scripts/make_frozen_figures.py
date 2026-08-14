"""Generate the figures for the frozen results, reading only ``provenance/``.

Every number plotted here comes from a published run manifest or its resolved eval report. Nothing is
hand-entered, so a figure cannot drift from the result it depicts — the previous figure set went stale
precisely because it was generated once, by hand, from a run nobody could later identify.

Four figures, each answering one question the frozen results actually settle:

  1. ``frozen_validation.png``  — retrospective vs chronological, per panel. The point of the paper:
     scaffold-CV clears the gate everywhere while temporal performance collapses for JAK1 and is
     negligible for BRD4. Plotted on one axis so the gap is the visual, not a caption.
  2. ``frozen_selectivity.png`` — per-pair R² with 95% intervals, supported pairs marked. Shows that
     NONE of BRD4's three BD1-explicit comparisons is distinguishable from noise while all three of
     JAK1's are strong — without implying the two are the same kind of evidence.
  3. ``frozen_attrition.png``   — assays and compounds surviving each cohort, per target. Makes the
     cost of the restriction visible: BRD4's headline uses 65% fewer compounds.
  4. ``frozen_cohort_mix.png``  — the assay-label composition that motivated the whole exercise.

Figures are written with metadata stripped (``Software: None``), because
``scripts/check_binary_metadata.py`` flags embedded tool and path metadata, and a published
repository should not carry it.

Usage:
    python scripts/make_frozen_figures.py             # write to results/figures/frozen/
    python scripts/make_frozen_figures.py --check     # fail if a figure is missing or stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROV = REPO / "provenance"
OUT = REPO / "results" / "figures" / "frozen"

PANELS = [
    ("jak1", "JAK1\nbiochemical\n(headline)"),
    ("jak1_sensitivity", "JAK1\npooled\n(sensitivity)"),
    # Tracks the cohort name, and has been wrong twice. "BD1+biochem" (spec 1.0) claimed a biochemical
    # confirmation the rules never performed; "BD1, non-cellular" (spec 1.1) was honest about the
    # text-only rule but is no longer what runs. "BD1-explicit" says where the domain came from -- an
    # explicit statement in the assay DESCRIPTION -- and claims nothing about ChEMBL's structured
    # fields, which confirm the assay FORMAT rather than the domain. The full cohort name carries both
    # halves; a tick label cannot, and compressing them into "confirmed" is what went wrong twice.
    #
    # KEEP THIS TO THREE LINES. The legend is placed below the tick labels at a fixed offset, so a
    # four-line label overlaps it and renders "(headline)" underneath the legend text.
    ("brd4", "BRD4\nBD1-explicit\n(headline)"),
    ("brd4_sensitivity", "BRD4\npooled\n(sensitivity)"),
]
# Headline panels carry the claim; sensitivity panels are context. Encode that, don't caption it.
HEAD = {"jak1", "brd4"}
NOMETA = {"Software": None}

# The key each figure carries so ``--check`` can tell current from stale WITHOUT filesystem mtimes.
# Git does not record mtimes: after any clone every file carries checkout time, so comparing a PNG's
# mtime against a provenance record's compared two arbitrary equal-ish numbers and passed by
# construction -- in CI, which is the one place the check had to work. Byte-comparing a regenerated
# figure is not an option either, because font rasterisation differs between platforms.
#
# So each figure embeds a digest of the provenance records it was drawn from, and --check recomputes
# that digest and compares. A hash is not identifying metadata, so this stays compatible with
# scripts/check_binary_metadata.py.
PROV_DIGEST_KEY = "ProvenanceDigest"


def _prov_digest() -> str:
    """Hash of every provenance record the figures read, in a fixed order."""
    h = hashlib.sha256()
    for panel, _ in PANELS:
        for name in ("run_manifest.json", "eval_report.json", "selectivity_metrics.json",
                     "composition.json"):
            f = PROV / panel / name
            if f.is_file():
                h.update(f"{panel}/{name}".encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def _figure_digest(path: Path) -> str | None:
    """The digest a shipped figure was drawn with, or None if it carries none."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return (im.info or {}).get(PROV_DIGEST_KEY)
    except Exception:
        return None


def _load() -> dict:
    out = {}
    for panel, _ in PANELS:
        d = PROV / panel
        if not (d / "run_manifest.json").exists():
            raise SystemExit(f"missing provenance for {panel}: run publish_provenance.py first")
        sel = d / "selectivity_metrics.json"
        out[panel] = {
            "m": json.loads((d / "run_manifest.json").read_text()),
            "e": json.loads((d / "eval_report.json").read_text()),
            "s": json.loads(sel.read_text()) if sel.exists() else None,
        }
    return out


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def fig_validation(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    labels = [lab for _, lab in PANELS]
    scaf = [data[p]["m"]["gates"]["scaffold_cv_r2_min"]["value"] for p, _ in PANELS]
    temp = [data[p]["e"]["temporal_split"]["r2"] for p, _ in PANELS]
    scr = [data[p]["m"]["gates"]["y_scramble_r2_max"]["value"] for p, _ in PANELS]

    x = np.arange(len(PANELS))
    w = 0.26
    for off, vals, name, col in (
        (-w, scaf, "scaffold-CV (retrospective)", "#2b6cb0"),
        (0.0, temp, "temporal (chronological)", "#c05621"),
        (w, scr, "y-scramble (negative control)", "#a0aec0"),
    ):
        bars = ax.bar(x + off, vals, w, label=name, color=col,
                      edgecolor="black", lw=0.5)
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3 if v >= 0 else -11),
                        ha="center", fontsize=7.4)

    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.55, color="#2b6cb0", ls="--", lw=1.0)
    # Both threshold labels are ANCHORED in the gaps between panel groups (bars span centre ±0.39),
    # but the anchor was not the whole story: this text is wider than the gap, so the gate label ran
    # under the neighbouring group's bars and rendered as "gate (identical on all pa…≤ 0.55".
    # Anchoring controls where a label starts, not how far it extends. So each label now carries an
    # opaque background and sits above the bars in z-order, which holds whatever the text width and
    # however the fonts rasterise on another platform.
    _lbl = dict(fontsize=7.6, ha="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2))
    ax.annotate("gate (identical on all panels)  R² ≥ 0.55", (0.5, 0.575),
                color="#2b6cb0", **_lbl)
    ax.axhline(0.10, color="#718096", ls=":", lw=1.0)
    ax.annotate("null ceiling  R² ≤ 0.10", (1.5, 0.125), color="#718096", **_lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.4)
    for tick, (p, _) in zip(ax.get_xticklabels(), PANELS, strict=True):
        if p in HEAD:
            tick.set_fontweight("bold")
    ax.set_ylabel("R²")
    ax.set_ylim(-0.55, 0.92)
    ax.set_title("Retrospective performance is strong; chronological performance is not\n"
                 "Both gates pass on all four panels — the gates were never the limitation",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.2, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    _style(ax)
    fig.tight_layout()
    p = OUT / "frozen_validation.png"
    fig.savefig(p, dpi=200, metadata={**NOMETA, PROV_DIGEST_KEY: _prov_digest()}, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_selectivity(data: dict) -> Path:
    rows = []
    for panel, lab in PANELS:
        s = data[panel]["s"]
        if not s:
            continue
        for pair, v in s.get("direct_delta_scaffold_cv", {}).items():
            sup = v.get("support", {})
            ci = sup.get("r2_ci95") or [v["r2"], v["r2"]]
            rows.append((f"{pair}  ({lab.splitlines()[1]})", v["r2"], ci,
                         bool(sup.get("supported")), v["n"], panel in HEAD,
                         bool(sup.get("supported")) and bool(sup.get("reasons"))))
    rows.reverse()

    fig, ax = plt.subplots(figsize=(11.0, 0.42 * len(rows) + 2.1))
    caveated = False
    for i, (_name, r2, ci, sup, n, is_head, warn) in enumerate(rows):
        caveated = caveated or warn
        col = "#2f855a" if sup else "#a0aec0"
        ax.plot([ci[0], ci[1]], [i, i], color=col, lw=2.4, solid_capstyle="round")
        ax.plot([r2], [i], "o", color=col, ms=6.5,
                markeredgecolor="black", markeredgewidth=0.6 if is_head else 0.0)
        # Two dedicated columns in a right margin. Anchoring both near the data edge made them
        # overlap each other and the JAK1 intervals, which sit above 0.75.
        ax.annotate(f"{n:,}", (1.04, i), fontsize=7.4, va="center", ha="right", color="#4a5568")
        verdict = ("supported †" if warn else "supported") if sup else \
                  "not distinguishable from noise"
        ax.annotate(verdict, (1.10, i), fontsize=7.4, va="center", ha="left",
                    color=col, fontweight="bold" if sup else "normal")

    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.1)
    ax.set_xlabel("scaffold-CV R² of the predicted potency difference (95% interval)")
    ax.set_xlim(-0.25, 1.78)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-1.15, len(rows) + 0.05)
    ax.annotate("n pairs", (1.04, len(rows) - 0.55), fontsize=7.4, ha="right",
                color="#4a5568", fontweight="bold")
    ax.annotate("support verdict", (1.10, len(rows) - 0.55), fontsize=7.4, ha="left",
                color="#4a5568", fontweight="bold")
    ax.set_title("Selectivity support is pair-by-pair, not target-family-wide\n"
                 "Intervals resample fixed out-of-fold predictions without refitting — "
                 "they ignore scaffold-group dependence",
                 fontsize=10.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.25, 1.0)   # never draw the axis under the label columns
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    if caveated:
        # Opaque background and above the axis in z-order, for the same reason the threshold labels in
        # fig_validation carry one: this footnote spans the plotting area, so the zero rule and the
        # x-grid were drawn through its text. A caveat is the last thing that should be hard to read.
        ax.annotate("†  interval excludes zero, but the positive class is under 3% — "
                    "every classification metric on that pair is unstable.\n"
                    "    \"Supported\" is a statement about the regression interval, "
                    "not a clean bill of health.",
                    (-0.25, -0.92), fontsize=7.2, color="#4a5568", va="center", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2))
    fig.tight_layout()
    p = OUT / "frozen_selectivity.png"
    fig.savefig(p, dpi=200, metadata={**NOMETA, PROV_DIGEST_KEY: _prov_digest()}, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_attrition(data: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    brd4_note = ""
    for ax, (a, b, title) in zip(
        axes,
        [("jak1", "jak1_sensitivity", "JAK1"), ("brd4", "brd4_sensitivity", "BRD4")],
        strict=True,
    ):
        per_a = data[a]["m"]["cohort"]["as_run"]["per_target"]
        per_b = data[b]["m"]["cohort"]["as_run"]["per_target"]
        targets = list(per_a)
        x = np.arange(len(targets))
        kept_a = [per_a[t]["activity_attrition"]["compounds_out"] for t in targets]
        kept_b = [per_b[t]["activity_attrition"]["compounds_out"] for t in targets]
        ax.bar(x - 0.2, kept_b, 0.4, label="pooled (sensitivity)",
               color="#cbd5e0", edgecolor="black", lw=0.5)
        ax.bar(x + 0.2, kept_a, 0.4, label="restricted (headline)",
               color="#2b6cb0", edgecolor="black", lw=0.5)
        for xi, (ka, kb) in enumerate(zip(kept_a, kept_b, strict=True)):
            pct = 100.0 * (1 - ka / kb) if kb else 0.0
            ax.annotate(f"−{pct:.0f}%", (xi + 0.2, ka), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.4, color="#2b6cb0")
        # These bars are the PER-TARGET count after the cohort filter but BEFORE final dedup and
        # drug-likeness. The number the model actually trains on is cohort.attrition.primary_compounds,
        # and the two percentages differ (BRD4: -56% here, -53% on the trained set). Stating one while
        # a reader can compute the other is how a figure ends up contradicting its own caption.
        n_a = data[a]["m"]["cohort"]["attrition"]["primary_compounds"]
        n_b = data[b]["m"]["cohort"]["attrition"]["primary_compounds"]
        drop = 100.0 * (1 - n_a / n_b) if n_b else 0.0
        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=8.6)
        # In the title, not floating in the axes: an annotation placed by axes fraction collides with
        # whatever the legend does, and the legend has to sit where the bars are shortest.
        ax.set_title(f"{title} — distinct compounds per target after the cohort filter\n"
                     f"model trains on {n_a:,} vs {n_b:,}  (−{drop:.0f}%)",
                     fontsize=9.4, loc="left")
        ax.set_ylabel("compounds (pre-dedup, pre-druglike)")
        ax.legend(fontsize=7.8, frameon=False, loc="upper right")
        _style(ax)
        if a == "brd4":
            # DERIVED, not written by hand. This line previously read "BRD4 comparator attrition
            # (BRD2 -72%, BRD3 -84%) is why only one pair is supported" -- a withdrawn conclusion (the
            # frozen result is that NO pair is supported) carrying two percentages that contradicted
            # this figure's own bar labels and every document. A hardcoded caption cannot be checked by
            # the documentation verifier, which reads prose, so it survived a correction that changed
            # the result it asserted. Both the percentages and the pair count now come from the same
            # records the bars do.
            comp = [(t, 100.0 * (1 - ka / kb) if kb else 0.0)
                    for t, ka, kb in zip(targets, kept_a, kept_b, strict=True)
                    if t.upper() != "BRD4"]
            sel = data[a].get("s") or {}
            pairs = (sel.get("direct_delta_scaffold_cv") or {})
            n_sup = sum(1 for v in pairs.values()
                        if isinstance(v, dict) and (v.get("support") or {}).get("supported"))
            verdict = ("no pair is supported" if n_sup == 0
                       else f"only {n_sup} of {len(pairs)} pairs is supported" if n_sup == 1
                       else f"{n_sup} of {len(pairs)} pairs are supported")
            brd4_note = ("BRD4 comparator attrition ("
                         + ", ".join(f"{t} −{p:.0f}%" for t, p in comp)
                         + f") is why {verdict}")
    fig.suptitle("What the cohort restriction costs, per target\n"
                 "bars are pre-dedup per-target counts; each subtitle gives the trained-on set.\n"
                 f"{brd4_note}",
                 fontsize=10.2, x=0.01, ha="left")
    fig.tight_layout()
    p = OUT / "frozen_attrition.png"
    fig.savefig(p, dpi=200, metadata={**NOMETA, PROV_DIGEST_KEY: _prov_digest()}, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_cohort_mix(data: dict) -> Path:
    """The composition that motivated cohorts at all: a single-protein target is a mixture of assays.

    Both denominators are plotted side by side, because quoting one while implying the other is exactly
    the error this figure replaced: BRD4 is 58.8% cell-based by assay count and 15.8% by measurement
    count, and an earlier draft reported ~5%.
    """
    comps = {}
    for panel in ("jak1", "brd4"):
        f = PROV / panel / "composition.json"
        if not f.exists():
            raise SystemExit("no composition.json — run scripts/derive_composition.py first")
        comps[panel] = json.loads(f.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    cols = {"biochemical": "#2b6cb0", "domain_1": "#2c7a7b", "domain_2": "#805ad5",
            "cell_based": "#c05621", "unmatched": "#a0aec0",
            "full_length": "#4a5568", "pseudokinase": "#718096"}
    for ax, panel in zip(axes, ("jak1", "brd4"), strict=True):
        c = comps[panel]
        labs = [k for k in c["by_label"]][::-1]
        y = np.arange(len(labs))
        rec = [c["by_label"][k]["pct_records"] for k in labs]
        asy = [c["by_label"][k]["pct_assays"] for k in labs]
        ax.barh(y + 0.2, rec, 0.4, color=[cols.get(k, "#cbd5e0") for k in labs],
                edgecolor="black", lw=0.5, label="% of IC50 measurements")
        ax.barh(y - 0.2, asy, 0.4, color="white", edgecolor="black", lw=0.5, hatch="////",
                label="% of assays")
        for yi, k in enumerate(labs):
            r = c["by_label"][k]
            ax.annotate(f"{r['pct_records']:.1f}%", (r["pct_records"], yi + 0.2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.2)
            ax.annotate(f"{r['pct_assays']:.1f}%", (r["pct_assays"], yi - 0.2),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7.2, color="#4a5568")
        ax.set_yticks(y)
        # Medians live in the tick label, not as floating annotations: anchoring them to the axis edge
        # put them on top of the long bars, which is the one place a reader is already looking.
        ax.set_yticklabels(
            [f"{k}\n median {c['by_label'][k]['median_pIC50_curated']:.2f}"
             if c["by_label"][k]["median_pIC50_curated"] is not None else k
             for k in labs],
            fontsize=8.0,
        )
        ax.set_xlim(0, max(max(rec), max(asy)) * 1.30)
        ax.set_xlabel("% (two different denominators)")
        ax.set_title(f"{c['target']} — {c['n_assays']:,} assays, "
                     f"{c['n_activity_records']:,} IC50 records", fontsize=9.8, loc="left")
        ax.legend(fontsize=7.4, frameon=False, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    gaps = "; ".join(
        f"{comps[p]['target']} {k.replace('_', ' ')} {v:+.2f}"
        for p in ("jak1", "brd4") for k, v in comps[p]["median_gaps_log_units"].items()
    )
    fig.suptitle("A single-protein ChEMBL target is not a single assay\n"
                 f"curated median gaps (log units): {gaps}",
                 fontsize=10.2, x=0.01, ha="left")
    fig.tight_layout()
    p = OUT / "frozen_cohort_mix.png"
    fig.savefig(p, dpi=200, metadata={**NOMETA, PROV_DIGEST_KEY: _prov_digest()}, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fail if any figure is absent or older than its provenance record")
    args = ap.parse_args()

    data = _load()
    if args.check:
        want = _prov_digest()
        names = ("frozen_validation.png", "frozen_selectivity.png",
                 "frozen_attrition.png", "frozen_cohort_mix.png")
        missing = [n for n in names if not (OUT / n).exists()]
        stale = [n for n in names
                 if (OUT / n).exists() and _figure_digest(OUT / n) != want]
        if missing or stale:
            for n in missing:
                print(f"  MISSING  {n}")
            for n in stale:
                print(f"  STALE    {n} (drawn from different provenance records)")
            print("\n  regenerate: python scripts/make_frozen_figures.py")
            return 1
        print(f"  4 figure(s) present, each stamped with the provenance digest they depict "
              f"({want[:12]}\u2026).")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("FROZEN FIGURES — every value read from provenance/")
    print("=" * 88)
    for fn in (fig_validation, fig_selectivity, fig_attrition, fig_cohort_mix):
        p = fn(data)
        print(f"  {p.relative_to(REPO)}  ({p.stat().st_size:,} bytes)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
