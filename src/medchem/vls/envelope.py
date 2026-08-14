"""Derive a physchem screening envelope from a target's OWN actives, instead of borrowing one.

The screening window was tuned on the flagship target: MW 300–460, cLogP ≤ 3.5. Measured against a
bromodomain, that window rejects **5 of 7** of its reference compounds — mostly on lipophilicity,
because an acetyl-lysine pocket is hydrophobic. Inherited unchanged it would have left the screen
structurally unable to find the target's own known chemistry *while reporting success*.

The obvious repair — hand-widen the numbers for the new target — is what was done first, and it is only
marginally better: those bounds came from seven compounds picked by a human. This module replaces both
with a derivation from the target's own curated potency data.

Two design constraints matter more than the mechanics:

* **Derive WIDER than the observed actives.** A window fitted tightly to known actives *defines* the
  chemotype, so the screen can only rediscover it — the same trap as a one-sided applicability-domain
  reward. ``margin`` expands each bound by a fraction of the observed spread. The filter's job is to
  exclude chemistry implausible for the class, not to describe the compounds already known.
* **"Potent" is relative.** Selecting actives with an absolute pIC50 cut imports the same borrowed
  assumption one level down: on a bromodomain, a threshold of 8.0 discards JQ1 (~7.1), the field's
  benchmark compound. The cut is a *quantile* of this target's own distribution, and the resolved
  absolute value is recorded so a reader can still see what was used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Descriptors the screening filter bounds, mapped to the config keys they populate. Only these are
# derived: counts (HBD/HBA/rotatable bonds) are left to explicit config because their distributions are
# discrete and small-integer, where a 5th-percentile bound is noise rather than a limit.
DERIVED_BOUNDS = {
    "MolWt": ("mw_min", "mw_max"),
    "MolLogP": (None, "logp_max"),
    "TPSA": (None, "tpsa_max"),
}


@dataclass
class Envelope:
    """A derived screening window plus everything needed to defend it."""

    bounds: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    admits_references: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "bounds": self.bounds,
            "provenance": self.provenance,
            "admits_references": self.admits_references,
        }


def resolve_potency_cut(
    values: list[float], *, quantile: float | None, absolute: float | None
) -> tuple[float, str]:
    """Resolve the "potent active" threshold, preferring a quantile over an absolute value.

    Returns ``(value, how)``. An explicit absolute wins only when no quantile is given, so a config can
    still pin a literal threshold — but it has to say so rather than inherit one.
    """
    if quantile is not None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"potency quantile must be in (0, 1); got {quantile}")
        if not values:
            raise ValueError("cannot resolve a potency quantile from an empty distribution")
        import numpy as np

        cut = float(np.quantile(np.asarray(values, dtype=float), quantile))
        return cut, f"quantile {quantile:.2f} of this target's own actives (n={len(values)})"
    if absolute is None:
        raise ValueError("neither a potency quantile nor an absolute threshold was supplied")
    return float(absolute), "explicit absolute threshold from config"


def derive_envelope(
    descriptors: dict[str, list[float]],
    *,
    quantiles: tuple[float, float] = (0.05, 0.95),
    margin: float = 0.15,
    n_actives: int = 0,
    potency_cut: float | None = None,
    potency_cut_how: str = "",
) -> Envelope:
    """Bound each descriptor by a quantile range of the actives, then widen by ``margin``.

    ``descriptors`` maps an RDKit descriptor name to its values across the selected actives. Quantiles
    rather than min/max because a single unusual compound should not set a screening bound, and the
    outer 5% of a real distribution is where the odd salt form and the mis-annotated entry live.
    """
    import numpy as np

    lo_q, hi_q = quantiles
    if not 0.0 <= lo_q < hi_q <= 1.0:
        raise ValueError(f"envelope quantiles must satisfy 0 <= lo < hi <= 1; got {quantiles}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0; got {margin}")

    bounds: dict[str, float] = {}
    observed: dict[str, dict[str, float]] = {}
    for name, (min_key, max_key) in DERIVED_BOUNDS.items():
        vals = [v for v in descriptors.get(name, []) if v is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        lo, hi = float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q))
        span = hi - lo
        pad = span * margin
        # Widen OUTWARD. A window fitted to the actives defines the chemotype; the filter should only
        # exclude what is implausible for the class.
        wide_lo, wide_hi = lo - pad, hi + pad
        observed[name] = {"q_lo": round(lo, 3), "q_hi": round(hi, 3), "pad": round(pad, 3),
                          "n": len(vals)}
        if min_key:
            bounds[min_key] = round(max(0.0, wide_lo), 3)
        if max_key:
            bounds[max_key] = round(wide_hi, 3)

    return Envelope(
        bounds=bounds,
        provenance={
            "derived_from": "this target's own curated actives",
            "n_actives": n_actives,
            "potency_cut": potency_cut,
            "potency_cut_how": potency_cut_how,
            "envelope_quantiles": [lo_q, hi_q],
            "margin": margin,
            "observed": observed,
            "note": (
                "bounds are the quantile range WIDENED by margin, so the window excludes chemistry "
                "implausible for the class rather than describing the actives already known"
            ),
        },
    )


# Lead-likeness is conventionally about SIZE and LIPOPHILICITY. TPSA is a permeability constraint, not
# a lead-likeness one, and a reference set is usually small enough that its TPSA range is an accident of
# which compounds reached the clinic: BRD4's 7 references span 69-94, which would cap the screen at 97
# against a defensible 140. So the sliding ceiling applies to these descriptors only, and the actives
# envelope (or explicit config) governs the rest.
CEILING_DESCRIPTORS = ("MolWt", "MolLogP")


def effective_margin(base: float, n_references: int, *, inflation: float = 2.0) -> float:
    """Widen the margin when the reference set is small, because the range is then less certain.

    A flat margin treats a 5-compound range as being as trustworthy as a 50-compound one. It is not:
    with few references the observed min and max are an artefact of which compounds happen to exist, so
    the window they imply should carry more headroom, not the same amount.

    ``base * (1 + inflation / sqrt(n))`` — standard-error-shaped, so the inflation decays as the set
    grows and vanishes in the limit. With base 0.15 and inflation 2.0: n=5 -> 0.28, n=7 -> 0.26,
    n=50 -> 0.19.
    """
    if n_references <= 0:
        return base
    return base * (1.0 + inflation / (n_references ** 0.5))


def derive_reference_ceiling(
    reference_descriptors: dict[str, dict[str, float]],
    *,
    margin: float = 0.15,
    descriptors: tuple[str, ...] = CEILING_DESCRIPTORS,
    inflation: float = 2.0,
) -> dict[str, Any]:
    """A lead-likeness ceiling that SLIDES with the target's representative compounds.

    The alternative was intersecting the actives-derived envelope with hand-written numbers — but those
    numbers are themselves a borrowed constant, which is the failure this module exists to remove. The
    representative set (the target's clinical and tool compounds) is the right anchor: it encodes what a
    successful compound for *this* target looks like, rather than everything ever synthesised against it.

    Bounds are the references' own range widened by a fraction of that range's span, so the ceiling
    cannot reject the compounds that defined it.

    This reproduces both hand-tuned windows without a constant, which is the check that matters:

        JAK1  references MW 312-426  ->  295-442     (hand-written 300-460)
        BRD4  references MW 347-492  ->  326-514     (hand-written 330-520)

    and it slides in the right direction on its own — BRD4's ceiling sits ~72 Da higher because BET
    chemotypes are larger, a difference that was previously a human decision per target.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0; got {margin}")
    eff = effective_margin(margin, len(reference_descriptors), inflation=inflation)
    out: dict[str, Any] = {"bounds": {}, "observed": {}, "n_references": len(reference_descriptors),
                           "base_margin": margin, "effective_margin": round(eff, 4),
                           "inflation": inflation}
    if not reference_descriptors:
        out["note"] = "no reference compounds configured: no sliding ceiling is derivable"
        return out
    if len(reference_descriptors) < 4:
        out["caveat"] = (
            f"only {len(reference_descriptors)} reference compound(s): the range is narrow by accident "
            f"of which compounds exist, so this ceiling is weakly determined"
        )
    for desc, (min_key, max_key) in DERIVED_BOUNDS.items():
        if desc not in descriptors:
            continue
        vals = [d[desc] for d in reference_descriptors.values() if d.get(desc) is not None]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * eff
        out["observed"][desc] = {"min": round(lo, 3), "max": round(hi, 3), "pad": round(pad, 3)}
        if min_key:
            out["bounds"][min_key] = round(max(0.0, lo - pad), 3)
        if max_key:
            out["bounds"][max_key] = round(hi + pad, 3)
    out["note"] = (
        "range of the target's own representative compounds, widened by a fraction of its span; "
        "slides with the target instead of encoding one target's lead-likeness as a constant"
    )
    return out


def intersect_bounds(envelope: dict[str, float], ceiling: dict[str, float]) -> dict[str, float]:
    """Tightest of the two windows per bound: maxima take the lower, minima the higher.

    The actives envelope supplies chemotype plausibility (wide, because optimised compounds drift
    large); the reference ceiling supplies lead-likeness (tighter). Neither alone is right.
    """
    out = dict(envelope)
    for key, val in ceiling.items():
        if key not in out:
            out[key] = val
        elif key.endswith("_max"):
            out[key] = min(out[key], val)
        elif key.endswith("_min"):
            out[key] = max(out[key], val)
    return out


def check_admits(
    envelope_bounds: dict[str, float], reference_descriptors: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Does the window admit the target's own reference compounds?

    This is the guard, and it is target-agnostic: a physchem filter that rejects the compounds you
    benchmark against is wrong regardless of target. It caught the inherited-window failure in both
    directions.
    """
    rejected: dict[str, list[str]] = {}
    for name, d in reference_descriptors.items():
        why = []
        mw, logp, tpsa = d.get("MolWt"), d.get("MolLogP"), d.get("TPSA")
        if mw is not None:
            if "mw_min" in envelope_bounds and mw < envelope_bounds["mw_min"]:
                why.append(f"MW {mw:.0f} < {envelope_bounds['mw_min']:.0f}")
            if "mw_max" in envelope_bounds and mw > envelope_bounds["mw_max"]:
                why.append(f"MW {mw:.0f} > {envelope_bounds['mw_max']:.0f}")
        if logp is not None and "logp_max" in envelope_bounds and logp > envelope_bounds["logp_max"]:
            why.append(f"cLogP {logp:.2f} > {envelope_bounds['logp_max']:.2f}")
        if tpsa is not None and "tpsa_max" in envelope_bounds and tpsa > envelope_bounds["tpsa_max"]:
            why.append(f"TPSA {tpsa:.0f} > {envelope_bounds['tpsa_max']:.0f}")
        if why:
            rejected[name] = why
    return {
        "n_references": len(reference_descriptors),
        "n_admitted": len(reference_descriptors) - len(rejected),
        "rejected": rejected,
        "passed": not rejected,
    }
