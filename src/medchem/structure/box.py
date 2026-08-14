"""Docking-box validation — target-agnostic.

The box centre and size are the most consequential numbers in a docking campaign and the easiest to
get silently wrong. A box offset from the real site, or too small to contain the pose it was derived
from, produces a complete run with plausible-looking scores and no error anywhere.

These checks were originally done by hand, once, on one target: a throwaway script counting atoms
inside a box. That is not a check, it is an anecdote. Everything here works from a PDB, a box, and an
optionally declared site residue, so it applies to any target and any pocket.

What is checked, and what each failure would mean:

* **the box contains the ligand that defined it.** If the co-crystal pose does not fit inside its own
  box, the box is too small and docking is being asked to reproduce a pose it cannot physically place.
* **a declared site residue exists, is the residue type declared, and lies inside the box.** This
  catches the two ways a site declaration goes wrong: a numbering mismatch between the config and the
  crystal construct, and a box centred on a different pocket. For a multi-domain protein it is the
  difference between two real sites — BRD4's BD1 and BD2 both bind, and a BD2 co-crystal would centre
  the box on the wrong one.
* **the box encloses a plausible FRACTION of the protein.** Too much and it is not a site, it is the
  fold, which makes docking a shape-matching exercise. Too little and the box is off the protein
  entirely, in solvent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# "Asn140", "ASN140", "N140", "asn-140" -> ("ASN", 140). One-letter codes are accepted because configs
# are written by humans, and rejecting a legible spelling is worse than translating it.
_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
    "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
    "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
Coord = tuple[float, float, float]


@dataclass
class BoxReport:
    """Outcome of validating one box against one structure."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    ligand_atoms_inside: int = 0
    ligand_atoms_total: int = 0
    protein_atoms_inside: int = 0
    protein_atoms_total: int = 0
    protein_fraction_inside: float = 0.0
    site_residue: dict | None = None

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "ligand_atoms_inside": f"{self.ligand_atoms_inside}/{self.ligand_atoms_total}",
            "protein_atoms_inside": f"{self.protein_atoms_inside}/{self.protein_atoms_total}",
            "protein_fraction_inside": round(self.protein_fraction_inside, 4),
            "site_residue": self.site_residue,
        }


def parse_residue_spec(spec: str) -> tuple[str | None, int]:
    """``"Asn140"`` -> ``("ASN", 140)``. Returns ``(None, n)`` when only a number is given."""
    m = re.fullmatch(r"\s*([A-Za-z]{0,3})[\s\-_]*(\d+)\s*", spec or "")
    if not m:
        raise ValueError(
            f"cannot parse residue spec {spec!r}; expected forms like 'Asn140', 'ASN 140' or '140'"
        )
    name, num = m.group(1).upper(), int(m.group(2))
    if not name:
        return None, num
    if len(name) == 1:
        name = _ONE_TO_THREE.get(name, name)
    return name, num


def _inside(p: Coord, center: Coord, size: Coord) -> bool:
    return all(abs(p[i] - center[i]) <= size[i] / 2.0 for i in range(3))


def _xyz(line: str) -> Coord | None:
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None


def validate_box(
    pdb_text: str,
    *,
    center: Coord,
    size: Coord,
    box_ligand_resname: str | None = None,
    site_residue: str | None = None,
    max_protein_fraction: float = 0.60,
    min_protein_fraction: float = 0.02,
) -> BoxReport:
    """Validate a docking box against the structure it was derived from.

    ``max_protein_fraction``/``min_protein_fraction`` are deliberately loose defaults: they exist to
    catch a box that is obviously the whole fold or obviously off the protein, not to encode a
    preferred pocket size. Tightening them per target is a config decision, not a library one.
    """
    rep = BoxReport(passed=True)
    lines = pdb_text.splitlines()

    protein = [p for p in (_xyz(x) for x in lines if x.startswith("ATOM")) if p]
    rep.protein_atoms_total = len(protein)
    rep.protein_atoms_inside = sum(_inside(p, center, size) for p in protein)
    if not protein:
        rep.passed = False
        rep.reasons.append("no protein ATOM records: nothing to dock against")
        return rep
    rep.protein_fraction_inside = rep.protein_atoms_inside / len(protein)

    if rep.protein_fraction_inside > max_protein_fraction:
        rep.passed = False
        rep.reasons.append(
            f"box encloses {rep.protein_fraction_inside:.1%} of protein atoms (> "
            f"{max_protein_fraction:.0%}): this is the fold, not a site"
        )
    if rep.protein_fraction_inside < min_protein_fraction:
        rep.passed = False
        rep.reasons.append(
            f"box encloses only {rep.protein_fraction_inside:.1%} of protein atoms (< "
            f"{min_protein_fraction:.0%}): it is probably off the protein, in solvent"
        )

    if box_ligand_resname:
        lig = [
            p for p in (
                _xyz(x) for x in lines
                if x.startswith("HETATM") and x[17:20].strip() == box_ligand_resname.upper()
            ) if p
        ]
        rep.ligand_atoms_total = len(lig)
        rep.ligand_atoms_inside = sum(_inside(p, center, size) for p in lig)
        if not lig:
            rep.passed = False
            rep.reasons.append(f"ligand {box_ligand_resname!r} not found; cannot confirm the box")
        elif rep.ligand_atoms_inside < rep.ligand_atoms_total:
            rep.passed = False
            rep.reasons.append(
                f"only {rep.ligand_atoms_inside}/{rep.ligand_atoms_total} atoms of the ligand that "
                f"DEFINED this box fall inside it -- the box is too small for its own pose"
            )

    if site_residue:
        want_name, want_num = parse_residue_spec(site_residue)
        atoms = [
            x for x in lines
            if x.startswith("ATOM") and x[22:26].strip().lstrip("-").isdigit()
            and int(x[22:26]) == want_num
        ]
        if not atoms:
            rep.passed = False
            rep.reasons.append(
                f"declared site residue {site_residue!r} (number {want_num}) is absent from this "
                f"structure: config numbering does not match the crystal construct"
            )
            rep.site_residue = {"declared": site_residue, "found": False}
        else:
            found_name = atoms[0][17:20].strip().upper()
            coords = [p for p in (_xyz(x) for x in atoms) if p]
            n_in = sum(_inside(p, center, size) for p in coords)
            rep.site_residue = {
                "declared": site_residue, "found": True, "resname_in_structure": found_name,
                "atoms_inside": f"{n_in}/{len(coords)}",
            }
            if want_name and found_name != want_name:
                rep.passed = False
                rep.reasons.append(
                    f"residue {want_num} is {found_name} in this structure, not {want_name} as "
                    f"declared: the config is describing a different structure or numbering"
                )
            if n_in == 0:
                rep.passed = False
                rep.reasons.append(
                    f"declared site residue {site_residue!r} lies OUTSIDE the box: the box is "
                    f"centred on a different pocket"
                )
    return rep
