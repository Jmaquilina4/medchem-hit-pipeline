"""Box validation must be target-agnostic and must actually fail.

These checks began as a throwaway script counting atoms inside one box for one target. That is an
anecdote, not a check. A box offset from the real site, or too small to hold the pose it was derived
from, produces a complete campaign with plausible scores and no error anywhere — so each failure mode
below is exercised, not just the passing case.
"""

from __future__ import annotations

import pytest

from medchem.structure.box import parse_residue_spec, validate_box


def _pdb(site_resname: str = "ASN", site_num: int = 140, lig_far: bool = False) -> str:
    lig_x = 100.0 if lig_far else 0.0
    rows = [
        f"ATOM      1  N   {site_resname} A{site_num:4d}       1.000   0.000   0.000  1.00 20.00",
        f"ATOM      2  CA  {site_resname} A{site_num:4d}       2.000   0.000   0.000  1.00 20.00",
        "ATOM      3  N   GLY A 200      60.000  60.000  60.000  1.00 20.00",
        "ATOM      4  CA  GLY A 201      61.000  61.000  61.000  1.00 20.00",
        "ATOM      5  CA  GLY A 202      62.000  62.000  62.000  1.00 20.00",
        f"HETATM  100  C1  LIG A 400    {lig_x:8.3f}   0.000   0.000  1.00 20.00",
        "HETATM  101  C2  LIG A 400       1.000   1.000   1.000  1.00 20.00",
        "END",
    ]
    return "\n".join(rows)


def test_a_good_box_passes():
    r = validate_box(_pdb(), center=(1.0, 0.5, 0.5), size=(12.0, 12.0, 12.0),
                     box_ligand_resname="LIG", site_residue="Asn140")
    assert r.passed, r.reasons
    assert r.ligand_atoms_inside == r.ligand_atoms_total == 2
    assert r.site_residue is not None
    assert r.site_residue["resname_in_structure"] == "ASN"


def test_box_too_small_for_its_own_ligand_fails():
    """If the pose that DEFINED the box does not fit inside it, docking cannot reproduce that pose."""
    r = validate_box(_pdb(lig_far=True), center=(1.0, 0.5, 0.5), size=(12.0, 12.0, 12.0),
                     box_ligand_resname="LIG", site_residue="Asn140")
    assert not r.passed
    assert any("too small for its own pose" in x for x in r.reasons)


def test_declared_site_residue_outside_the_box_fails():
    """The multi-domain trap: a box centred on the wrong pocket still scores everything happily."""
    r = validate_box(_pdb(), center=(61.0, 61.0, 61.0), size=(8.0, 8.0, 8.0),
                     site_residue="Asn140")
    assert not r.passed
    assert any("OUTSIDE the box" in x for x in r.reasons)


def test_residue_type_mismatch_fails():
    """Config says Asn140 but the crystal construct has a different residue there: a numbering
    mismatch, which would otherwise silently validate a box against the wrong site."""
    r = validate_box(_pdb(site_resname="LEU"), center=(1.0, 0.5, 0.5), size=(12.0, 12.0, 12.0),
                     box_ligand_resname="LIG", site_residue="Asn140")
    assert not r.passed
    assert any("is LEU in this structure, not ASN" in x for x in r.reasons)


def test_absent_site_residue_fails_with_a_diagnosis():
    r = validate_box(_pdb(), center=(1.0, 0.5, 0.5), size=(12.0, 12.0, 12.0), site_residue="Asn999")
    assert not r.passed
    assert any("absent from this structure" in x for x in r.reasons)


def test_box_swallowing_the_fold_fails():
    """A box containing the whole protein makes docking a shape-matching exercise."""
    r = validate_box(_pdb(), center=(30.0, 30.0, 30.0), size=(200.0, 200.0, 200.0),
                     box_ligand_resname="LIG")
    assert not r.passed
    assert any("this is the fold, not a site" in x for x in r.reasons)


def test_box_off_the_protein_fails():
    r = validate_box(_pdb(), center=(500.0, 500.0, 500.0), size=(10.0, 10.0, 10.0))
    assert not r.passed
    assert any("off the protein" in x for x in r.reasons)


def test_missing_box_ligand_is_reported():
    r = validate_box(_pdb(), center=(1.0, 0.5, 0.5), size=(12.0, 12.0, 12.0),
                     box_ligand_resname="ZZZ")
    assert not r.passed
    assert any("not found" in x for x in r.reasons)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("Asn140", ("ASN", 140)), ("ASN 140", ("ASN", 140)), ("N140", ("ASN", 140)),
     ("leu-959", ("LEU", 959)), ("140", (None, 140))],
)
def test_residue_specs_humans_actually_write(spec, expected):
    assert parse_residue_spec(spec) == expected


def test_unparseable_residue_spec_raises():
    with pytest.raises(ValueError, match="cannot parse residue spec"):
        parse_residue_spec("the asparagine one")
