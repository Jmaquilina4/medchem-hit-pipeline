"""Structure tier: docking-box derivation + ligand prep."""

from __future__ import annotations

import pytest

from medchem.structure.prep import DockingBox, prepare_ligand

# Minimal synthetic PDB: a 4-atom HETATM ligand spanning 6 A in x, 2 A in y, 0 in z,
# plus a decoy ligand that must be ignored.
_PDB = """\
ATOM      1  CA  LEU A 959      99.000  99.000  99.000  1.00 10.00           C
HETATM    2  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C
HETATM    3  C2  LIG A   1       6.000   0.000   0.000  1.00 10.00           C
HETATM    4  C3  LIG A   1       0.000   2.000   0.000  1.00 10.00           C
HETATM    5  C4  LIG A   1       6.000   2.000   0.000  1.00 10.00           C
HETATM    6  O1  HOH A   2      50.000  50.000  50.000  1.00 10.00           O
"""


@pytest.fixture
def pdb(tmp_path):
    p = tmp_path / "mini.pdb"
    p.write_text(_PDB, encoding="utf-8")
    return p


def test_box_centres_on_the_named_ligand_only(pdb):
    box = DockingBox.from_pdb_ligand(pdb, "LIG")
    # centroid of the 4 LIG atoms — the water and the protein CA must not shift it
    assert box.center == pytest.approx((3.0, 1.0, 0.0))


def test_box_is_clamped_to_min_size_for_a_small_reference_ligand(pdb):
    # extent 6 x 2 x 0 + 2*3.5 padding = 13 x 9 x 7, all below the 22 A floor: a small
    # co-crystal ligand must still yield a box able to hold lead-like candidates.
    box = DockingBox.from_pdb_ligand(pdb, "LIG", padding=3.5, min_size=22.0)
    assert box.size == pytest.approx((22.0, 22.0, 22.0))


def test_box_grows_past_the_floor_for_a_large_ligand(pdb):
    box = DockingBox.from_pdb_ligand(pdb, "LIG", padding=10.0, min_size=22.0)
    assert box.size[0] == pytest.approx(26.0)   # 6 + 2*10
    assert box.size[1] == pytest.approx(22.0)   # 2 + 2*10 = 22 -> at the floor


def test_missing_ligand_is_an_error_not_a_silent_default(pdb):
    with pytest.raises(ValueError, match="NOPE"):
        DockingBox.from_pdb_ligand(pdb, "NOPE")


def test_prepare_ligand_emits_pdbqt_with_torsions():
    pytest.importorskip("meeko")
    # upadacitinib — the JAK1 fast-follower reference
    pdbqt = prepare_ligand("CC[C@@H]1CN(C(=O)NCC(F)(F)F)C[C@@H]1c1cnc2cnc3[nH]ccc3n12")
    assert pdbqt is not None
    assert "ROOT" in pdbqt
    assert sum(1 for ln in pdbqt.splitlines() if ln.startswith("ATOM")) > 20
    assert sum(1 for ln in pdbqt.splitlines() if ln.startswith("BRANCH")) >= 1


def test_prepare_ligand_returns_none_on_bad_input_rather_than_raising():
    pytest.importorskip("meeko")
    # a 10^6-ligand run must not die on one bad row
    assert prepare_ligand("$$not-a-smiles$$") is None


def test_prepare_ligand_is_deterministic_for_a_fixed_seed():
    pytest.importorskip("meeko")
    a = prepare_ligand("CCOc1ccccc1", seed=7)
    b = prepare_ligand("CCOc1ccccc1", seed=7)
    assert a == b
