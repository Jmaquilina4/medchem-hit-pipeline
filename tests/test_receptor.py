"""Receptor preparation is a pipeline INPUT, so its provenance has to be right.

The resolution parser is tested first because it was wrong in a way that looked fine: it scanned the
whole `REMARK   2 RESOLUTION.  1.24 ANGSTROMS.` line for a float and found the "2" of "REMARK   2",
reporting every structure as 2.0 A. It went unnoticed on the first entry tried because that entry's
true resolution was in fact 2.0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medchem.structure.receptor import _centroid, _het_residues, _pick_box_ligand, _protein_only


def _fake_pdb(resolution_line: str = "REMARK   2 RESOLUTION.    1.24 ANGSTROMS.") -> str:
    return "\n".join([
        "TITLE     A TEST STRUCTURE",
        resolution_line,
        "ATOM      1  N   SER A  42      10.000  10.000  10.000  1.00 20.00           N",
        "ATOM      2  CA  SER A  42      11.000  10.000  10.000  1.00 20.00           C",
        "HETATM  100  O   HOH A 300      50.000  50.000  50.000  1.00 20.00           O",
        "HETATM  101  ZN  ZN  A 301      60.000  60.000  60.000  1.00 20.00          ZN",
        "HETATM  200  C1  LIG A 400       0.000   0.000   0.000  1.00 20.00           C",
        "HETATM  201  C2  LIG A 400       2.000   0.000   0.000  1.00 20.00           C",
        "HETATM  202  C3  LIG A 400       1.000   3.000   0.000  1.00 20.00           C",
        "HETATM  300  C1  SML A 500      20.000  20.000  20.000  1.00 20.00           C",
        "END",
    ])


def test_resolution_is_parsed_after_the_keyword_not_from_the_remark_number(tmp_path, monkeypatch):
    from medchem.structure import receptor as mod

    dest = tmp_path / "x.pdb"
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: _FakeResp(_fake_pdb()))
    prov = mod._fetch_pdb("1abc", dest)
    assert prov["resolution_angstrom"] == 1.24, "the '2' in 'REMARK   2' must not be read as resolution"
    assert prov["pdb_id"] == "1ABC"
    assert prov["sha256"] and prov["bytes"] > 0


class _FakeResp:
    def __init__(self, text: str) -> None:
        self._b = text.encode()

    def read(self) -> bytes:
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a) -> None:
        return None


def test_solvent_and_ions_are_never_chosen_as_the_box_ligand():
    het = _het_residues(_fake_pdb())
    names = {k[0] for k in het}
    assert "HOH" not in names and "ZN" not in names
    assert {"LIG", "SML"} <= names


def test_largest_heteroresidue_wins_when_config_names_none():
    (resname, _chain, _seq), xyz = _pick_box_ligand(_fake_pdb(), None)
    assert resname == "LIG" and len(xyz) == 3


def test_named_ligand_is_honoured_even_when_smaller():
    """A named ligand must win over a bigger one: the config author knows which pocket matters."""
    (resname, _c, _s), xyz = _pick_box_ligand(_fake_pdb(), "SML")
    assert resname == "SML" and len(xyz) == 1


def test_missing_named_ligand_fails_loudly_with_what_is_available():
    """Silently substituting another ligand would move the box to a different site, and no downstream
    number would reveal it."""
    with pytest.raises(ValueError, match="not in this entry"):
        _pick_box_ligand(_fake_pdb(), "NOPE")


def test_apo_structure_is_rejected_rather_than_boxed_arbitrarily():
    apo = "\n".join(["ATOM      1  N   SER A  42      10.0  10.0  10.0  1.00 20.00           N", "END"])
    with pytest.raises(ValueError, match="apo"):
        _pick_box_ligand(apo, None)


def test_box_centre_is_the_ligand_centroid():
    assert _centroid([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 3.0, 0.0)]) == (1.0, 1.0, 0.0)


def test_protein_extraction_drops_heteroatoms_and_water():
    """_protein_only now returns (text, n_waters_kept): retention is opt-in, so the default path must
    still strip every heteroatom."""
    out, n_waters = _protein_only(_fake_pdb())
    assert n_waters == 0
    assert "HETATM" not in out and "HOH" not in out
    assert out.count("ATOM") == 2


# --- water retention and explicit box centres -------------------------------------------------------

def _pdb_with_waters() -> str:
    return "\n".join([
        "ATOM      1  N   SER A  42      10.000  10.000  10.000  1.00 20.00           N",
        "HETATM  200  C1  LIG A 400       0.000   0.000   0.000  1.00 20.00           C",
        "HETATM  300  O   HOH A 500       1.000   0.000   0.000  1.00 20.00           O",  # 1.0 A
        "HETATM  301  O   HOH A 501       3.000   0.000   0.000  1.00 20.00           O",  # 3.0 A
        "HETATM  302  O   HOH A 502      30.000   0.000   0.000  1.00 20.00           O",  # far
        "END",
    ])


def test_waters_are_dropped_by_default():
    """The right default for an ATP site, and the reason it must be configurable rather than assumed."""
    from medchem.structure.receptor import _protein_only

    text, n = _protein_only(_pdb_with_waters())
    assert n == 0
    assert "HOH" not in text


def test_waters_within_the_cutoff_of_the_ligand_are_retained():
    """A bromodomain recognises acetyl-lysine THROUGH a conserved water network; docking a dry BET
    pocket scores a site that does not exist. Measured consequence: on BRD4/4J3I the dry-vs-hydrated
    ranking of the same 20 ligands correlates at Spearman 0.185."""
    from medchem.structure.receptor import _protein_only

    text, n = _protein_only(
        _pdb_with_waters(), keep_waters_within=4.0, ligand_xyz=[(0.0, 0.0, 0.0)]
    )
    assert n == 2, "the two waters within 4 A should be kept, the distant one dropped"
    assert text.count("HOH") == 2


def test_water_cutoff_measures_to_the_ligand_not_the_box_centre():
    """The ligand traces the pocket surface; the network that matters is the shell touching it."""
    from medchem.structure.receptor import _protein_only

    _text, near = _protein_only(_pdb_with_waters(), keep_waters_within=4.0,
                                ligand_xyz=[(30.0, 0.0, 0.0)])
    assert near == 1, "with the ligand moved, a different water is in contact"


def test_explicit_box_center_is_validated_as_three_numbers():
    """The apo escape hatch. A malformed centre must not fall back to a ligand centroid that does not
    exist -- the error message used to name this option before it was implemented."""
    import pytest
    from pydantic import ValidationError

    from medchem.config import Config

    cfg = Config.model_validate({"structure": {"box_center": [1.0, 2.0, 3.0]}})
    assert cfg.structure.box_center == [1.0, 2.0, 3.0]
    with pytest.raises(ValidationError, match="must be \\[x, y, z\\]"):
        Config.model_validate({"structure": {"box_center": [1.0, 2.0]}})


# ---------------------------------------------------------------------------------------------
# hinge_residue / anchor_residue are ONE slot in two vocabularies. The receptor stage resolved
# them as `anchor_residue or hinge_residue`, so a config supplying both had its hinge silently
# dropped -- and the site residue is precisely what proves the box is on the intended pocket.
# ---------------------------------------------------------------------------------------------

def test_anchor_only_resolves_to_the_anchor():
    from medchem.config import StructureConfig

    st = StructureConfig(anchor_residue="Asn140")
    assert st.site_residue == "Asn140"


def test_hinge_only_resolves_to_the_hinge():
    """The case the `or` precedence could not get wrong, included so the pair is covered."""
    from medchem.config import StructureConfig

    st = StructureConfig(hinge_residue="Leu959")
    assert st.site_residue == "Leu959"


def test_neither_resolves_to_none_and_is_allowed():
    """A target may legitimately assert no site residue; the box check then simply has no site term."""
    from medchem.config import StructureConfig

    assert StructureConfig().site_residue is None


def test_supplying_both_vocabularies_is_rejected():
    """The silent case: `anchor or hinge` returned the anchor and dropped the hinge with no signal."""
    from pydantic import ValidationError

    from medchem.config import StructureConfig

    with pytest.raises(ValidationError, match="names BOTH a hinge residue"):
        StructureConfig(hinge_residue="Leu959", anchor_residue="Asn140")


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_site_residue_is_rejected(blank: str):
    """A blank name asserts nothing and would be treated as unset, so it reads as a site check that
    is silently not happening -- worse than omitting the key.

    Both fields are constructed explicitly rather than through ``**{field: value}``: the dynamic form
    defeats the type checker on every OTHER field of the model, which turns one real check into six
    spurious errors.
    """
    from pydantic import ValidationError

    from medchem.config import StructureConfig

    with pytest.raises(ValidationError, match="empty value"):
        StructureConfig(hinge_residue=blank)
    with pytest.raises(ValidationError, match="empty value"):
        StructureConfig(anchor_residue=blank)


def test_the_receptor_stage_reads_the_resolved_property_not_an_or_expression():
    """Consumer-level. The resolution must live where the validation does, or the two can diverge."""
    import ast

    src = Path(__file__).resolve().parent.parent / "src" / "medchem" / "structure" / "receptor.py"
    code = ast.unparse(ast.parse(src.read_text(encoding="utf-8")))
    assert "st.site_residue" in code, "the receptor stage no longer reads the resolved property"
    assert "anchor_residue or" not in code and "hinge_residue or" not in code, (
        "the truthiness precedence is back; a config supplying both would silently lose one"
    )


def test_every_shipped_config_names_at_most_one_site_vocabulary():
    """The shipped panels each name exactly one, which is why this fix moves no number."""
    import yaml

    repo = Path(__file__).resolve().parent.parent
    for cfg_path in sorted((repo / "configs").glob("*.yaml")):
        st = (yaml.safe_load(cfg_path.read_text()) or {}).get("structure") or {}
        named = [k for k in ("hinge_residue", "anchor_residue") if st.get(k) is not None]
        assert len(named) <= 1, f"{cfg_path.name} names both: {named}"
