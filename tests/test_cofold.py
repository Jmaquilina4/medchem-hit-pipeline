"""Co-folding input guards. Each test corresponds to a documented silent failure.

Co-folding is the tier where bad inputs cost the most, because the model does not complain: a
mismatched MSA is swapped for a dummy, a malformed YAML is skipped while the run continues over zero
records, and an unquoted '#' truncates a nitrile. Every check below runs on CPU at write time, which is
the point — each one converts a wasted GPU reservation into an immediate local error.
"""

from __future__ import annotations

import pytest

from medchem.structure.cofold import assert_msa_matches, sequence_from_pdb, write_boltz_yaml

SEQ = "MKTAYIAKQR"


def _pdb_with_seqres() -> str:
    return "\n".join([
        "SEQRES   1 A   10  MET LYS THR ALA TYR ILE ALA LYS GLN ARG",
        "ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 20.00",
        "ATOM      2  CA  LYS A   2       3.800   0.000   0.000  1.00 20.00",
        "END",
    ])


def test_sequence_prefers_seqres_over_observed_atoms():
    """SEQRES is the construct that was crystallised. Falling back to observed residues silently
    shortens the chain at every disordered loop."""
    seq, source = sequence_from_pdb(_pdb_with_seqres())
    assert seq == SEQ
    assert "SEQRES" in source


def test_sequence_falls_back_to_atoms_and_says_so():
    pdb = "\n".join([
        "ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 20.00",
        "ATOM      2  CA  LYS A   2       3.800   0.000   0.000  1.00 20.00",
        "END",
    ])
    seq, source = sequence_from_pdb(pdb)
    assert seq == "MK"
    assert "no SEQRES" in source


def test_modified_residues_are_translated_not_dropped():
    """MSE (selenomethionine) is common in crystal structures and is a methionine."""
    pdb = "SEQRES   1 A    3  MSE LYS THR\nEND"
    seq, _ = sequence_from_pdb(pdb)
    assert seq == "MKT"


def test_unknown_residue_becomes_X_and_is_reported():
    pdb = "SEQRES   1 A    3  MET ZZZ THR\nEND"
    seq, source = sequence_from_pdb(pdb)
    assert seq == "MXT"
    assert "non-standard" in source


def test_yaml_round_trips_and_declares_the_affinity_binder(tmp_path):
    rec = write_boltz_yaml(tmp_path / "a.yaml", sequence=SEQ, smiles="CCO")
    text = rec.path.read_text()
    assert "binder: L" in text
    assert rec.warnings and "single-sequence mode" in rec.warnings[0]


def test_a_nitrile_survives_because_smiles_is_quoted(tmp_path):
    """An unquoted '#' starts a YAML comment and truncates every nitrile. The parse-back is what
    catches it, so this test would fail loudly rather than produce a shortened molecule."""
    nitrile = "CC(C)(C#N)c1ccccc1"
    rec = write_boltz_yaml(tmp_path / "b.yaml", sequence=SEQ, smiles=nitrile)
    import yaml
    doc = yaml.safe_load(rec.path.read_text())
    lig = next(e["ligand"] for e in doc["sequences"] if "ligand" in e)
    assert lig["smiles"] == nitrile, "the nitrile was truncated at a YAML comment"


def test_msa_path_must_be_absolute(tmp_path):
    """A relative msa path resolves against the process working directory, not the input file."""
    with pytest.raises(ValueError, match="must be ABSOLUTE"):
        write_boltz_yaml(tmp_path / "c.yaml", sequence=SEQ, smiles="CCO", msa_abs="msa.csv")


def test_absolute_msa_lands_as_a_sibling_key_not_folded_into_the_sequence(tmp_path):
    """The eight-space indentation bug: `msa:` becomes a continuation of the sequence scalar, the file
    is skipped, and the run proceeds over zero records while looking busy."""
    msa = tmp_path / "msa.csv"
    msa.write_text(f"key,sequence\n101,{SEQ}\n")
    rec = write_boltz_yaml(tmp_path / "d.yaml", sequence=SEQ, smiles="CCO", msa_abs=str(msa))
    import yaml
    doc = yaml.safe_load(rec.path.read_text())
    prot = next(e["protein"] for e in doc["sequences"] if "protein" in e)
    assert prot["msa"] == str(msa)
    assert prot["sequence"] == SEQ


def test_empty_sequence_or_smiles_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty protein sequence"):
        write_boltz_yaml(tmp_path / "e.yaml", sequence="", smiles="CCO")
    with pytest.raises(ValueError, match="empty SMILES"):
        write_boltz_yaml(tmp_path / "f.yaml", sequence=SEQ, smiles="")


def test_matching_msa_passes(tmp_path):
    msa = tmp_path / "m.csv"
    msa.write_text(f"key,sequence\n101,{SEQ}\n102,{SEQ}\n")
    info = assert_msa_matches(msa, SEQ)
    assert info["n_rows"] == 2 and info["query_len"] == len(SEQ)


def test_mismatched_msa_is_caught_with_the_first_difference(tmp_path):
    """THE failure this guard exists for: on any mismatch the model substitutes a dummy alignment and
    reports nothing, moving the affinity head into a regime it was never fitted for."""
    msa = tmp_path / "m.csv"
    msa.write_text(f"key,sequence\n101,{SEQ[:-1] + 'K'}\n")
    with pytest.raises(ValueError, match="silently.*DUMMY|DUMMY"):
        assert_msa_matches(msa, SEQ)


def test_an_a3m_masquerading_as_an_msa_is_rejected(tmp_path):
    """Passing an a3m raises a KeyError on a NUL byte deep inside the loader, which is a confusing
    way to learn the format is wrong."""
    a3m = tmp_path / "m.a3m"
    a3m.write_text(f">101\n{SEQ}\n>102\n{SEQ}\n")
    with pytest.raises(ValueError, match="key,sequence"):
        assert_msa_matches(a3m, SEQ)


def test_header_only_msa_is_rejected(tmp_path):
    msa = tmp_path / "m.csv"
    msa.write_text("key,sequence\n")
    with pytest.raises(ValueError, match="no alignment rows"):
        assert_msa_matches(msa, SEQ)
