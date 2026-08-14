"""Co-folding input preparation — target-agnostic, with the silent failures guarded at write time.

Co-folding is the tier where wrong inputs cost the most, because the model does not complain. Three
failure modes are documented in docs/PITFALLS.md and every one of them produces a run that looks
healthy:

1. **A mismatched MSA is silently replaced with a dummy.** Any difference between the MSA's sequence
   and the input sequence — one residue, a trailing newline — and the model proceeds with a
   single-sequence stand-in. Nothing in the output says so, and the affinity head was trained with
   real alignments.
2. **Malformed YAML is skipped per-file.** ``msa:`` at eight spaces instead of six becomes a
   continuation of the sequence scalar; the loader raises, the framework catches it, prints
   ``Failed to process ... Skipping``, and then runs the trainer over **zero records** while looking
   busy. That cost one GPU run at 0/4 records.
3. **A ``#`` in an unquoted SMILES starts a YAML comment**, truncating every nitrile.

The guards here are cheap and they run on CPU at write time, which is the entire point: each one turns
a wasted GPU reservation into an immediate local error. The previous implementations of these checks
lived in the out-of-package execution layer, hardcoded to one target; this version takes a sequence, a
ligand and an optional MSA, so it applies to any target.

The protein sequence is DERIVED from the fetched structure rather than supplied as a hand-written
FASTA — the flagship target's FASTA was a committed file with a hand-annotated header, which is exactly
the asymmetry the ``receptor`` stage removed for docking.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common modified residues that appear in crystal structures.
    "MSE": "M", "SEC": "U", "PYL": "O", "HYP": "P", "CSO": "C", "PTR": "Y", "SEP": "S", "TPO": "T",
}


@dataclass
class CofoldInput:
    """One prepared co-folding record."""

    stem: str
    path: Path
    smiles: str
    msa: str | None = None
    warnings: list[str] = field(default_factory=list)


def sequence_from_pdb(pdb_text: str, chain: str | None = None) -> tuple[str, str]:
    """Derive a one-letter sequence from a PDB entry. Returns ``(sequence, source)``.

    SEQRES is preferred because it is the construct that was crystallised, including residues that
    were disordered and therefore absent from ATOM records. Co-folding predicts a structure rather
    than reproducing this one, so the full construct is the honest input; falling back to observed
    residues silently shortens the chain across every disordered loop.
    """
    seqres: dict[str, list[str]] = {}
    for ln in pdb_text.splitlines():
        if ln.startswith("SEQRES"):
            ch = ln[11:12].strip() or "A"
            seqres.setdefault(ch, []).extend(ln[19:].split())
    if seqres:
        ch = chain or next(iter(seqres))
        if ch not in seqres:
            raise ValueError(f"chain {ch!r} absent from SEQRES; present: {sorted(seqres)}")
        residues = seqres[ch]
        unknown = {r for r in residues if r not in _THREE_TO_ONE}
        seq = "".join(_THREE_TO_ONE.get(r, "X") for r in residues)
        src = f"SEQRES chain {ch}"
        if unknown:
            src += f" ({len(unknown)} non-standard residue type(s) as X: {sorted(unknown)[:5]})"
        return seq, src

    # No SEQRES: fall back to observed CA atoms, and say so, because the two differ wherever the
    # crystal has disorder.
    seen: list[tuple[int, str]] = []
    last = None
    for ln in pdb_text.splitlines():
        if not ln.startswith("ATOM") or ln[12:16].strip() != "CA":
            continue
        ch = ln[21:22].strip() or "A"
        if chain and ch != chain:
            continue
        num = ln[22:27].strip()
        if num == last:
            continue
        last = num
        seen.append((len(seen), ln[17:20].strip()))
    if not seen:
        raise ValueError("no SEQRES records and no CA atoms: cannot derive a sequence")
    return "".join(_THREE_TO_ONE.get(r, "X") for _i, r in seen), "observed CA atoms (no SEQRES)"


def assert_msa_matches(msa_csv: str | Path, sequence: str) -> dict:
    """Verify an MSA's query sequence is EXACTLY the input sequence.

    This is the guard for the worst failure mode: on any mismatch the model substitutes a dummy
    alignment and reports nothing. The affinity head was trained with real alignments, so a dummy
    quietly moves the prediction into a regime it was never fitted for.

    The file must be the ``key,sequence`` CSV, not an a3m — parsing an a3m as a CSV raises a
    ``KeyError`` on a NUL byte, which is a confusing way to learn the format is wrong.
    """
    path = Path(msa_csv)
    if not path.is_file():
        raise FileNotFoundError(f"MSA not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"MSA {path} is empty")
    header = [c.strip().lower() for c in rows[0]]
    if header[:2] != ["key", "sequence"]:
        raise ValueError(
            f"MSA {path} does not start with a 'key,sequence' header (got {rows[0][:2]}). An a3m "
            f"file will not work here: it must be the CSV produced by the MSA step."
        )
    if len(rows) < 2:
        raise ValueError(f"MSA {path} has a header but no alignment rows")
    query = rows[1][1].strip()
    if query != sequence:
        raise ValueError(
            "MSA query sequence does not match the input sequence, so the model would silently "
            "substitute a DUMMY alignment.\n"
            f"  msa    len={len(query)} {query[:40]}...\n"
            f"  input  len={len(sequence)} {sequence[:40]}...\n"
            f"  first difference at index {_first_diff(query, sequence)}"
        )
    return {"path": str(path), "n_rows": len(rows) - 1, "query_len": len(query)}


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def write_boltz_yaml(
    path: str | Path,
    *,
    sequence: str,
    smiles: str,
    msa_abs: str | None = None,
    protein_id: str = "A",
    ligand_id: str = "L",
    affinity_binder: bool = True,
) -> CofoldInput:
    """Write one co-folding YAML and PARSE IT BACK before returning.

    One file per ligand: there is no multi-ligand input format, and the file stem becomes the record
    id that names every output, so stems must be unique.

    The parse-back is the guard that matters. Indentation errors here are caught per-file by the
    framework and reported as a skipped input, after which it runs over zero records while appearing
    to work. Validating on CPU at write time costs microseconds; discovering it on a GPU costs a run.
    """
    p = Path(path)
    if not sequence or not sequence.strip():
        raise ValueError("empty protein sequence")
    if not smiles or not smiles.strip():
        raise ValueError("empty SMILES")
    if msa_abs is not None and not Path(msa_abs).is_absolute():
        raise ValueError(
            f"msa path must be ABSOLUTE ({msa_abs!r} is not): a relative path is resolved against "
            f"the process working directory, not the input file"
        )

    warnings: list[str] = []
    if msa_abs is None:
        warnings.append(
            "no MSA supplied: this runs in single-sequence mode, which is a different and weaker "
            "regime than the aligned inputs the affinity head was trained on"
        )

    # `msa:` is a sibling of `id:`/`sequence:` at SIX spaces. At eight it becomes a continuation of
    # the sequence scalar and the file is skipped. SMILES is single-quoted so '#' cannot start a
    # comment and truncate a nitrile.
    msa_line = f"\n      msa: {msa_abs}" if msa_abs else ""
    text = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        f"      id: {protein_id}\n"
        f"      sequence: {sequence}{msa_line}\n"
        "  - ligand:\n"
        f"      id: {ligand_id}\n"
        f"      smiles: '{smiles}'\n"
    )
    if affinity_binder:
        text += "properties:\n  - affinity:\n" f"      binder: {ligand_id}\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

    # ---- parse back: every field must survive a real YAML load ----------------------------------
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    try:
        prot = next(e["protein"] for e in doc["sequences"] if "protein" in e)
        lig = next(e["ligand"] for e in doc["sequences"] if "ligand" in e)
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError(f"{p} did not parse back into protein+ligand entries: {exc}") from exc
    if prot["sequence"] != sequence:
        raise ValueError(f"{p}: sequence did not survive the round trip")
    if lig["smiles"] != smiles:
        raise ValueError(
            f"{p}: SMILES did not survive the round trip -- wrote {smiles!r}, read "
            f"{lig['smiles']!r} (an unquoted '#' truncates at a comment)"
        )
    if msa_abs is not None and prot.get("msa") != msa_abs:
        raise ValueError(
            f"{p}: msa key missing or wrong after parse-back (got {prot.get('msa')!r}). This is the "
            f"eight-space indentation bug: the value folded into the sequence scalar."
        )
    if affinity_binder:
        binder = doc.get("properties", [{}])[0].get("affinity", {}).get("binder")
        if binder != ligand_id:
            raise ValueError(f"{p}: affinity binder is {binder!r}, expected {ligand_id!r}")
    return CofoldInput(stem=p.stem, path=p, smiles=smiles, msa=msa_abs, warnings=warnings)
