"""Receptor stage: fetch a PDB entry and prepare a docking-ready receptor, from config alone.

**Why this stage exists.** Ligand data was already a first-class pipeline input — pulled by
``data_pull``, driven by config, with provenance recorded. The receptor was not: it was a
hand-prepared PDBQT committed under ``assets/``, produced once by a human running tools by hand.
``structure.reference_pdb`` sat in every config and **nothing read it**.

That asymmetry is why a second target had no receptor. Retargeting was config-only for the ligand arm
and manual for the structural arm, which makes "point it at a new target" false for half the pipeline.

What this records, and why each piece is provenance rather than decoration:

* the **fetched PDB bytes and their sha256** — a receptor is an input like any other, and an input
  without a checksum cannot be shown to be the same one next time;
* the **entry title and resolution**, so a reader can see what structure a docking box came from
  without re-fetching it;
* the **ligand chosen to define the box, its residue name, and its atom count** — the box centre is
  the single most consequential number in a docking campaign, and "the centroid of the co-crystal
  ligand" is only reproducible if the ligand is named;
* every **external command run**, with its exit status. ``pdb2pqr`` and ``mk_prepare_receptor`` do the
  chemistry here; if either changes behaviour, the recorded command line is what explains a moved box.

Deliberately NOT done here: docking. This stage produces inputs. Docking engines are separate because
Uni-Dock/Vina-GPU need CUDA while this preparation is pure CPU and runs anywhere (ADR 0005).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from medchem.pipeline.stage import StageContext, StageResult, stage
from medchem.structure.box import validate_box
from medchem.structure.cofold import sequence_from_pdb, write_boltz_yaml

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Solvent, cryoprotectants and buffer components are never the ligand that defines a binding site.
_NOT_LIGANDS = {
    "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "MPD", "DMS", "ACT", "ACY", "FMT",
    "CL", "NA", "MG", "CA", "ZN", "K", "MN", "IOD", "BR", "NO3", "TRS", "EPE", "IMD",
}


def _fetch_pdb(pdb_id: str, dest: Path) -> dict[str, Any]:
    """Download one PDB entry from RCSB and record what arrived."""
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed public host)
        raw = resp.read()
    dest.write_bytes(raw)
    text = raw.decode("utf-8", errors="replace")
    title = " ".join(
        ln[10:].strip() for ln in text.splitlines() if ln.startswith("TITLE")
    ).strip()
    resolution = None
    for ln in text.splitlines():
        if ln.startswith("REMARK   2 RESOLUTION."):
            # Parse only what follows "RESOLUTION." -- scanning the whole line finds the "2" of
            # "REMARK   2" first and reports every structure as 2.0 A. That bug read 1.24 A as 2.0,
            # and went unnoticed on the previous entry because its true resolution WAS 2.0.
            tail = ln.split("RESOLUTION.", 1)[1]
            for tok in tail.replace(";", " ").split():
                try:
                    resolution = float(tok)
                    break
                except ValueError:
                    continue
            break
    return {
        "pdb_id": pdb_id.upper(),
        "url": url,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "title": title,
        "resolution_angstrom": resolution,
    }


def _het_residues(pdb_text: str) -> dict[tuple[str, str, str], list[tuple[float, float, float]]]:
    """HETATM coordinates grouped by (resname, chain, resseq), skipping solvent and ions."""
    out: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    for ln in pdb_text.splitlines():
        if not ln.startswith("HETATM"):
            continue
        resname = ln[17:20].strip()
        if resname in _NOT_LIGANDS:
            continue
        key = (resname, ln[21:22].strip(), ln[22:26].strip())
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        out.setdefault(key, []).append(xyz)
    return out


def _pick_box_ligand(
    pdb_text: str, wanted: str | None
) -> tuple[tuple[str, str, str], list[tuple[float, float, float]]]:
    """Choose the ligand whose centroid defines the docking box.

    When the config names one, use it and fail loudly if absent — a silently substituted ligand moves
    the box to a different site, which no downstream number would reveal. Otherwise take the largest
    non-solvent heteroresidue, which is the co-crystallised ligand in a liganded structure.
    """
    het = _het_residues(pdb_text)
    if not het:
        raise ValueError(
            "no non-solvent HETATM residues found: this entry appears to be apo, so there is no "
            "co-crystal ligand to centre a box on. Set structure.box_center explicitly, or choose a "
            "liganded entry."
        )
    if wanted:
        matches = {k: v for k, v in het.items() if k[0] == wanted.upper()}
        if not matches:
            raise ValueError(
                f"structure.reference_ligand={wanted!r} is not in this entry. Present: "
                f"{sorted({k[0] for k in het})}"
            )
        return max(matches.items(), key=lambda kv: len(kv[1]))
    return max(het.items(), key=lambda kv: len(kv[1]))


def _centroid(xyz: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    n = len(xyz)
    return (
        round(sum(p[0] for p in xyz) / n, 3),
        round(sum(p[1] for p in xyz) / n, 3),
        round(sum(p[2] for p in xyz) / n, 3),
    )


def _protein_only(
    pdb_text: str,
    *,
    keep_waters_within: float | None = None,
    ligand_xyz: list[tuple[float, float, float]] | None = None,
) -> tuple[str, int]:
    """Protein ATOM records, optionally retaining nearby crystallographic waters.

    Returns ``(pdb_text, n_waters_kept)``. Waters are selected by distance to the ligand that DEFINES
    the box rather than to the box centre: the ligand traces the pocket surface, and the network that
    matters is the shell in contact with it.

    Dropping every water is the right default for an ATP site and wrong for a bromodomain, where
    ordered waters mediate acetyl-lysine recognition. That is a target-class decision, so it is
    configured, not assumed.
    """
    keep = [ln for ln in pdb_text.splitlines() if ln.startswith(("ATOM", "TER"))]
    n_waters = 0
    if keep_waters_within and ligand_xyz:
        cutoff_sq = float(keep_waters_within) ** 2
        for ln in pdb_text.splitlines():
            if not ln.startswith("HETATM") or ln[17:20].strip() != "HOH":
                continue
            try:
                w = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            if any(sum((w[i] - L[i]) ** 2 for i in range(3)) <= cutoff_sq for L in ligand_xyz):
                keep.append(ln)
                n_waters += 1
    return "\n".join(keep) + "\nEND\n", n_waters


def _run(cmd: list[str], cwd: Path, timeout: int = 1800) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-800:],
        "stderr_tail": proc.stderr[-800:],
    }


# config_keys includes "vls": the co-folding panel is drawn from vls.known_reference, so a
# change there must invalidate this stage. Declaring only "structure" would let an edited
# reference set serve stale inputs from cache.
@stage("discovery", "receptor", config_keys=("structure", "vls"))
def receptor(ctx: StageContext) -> StageResult:
    """Fetch ``structure.reference_pdb`` and prepare a docking-ready receptor + box."""
    out = Path(ctx.workdir)
    st = ctx.config.structure
    pdb_id = (st.reference_pdb or "").strip()
    if not pdb_id:
        metrics = {
            "status": "skipped",
            "reason": "structure.reference_pdb is empty",
            "hint": "set structure.reference_pdb to a PDB entry (e.g. 3MXF) to prepare a receptor",
        }
        (out / "receptor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return StageResult(
            name="receptor",
            outputs={"metrics": str(out / "receptor_metrics.json")},
            metrics=metrics,
        )

    raw_pdb = out / f"{pdb_id.upper()}.pdb"
    provenance = _fetch_pdb(pdb_id, raw_pdb)
    text = raw_pdb.read_text(encoding="utf-8", errors="replace")

    # An explicit centre is the escape hatch for an apo structure. Honour it BEFORE looking for a
    # ligand, so an apo entry with a configured centre does not fail on the missing ligand.
    if st.box_center is not None:
        center = (float(st.box_center[0]), float(st.box_center[1]), float(st.box_center[2]))
        try:
            (resname, chain, resseq), lig_xyz = _pick_box_ligand(text, st.reference_ligand)
        except ValueError:
            resname, chain, resseq, lig_xyz = "", "", "", []
        center_source = "structure.box_center (explicit)"
    else:
        (resname, chain, resseq), lig_xyz = _pick_box_ligand(text, st.reference_ligand)
        center = _centroid(lig_xyz)
        center_source = f"centroid of {resname} {chain}{resseq}"
    size = float(st.box_size)

    protein_pdb = out / "protein.pdb"
    protein_text, n_waters = _protein_only(
        text, keep_waters_within=st.keep_waters_within, ligand_xyz=lig_xyz or None
    )
    protein_pdb.write_text(protein_text, encoding="utf-8")

    steps: list[dict[str, Any]] = []
    # Protonate at physiological pH. Docking scores depend on which groups carry hydrogens, so this is
    # chemistry, not a format conversion, and the pH used belongs in provenance.
    pqr = out / "protein.pqr"
    # Basenames, not absolute paths: these tools write side files (logs, JSON) relative to cwd, and
    # an absolute path combined with cwd=out makes pdb2pqr try to open <out>/<out>/protein.log.
    steps.append(_run(
        ["pdb2pqr", f"--with-ph={st.ph}", "--ff=AMBER",
         *([] if n_waters else ["--drop-water"]),
         protein_pdb.name, pqr.name],
        cwd=out,
    ))
    receptor_pdbqt = out / "receptor.pdbqt"
    if pqr.exists():
        steps.append(_run(
            ["mk_prepare_receptor.py", "--read_pqr", pqr.name, "-o", "receptor",
             "-p", "--box_size", str(size), str(size), str(size),
             "--box_center", str(center[0]), str(center[1]), str(center[2])],
            cwd=out,
        ))

    # ---- co-folding inputs, from the SAME fetched structure -------------------------------------
    # The flagship target's co-folding sequence was a committed FASTA with a hand-annotated header.
    # Deriving it here removes the same asymmetry the docking receptor had: one input reproducible
    # from config, the other a file somebody made once.
    sequence, seq_source = sequence_from_pdb(text)
    cofold_dir = out / "cofold_inputs"
    cofold: list[dict[str, Any]] = []
    # Default panel: this target's own reference compounds. They are the positive control -- if a
    # co-folding run cannot rank known binders, its ranking of novel compounds is not evidence.
    for name, smi in (ctx.config.vls.known_reference or {}).items():
        rec = write_boltz_yaml(
            cofold_dir / f"{pdb_id.upper()}_{name}.yaml",
            sequence=sequence, smiles=smi, msa_abs=None,
        )
        cofold.append({"name": name, "stem": rec.stem, "smiles": smi, "warnings": rec.warnings})

    box = out / "receptor.box.txt"
    box.write_text(
        f"center_x = {center[0]:.3f}\ncenter_y = {center[1]:.3f}\ncenter_z = {center[2]:.3f}\n"
        f"size_x = {size:.3f}\nsize_y = {size:.3f}\nsize_z = {size:.3f}\n",
        encoding="utf-8",
    )

    # Validate the box against the structure it came from. This used to be a hand-run script on one
    # target; a box is the most consequential number in a campaign and the easiest to get silently
    # wrong, so it is checked here for every target, every run.
    # `site_residue`, not `anchor_residue or hinge_residue`. The two are one slot in two vocabularies,
    # and `or` silently dropped the hinge when a config supplied both. medchem.config now rejects that
    # config outright and resolves the survivor explicitly.
    site = st.site_residue
    report = validate_box(
        text, center=center, size=(size, size, size),
        box_ligand_resname=resname, site_residue=site,
        max_protein_fraction=st.max_protein_fraction_in_box,
    )

    prepared = receptor_pdbqt.exists() and receptor_pdbqt.stat().st_size > 0
    metrics = {
        "status": ("prepared" if prepared else "box_only") if report.passed else "box_invalid",
        "entry": provenance,
        "box_ligand": {
            "resname": resname, "chain": chain, "resseq": resseq, "n_atoms": len(lig_xyz),
            "selected_by": "config" if st.reference_ligand else "largest non-solvent heteroresidue",
        },
        "box": {"center": list(center), "size": [size, size, size]},
        "box_validation": report.as_dict(),
        "cofold": {
            "sequence_length": len(sequence),
            "sequence_source": seq_source,
            "inputs_written": len(cofold),
            "records": cofold,
            "msa": "ABSENT -- single-sequence mode. Supply an MSA (key,sequence CSV) and re-write "
                   "these inputs before trusting affinity values; a MISMATCHED MSA is silently "
                   "replaced with a dummy, which is why assert_msa_matches exists.",
        },
        "ph": st.ph,
        "box_center_source": center_source,
        "waters": {
            "kept": n_waters,
            "cutoff_angstrom": st.keep_waters_within,
            "note": (
                "waters within the cutoff of the box-defining ligand are retained; None drops all, "
                "which is wrong for a bromodomain where ordered waters mediate recognition"
            ),
        },
        "commands": steps,
        "note": (
            "Inputs for docking, not a docking run. If status is 'box_only' the receptor PDBQT was "
            "not produced and the command log above says why -- the box is still valid and recorded."
        ),
    }
    (out / "receptor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    outputs = {
        "metrics": str(out / "receptor_metrics.json"),
        "pdb": str(raw_pdb),
        "protein_pdb": str(protein_pdb),
        "box": str(box),
    }
    if cofold:
        # A MANIFEST, not the directory: every output is content-hashed for the cache, and a directory
        # has no hash. The manifest also records which stems exist, which is what a consumer needs.
        manifest = cofold_dir / "cofold_manifest.json"
        manifest.write_text(json.dumps({
            "sequence": sequence,
            "sequence_source": seq_source,
            "msa": None,
            "records": [{**c, "yaml": f"{c['stem']}.yaml"} for c in cofold],
        }, indent=2), encoding="utf-8")
        outputs["cofold_manifest"] = str(manifest)
    if prepared:
        outputs["receptor_pdbqt"] = str(receptor_pdbqt)
    failed_steps = [st_ for st_ in steps if st_.get("returncode") != 0]
    if failed_steps and not st.allow_box_only:
        raise ValueError(
            "receptor preparation failed and structure.allow_box_only is false, so this is an error "
            "rather than a partial result:\n  - "
            + "\n  - ".join(
                f"{f['cmd'].split()[0]} exited {f['returncode']}: {f['stderr_tail'][-200:].strip()}"
                for f in failed_steps
            )
        )
    if not prepared and not st.allow_box_only:
        raise ValueError(
            "no receptor PDBQT was produced. Every external command reported success, so the failure "
            "is upstream of them -- inspect receptor_metrics.json. Set structure.allow_box_only to "
            "accept a box without a receptor deliberately."
        )
    if not report.passed:
        # Refuse rather than hand a wrong box downstream. Everything computed is already recorded, so
        # the failure is diagnosable without re-running.
        raise ValueError(
            "docking box failed validation against its own structure:\n  - "
            + "\n  - ".join(report.reasons)
            + f"\n(details recorded in {(out / 'receptor_metrics.json').name})"
        )
    return StageResult(name="receptor", outputs=outputs, metrics=metrics)
