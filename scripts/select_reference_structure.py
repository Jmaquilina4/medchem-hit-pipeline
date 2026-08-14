"""Choose a reference crystal structure by EVIDENCE, for any target.

The config used to name a PDB entry from memory. That produced a defensible-looking but weak choice:
BRD4's entry was 3MXF at 2.00 A holding JQ1 -- a TOOL compound -- when 1.24 A structures with an actual
phase-3 drug bound were available.

Two things make this reliable rather than a search-and-eyeball:

* **Match ligands by STRUCTURE, not by name.** PDB chemical-component names are systematic
  ("2-[4-(2-hydroxyethoxy)-3,5-dimethylphenyl]..."), not commercial, so searching for "apabetalone"
  finds nothing. Each entry's bound ligand is fingerprinted and compared by Tanimoto against the
  target's own clinical compounds from `vls.known_reference`.
* **Check the domain.** BRD4 has two bromodomains and apabetalone is BD2-SELECTIVE in function, so a
  BD2 co-crystal would centre a docking box on a different pocket from the one a BD1 chemotype targets.
  Residue ranges are reported so that cannot pass unnoticed.

Usage:
    uv run python scripts/select_reference_structure.py --config configs/brd4.yaml \
        --uniprot O60885 --max-resolution 1.6
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GQL = "https://data.rcsb.org/graphql"


def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed public hosts)
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="target config supplying vls.known_reference")
    ap.add_argument("--uniprot", required=True, help="UniProt accession for the target protein")
    ap.add_argument("--max-resolution", type=float, default=1.6)
    ap.add_argument("--min-tanimoto", type=float, default=0.55)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    from medchem.config import load_config

    RDLogger.DisableLog("rdApp.*")
    cfg = load_config(args.config)
    refs = cfg.vls.known_reference
    if not refs:
        print("  config defines no vls.known_reference compounds to match against")
        return 1

    def fp(smi: str):
        m = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

    ref_fps = {k: fp(v) for k, v in refs.items()}
    ref_fps = {k: v for k, v in ref_fps.items() if v is not None}

    ids = [h["identifier"] for h in _post(SEARCH, {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers."
                             "reference_sequence_identifiers.database_accession",
                "operator": "exact_match", "value": args.uniprot}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": args.max_resolution}},
        ]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 250},
                            "results_content_type": ["experimental"]},
    })["result_set"]]
    print(f"  {len(ids)} entries for {args.uniprot} at <= {args.max_resolution} A")

    gql = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id
      rcsb_entry_info{resolution_combined}
      nonpolymer_entities{nonpolymer_comp{chem_comp{id name}
        rcsb_chem_comp_descriptor{SMILES}}}}}"""
    hits = []
    for i in range(0, len(ids), 50):
        for e in _post(GQL, {"query": gql, "variables": {"ids": ids[i:i + 50]}})["data"]["entries"]:
            res = (e["rcsb_entry_info"]["resolution_combined"] or [None])[0]
            for ne in (e.get("nonpolymer_entities") or []):
                comp = ne["nonpolymer_comp"]
                smi = (comp.get("rcsb_chem_comp_descriptor") or {}).get("SMILES")
                f = fp(smi) if smi else None
                if f is None:
                    continue
                for name, rf in ref_fps.items():
                    t = DataStructs.TanimotoSimilarity(f, rf)
                    if t >= args.min_tanimoto:
                        hits.append({"resolution": res, "pdb_id": e["rcsb_id"],
                                     "ligand_id": comp["chem_comp"]["id"],
                                     "ligand_name": comp["chem_comp"]["name"],
                                     "matches": name, "tanimoto": round(t, 3)})
    # An EXACT ligand match beats a better resolution holding an analogue: the point is a clinically
    # relevant pose, and an analogue's pose is a different compound's pose.
    hits.sort(key=lambda h: (-(h["tanimoto"] >= 0.999), h["resolution"] or 99, -h["tanimoto"]))
    print(f"  {len(hits)} entries hold a ligand matching a reference compound\n")
    print(f"  {'res':>5}  {'PDB':<6}{'lig':<5}{'matches':<14}{'T':<7}domain-range")
    for h in hits[:args.top]:
        rng = ""
        try:
            with urllib.request.urlopen(  # noqa: S310
                f"https://files.rcsb.org/download/{h['pdb_id']}.pdb", timeout=90) as r:
                txt = r.read().decode("utf-8", errors="replace")
            nums = [int(x[22:26]) for x in txt.splitlines()
                    if x.startswith("ATOM") and x[22:26].strip().lstrip("-").isdigit()]
            rng = f"{min(nums)}-{max(nums)}" if nums else "?"
        except Exception:
            rng = "fetch failed"
        h["residue_range"] = rng
        print(f"  {h['resolution'] or 0:>5.2f}  {h['pdb_id']:<6}{h['ligand_id']:<5}"
              f"{h['matches']:<14}{h['tanimoto']:<7}{rng}")

    if hits:
        best = hits[0]
        print(f"\n  RECOMMENDED: reference_pdb: {best['pdb_id']}  reference_ligand: {best['ligand_id']}")
        print(f"    {best['resolution']} A, matches {best['matches']} at Tanimoto {best['tanimoto']}")
        print("    Check the residue range against the domain your chemotype targets before adopting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
