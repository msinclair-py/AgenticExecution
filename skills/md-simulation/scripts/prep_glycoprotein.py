"""Prepare a Chai glycoprotein fold for explicit-solvent MD.

Two things must happen before building a glycoprotein:
  1. Move the binder chain to LAST (so the MM-PBSA fragment split is clean).
  2. Build with ExplicitSolvent(glycans=True), KEEPING Chai's PDB sugar names —
     the builder's cpptraj prepareforleap renames to GLYCAM and forms the
     glycosidic/disulfide bonds itself. Pre-renaming would hide the sugars.

Implicit solvent is rejected: it has no cpptraj sugar prep, so glycosidic bonds
never form. Use explicit only.

Usage:
    AMBERHOME=/path/to/amber \
        python prep_glycoprotein.py <fold.pdb> <binder_sequence> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import MDAnalysis as mda

from molecular_simulations.build import ExplicitSolvent

# 3-letter -> 1-letter, incl. AMBER/GLYCAM variants so a content match survives
# renaming (NLN=glycosylated Asn, CYX=disulfide Cys, His protonation states).
_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'NLN': 'N', 'CYX': 'C', 'HID': 'H', 'HIE': 'H',
    'HIP': 'H',
}
SUGARS = {'NAG', 'FUC', 'MAN', 'BMA', 'GAL', 'SIA', 'GLC', 'XYS'}


def _has_sugars(pdb: Path) -> bool:
    with open(pdb) as fh:
        return any(
            line[:6].strip() in ('ATOM', 'HETATM') and line[17:20].strip() in SUGARS
            for line in fh
        )


def _ordered_chains(atoms) -> list[str]:
    seen: list[str] = []
    for c in atoms.chainIDs:
        if c not in seen:
            seen.append(c)
    return seen


def reorder_binder_last(pdb: Path, binder_seq: str, out_dir: Path) -> Path:
    """Rewrite the PDB with the binder chain after all target/glycan chains."""
    u = mda.Universe(str(pdb))
    target = binder_seq.strip()
    binder_chain = None
    for chain in _ordered_chains(u.atoms):
        residues = u.select_atoms(f'chainID {chain}').residues
        seq = ''.join(_THREE_TO_ONE[r.resname] for r in residues
                      if r.resname in _THREE_TO_ONE)
        if seq == target:
            binder_chain = chain            # last match wins (binder folded last)
    if binder_chain is None:
        raise ValueError('No chain matches the binder sequence; cannot place it last.')

    others = [c for c in _ordered_chains(u.atoms) if c != binder_chain]
    rest = u.select_atoms(f'chainID {" ".join(others)}') if others else u.atoms[[]]
    ordered = rest + u.select_atoms(f'chainID {binder_chain}')

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{pdb.stem}_binderlast.pdb'
    ordered.write(str(out))
    return out


def main(pdb: str, binder_seq: str, out_dir: str) -> None:
    pdb, out_dir = Path(pdb), Path(out_dir)
    if not _has_sugars(pdb):
        print('No sugars found — build normally with build_explicit.py.')
        return
    build_pdb = reorder_binder_last(pdb, binder_seq, out_dir)
    print('Binder moved last ->', build_pdb)
    ExplicitSolvent(path=out_dir, pdb=build_pdb, glycans=True).build()  # keep PDB sugar names
    print('Built glycoprotein system in', out_dir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
