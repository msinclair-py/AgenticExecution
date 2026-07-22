"""Build an explicit-solvent (OPC, ~150 mM NaCl) AMBER system from a PDB.

Writes system.prmtop / system.inpcrd / system.pdb into <out_dir>.

Usage:
    AMBERHOME=/path/to/amber python build_explicit.py <protein.pdb> <out_dir> [padding]
"""
from __future__ import annotations

import sys
from pathlib import Path

from molecular_simulations.build import ExplicitSolvent, convert_cif_with_gemmi


def main(pdb: str, out_dir: str, padding: float = 10.0) -> None:
    pdb_path = Path(pdb)
    if pdb_path.suffix == '.cif':                # Chai emits CIF; builders want PDB
        pdb_path = Path(convert_cif_with_gemmi(pdb_path))

    ExplicitSolvent(
        path=Path(out_dir),
        pdb=pdb_path,
        padding=padding,
        # disulfide_residues=[23, 88],          # force CYX + let tleap bond
    ).build()
    print('Built explicit system in', out_dir)


if __name__ == '__main__':
    pad = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    main(sys.argv[1], sys.argv[2], pad)
