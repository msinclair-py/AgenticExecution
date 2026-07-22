"""Build an implicit-solvent (GB, mbondi3 radii) AMBER system from a PDB.

Pair the output with ImplicitSimulator. NOTE: implicit solvent cannot handle
glycans (no cpptraj sugar prep) — use explicit solvent for glycoproteins.

Usage:
    AMBERHOME=/path/to/amber python build_implicit.py <protein.pdb> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from molecular_simulations.build import ImplicitSolvent


def main(pdb: str, out_dir: str) -> None:
    ImplicitSolvent(path=Path(out_dir), pdb=Path(pdb), protein=True).build()
    print('Built implicit system in', out_dir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
