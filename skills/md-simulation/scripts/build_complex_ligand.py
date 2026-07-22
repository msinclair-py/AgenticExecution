"""Build a protein-ligand complex: GAFF2 + AM1-BCC charges (antechamber/SQM).

Ligand residues are named LG0, LG1, ... in the topology. Requires the [ligand]
extra (RDKit + OpenBabel). Reuse precomputed params with lig_param_prefix.

Usage:
    AMBERHOME=/path/to/amber \
        python build_complex_ligand.py <protein.pdb> <ligand.sdf> <out_dir> [padding]
"""
from __future__ import annotations

import sys
from pathlib import Path

from molecular_simulations.build import ComplexBuilder


def main(pdb: str, lig: str, out_dir: str, padding: float = 12.0) -> None:
    ComplexBuilder(
        path=Path(out_dir),
        pdb=Path(pdb),
        lig=Path(lig),                     # or a list of SDFs
        padding=padding,
        # lig_param_prefix='/path/params/ligand',  # reuse frcmod/lib/mol2
    ).build()
    print('Built protein-ligand system in', out_dir)


if __name__ == '__main__':
    pad = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
    main(sys.argv[1], sys.argv[2], sys.argv[3], pad)
