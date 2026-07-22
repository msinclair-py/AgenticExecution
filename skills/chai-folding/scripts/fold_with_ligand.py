"""Fold a protein with a bound ligand (SMILES), optionally placed at a pocket.

The ligand is a `>ligand|X` record whose sequence is a SMILES string. To steer
it into a known pocket, add a `pocket` constraint from the ligand chain to a
target pocket residue.

Usage:
    python fold_with_ligand.py "<PROTEIN>" "<SMILES>" <out_dir> [POCKET_RESIDUE]

POCKET_RESIDUE is a target residue like 'H41' (one-letter AA + index); when
given, the ligand is constrained within 5 A of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

from chai_common import build_fasta, fold
from write_constraints import write_constraints


def main(protein: str, smiles: str, out_dir: str,
         pocket_residue: str | None = None) -> None:
    out = Path(out_dir)
    records = [('protein', protein), ('ligand', smiles)]   # protein=A, ligand=B
    fasta = build_fasta(records, out / 'protein_ligand.fa')

    constraint_path = None
    if pocket_residue:
        constraints = {
            0: {'chainA': 'B', 'resA': '', 'chainB': 'A', 'resB': pocket_residue,
                'const_type': 'pocket', 'distance': 5.0},
        }
        constraint_path = write_constraints(constraints, out / 'constraints.txt')

    fold(fasta, out, constraint_path=constraint_path)
    print('Protein+ligand fold written to', out)


if __name__ == '__main__':
    pocket = sys.argv[4] if len(sys.argv) > 4 else None
    main(sys.argv[1], sys.argv[2], sys.argv[3], pocket)
