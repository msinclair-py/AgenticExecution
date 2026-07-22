"""Restraint-guided folding: pocket/contact hotspots and disulfides.

Demonstrates composing multiple restraints in one constraint file for a
target+binder co-fold:
  - a pocket restraint steering the binder to an epitope residue, and
  - a disulfide inside the binder (SG-SG covalent link).

Usage:
    python fold_with_constraints.py "<TARGET>" "<BINDER>" <out_dir>

Edit `constraints` below to match your epitope residue and disulfide positions.
"""
from __future__ import annotations

import sys
from pathlib import Path

from chai_common import build_fasta, fold
from write_constraints import write_constraints


def main(target: str, binder: str, out_dir: str) -> None:
    out = Path(out_dir)
    records = [('protein', target), ('protein', binder)]   # A=target, B=binder
    fasta = build_fasta(records, out / 'complex.fa')

    constraints = {
        # Steer the binder (B) to bind within 5 A of target residue R47 (chain A).
        0: {'chainA': 'B', 'resA': '', 'chainB': 'A', 'resB': 'R47',
            'const_type': 'pocket', 'distance': 5.0},
        # Enforce a disulfide inside the binder between Cys10 and Cys24 (chain B).
        1: {'chainA': 'B', 'resA': 'C10@SG', 'chainB': 'B', 'resB': 'C24@SG',
            'const_type': 'covalent', 'distance': 2.05},
    }
    constraint_path = write_constraints(constraints, out / 'constraints.txt')

    fold(fasta, out, constraint_path=constraint_path)
    print('Constraint-guided co-fold written to', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
