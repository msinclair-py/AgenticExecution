"""Fold a glycoprotein: attach glycan chains via covalent linkages.

Each glycan is its own `>glycan|X` chain, placed after the protein chain(s) and
before the binder. A `covalent` constraint links the glycan root C1 to the
target acceptor atom (N-linked: Asn ND2; O-linked: Ser OG / Thr OG1).

This example shows a single-chain target with one N-linked glycan at N343.
Extend `glycan_sites` for more sites; each glycan takes the next free letter.

Usage:
    python fold_with_glycans.py "<PROTEIN>" "<GLYCAN_TREE>" <ACCEPTOR e.g. N343> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from chai_common import build_fasta, fold
from write_constraints import write_constraints

# Acceptor atom by linkage/residue letter.
_ACCEPTOR_ATOM = {'N': 'ND2', 'S': 'OG', 'T': 'OG1'}


def main(protein: str, glycan_tree: str, acceptor: str, out_dir: str) -> None:
    out = Path(out_dir)

    # Chain A = protein, chain B = glycan (next free letter after the target).
    records = [('protein', protein), ('glycan', glycan_tree)]
    fasta = build_fasta(records, out / 'glycoprotein.fa')

    aa = acceptor[0].upper()               # residue letter, e.g. 'N' in 'N343'
    atom = _ACCEPTOR_ATOM.get(aa)
    if atom is None:
        raise ValueError(f'{acceptor!r} is not a supported glycan acceptor (N/S/T)')

    constraints = {
        0: {'chainA': 'A', 'resA': f'{acceptor}@{atom}',   # target acceptor atom
            'chainB': 'B', 'resB': '@C1',                  # glycan root C1
            'const_type': 'covalent', 'distance': 1.5},
    }
    constraint_path = write_constraints(constraints, out / 'constraints.txt')

    fold(fasta, out, constraint_path=constraint_path)
    print(f'Glycoprotein fold ({acceptor}@{atom} <-> glycan C1) written to', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
