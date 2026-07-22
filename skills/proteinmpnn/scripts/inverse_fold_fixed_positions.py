"""Redesign a binder while keeping interface / motif residues fixed.

`fixed_positions` are 1-indexed residues *within the binder chain* to hold
constant. Use this to preserve a validated interface or a functional motif while
diversifying the rest of the sequence.

Usage:
    PROTEINMPNN_PATH=/path/to/ProteinMPNN \
        python inverse_fold_fixed_positions.py <complex.pdb> <fixed e.g. 12,13,45> [num_seq]

Remember: listed positions are KEPT (not remodeled); everything else is redesigned.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from mpnn_runner import design_sequences


def main(pdb: str, fixed_csv: str, num_seq: int = 32) -> None:
    fixed = [int(x) for x in fixed_csv.split(',') if x.strip()]
    seqs = design_sequences(
        pdb_path=Path(pdb),
        proteinmpnn_path=Path(os.environ['PROTEINMPNN_PATH']),
        num_seq=num_seq,
        fixed_positions=fixed,          # KEEP these binder residues constant
        sampling_temp=0.1,
    )
    print(f'{len(seqs)} designs (kept binder positions {fixed}):')
    for s in seqs:
        print(s)


if __name__ == '__main__':
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    main(sys.argv[1], sys.argv[2], n)
