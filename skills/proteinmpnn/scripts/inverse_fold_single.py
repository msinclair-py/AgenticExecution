"""Redesign the binder chain of a folded complex (in-process, no fixed positions).

Usage:
    PROTEINMPNN_PATH=/path/to/ProteinMPNN \
        python inverse_fold_single.py <complex.pdb> [num_seq]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from mpnn_runner import design_sequences


def main(pdb: str, num_seq: int = 32) -> None:
    seqs = design_sequences(
        pdb_path=Path(pdb),
        proteinmpnn_path=Path(os.environ['PROTEINMPNN_PATH']),
        num_seq=num_seq,
        sampling_temp=0.1,
        model_name='v_48_020',
        model_weights='soluble_model_weights',
    )
    print(f'{len(seqs)} unique designs:')
    for s in seqs:
        print(s)


if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    main(sys.argv[1], n)
