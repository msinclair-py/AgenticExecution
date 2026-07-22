"""Multi-chain target, binder-only design.

No special handling is needed: the binder is the last chain, and every target
chain (however many) is held fixed automatically. This is ProteinMPNN's native
multi-chain case. This script is identical in call to inverse_fold_single but is
kept separate to document the multi-chain contract explicitly.

Usage:
    PROTEINMPNN_PATH=/path/to/ProteinMPNN \
        python inverse_fold_multichain.py <multichain_complex.pdb> [num_seq]

The PDB should contain the target chains (A, B, ...) then the binder as the
final chain. Only the binder is redesigned; all target chains stay fixed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from mpnn_runner import _binder_chain, _clean_pdb, design_sequences


def main(pdb: str, num_seq: int = 32) -> None:
    cleaned = _clean_pdb(Path(pdb), Path(pdb).parent / 'clean')
    print('Binder (designed) chain =', _binder_chain(cleaned),
          '— all other chains held fixed')
    seqs = design_sequences(
        pdb_path=Path(pdb),
        proteinmpnn_path=Path(os.environ['PROTEINMPNN_PATH']),
        num_seq=num_seq,
    )
    for s in seqs:
        print(s)


if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    main(sys.argv[1], n)
