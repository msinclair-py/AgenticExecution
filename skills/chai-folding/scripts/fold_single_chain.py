"""Fold a single-chain protein (monomer) with Chai-1.

Usage:
    python fold_single_chain.py <SEQUENCE> <out_dir>

The one record lands on chain A. This is the simplest case — no complex, no
constraints. Score the result with score_ipsae.py (monomer scoring returns
pLDDT-based confidence; ipSAE interface metrics apply only to complexes).
"""
from __future__ import annotations

import sys
from pathlib import Path

from chai_common import build_fasta, fold


def main(sequence: str, out_dir: str) -> None:
    out = Path(out_dir)
    fasta = build_fasta([('protein', sequence)], out / 'monomer.fa')
    fold(fasta, out)
    print('Models written to', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
