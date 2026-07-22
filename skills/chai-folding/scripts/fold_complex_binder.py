"""Co-fold a binder against a target (the core binder-design case).

Layout: target chain(s) first, binder LAST — so the binder is the final chain,
which downstream steps (ProteinMPNN binder identification, MM-PBSA fragment
split) rely on. The target may be a single sequence or a ':'-delimited
multi-chain string.

Usage:
    python fold_complex_binder.py "<TARGET[:TARGET2...]>" "<BINDER>" <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from chai_common import build_fasta, fold, split_target_chains


def main(target: str, binder: str, out_dir: str) -> None:
    out = Path(out_dir)
    records = [('protein', chain) for chain in split_target_chains(target)]
    records.append(('protein', binder))          # binder is always last
    fasta = build_fasta(records, out / 'complex.fa')
    fold(fasta, out)
    print('Co-fold written to', out, '(binder is chain',
          chr(ord('A') + len(records) - 1) + ')')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
