"""Fold a multi-chain (hetero- or homo-multimeric) target complex.

Each target chain becomes its own record/chain so a homodimer folds as two
chains instead of collapsing into one. Pass the chains as a ':'-delimited
string. Optionally append a binder as a final chain.

Usage:
    python fold_multichain_target.py "<CHAIN_A>:<CHAIN_B>[:...]" <out_dir> [BINDER]
"""
from __future__ import annotations

import sys
from pathlib import Path

from chai_common import build_fasta, fold, split_target_chains


def main(target: str, out_dir: str, binder: str | None = None) -> None:
    out = Path(out_dir)
    records = [('protein', chain) for chain in split_target_chains(target)]
    if binder:
        records.append(('protein', binder))
    fasta = build_fasta(records, out / 'multichain.fa')
    fold(fasta, out)
    print(f'Folded {len(records)} chains into', out)


if __name__ == '__main__':
    binder = sys.argv[3] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], binder)
