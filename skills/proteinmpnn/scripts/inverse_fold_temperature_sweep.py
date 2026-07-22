"""Sweep sampling temperature x model variant to trade conservatism vs diversity.

Low temperature + low-noise model -> conservative, native-like.
Higher temperature + higher-noise model -> more diverse (useful to unstick a
stalled campaign), at the cost of exploring less favourable sequence space.

Usage:
    PROTEINMPNN_PATH=/path/to/ProteinMPNN \
        python inverse_fold_temperature_sweep.py <complex.pdb> [per_cell]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from mpnn_runner import design_sequences

TEMPS = [0.1, 0.2, 0.3]
MODELS = ['v_48_020', 'v_48_030']   # default vs high-noise (for predicted backbones)


def main(pdb: str, per_cell: int = 8) -> None:
    ppath = Path(os.environ['PROTEINMPNN_PATH'])
    for model in MODELS:
        for temp in TEMPS:
            seqs = design_sequences(
                pdb_path=Path(pdb), proteinmpnn_path=ppath,
                num_seq=per_cell, batch_size=per_cell,
                sampling_temp=temp, model_name=model,
            )
            print(f'\n=== {model}  T={temp}  ({len(seqs)} unique) ===')
            for s in seqs:
                print(s)


if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(sys.argv[1], n)
