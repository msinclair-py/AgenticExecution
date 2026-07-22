"""Diffuse binder backbones docked against a target hotspot.

Usage:
    python design_binder_hotspot.py <target.pdb> <backbone_weights> <design_weights> \
        "<hotspot resids e.g. 47 48 49>" <binder_len> <out_dir> \
        [n_backbones=10] [steps=200]

Outputs backbone PDBs under <out_dir> — feed them to proteinmpnn for sequences.
"""
from __future__ import annotations

import sys

from chroma_designer import design


def main() -> None:
    pdb, bb_w, des_w, hotspot, blen, out = sys.argv[1:7]
    n_backbones = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    steps = int(sys.argv[8]) if len(sys.argv) > 8 else 200
    written = design(
        pdb=pdb, backbone_weights=bb_w, design_weights=des_w,
        hotspot_resids=hotspot, len_binder=int(blen), out_dir=out,
        steps=steps, n_backbones=n_backbones, n_designs=1,
    )
    print(f'Wrote {len(written)} backbones to {out}')
    for p in written:
        print(' ', p)


if __name__ == '__main__':
    main()
