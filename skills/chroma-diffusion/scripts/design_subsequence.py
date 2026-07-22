"""Graft a fixed subsequence/motif into the diffused binder (SubsequenceConditioner).

Use when the binder must contain a known functional motif (e.g. a peptide epitope
or a binding loop): the motif residues are fixed in the binder's sequence tensor
and a SubsequenceConditioner biases the design toward realising them, while the
rest of the binder is diffused freely.

This shows the essential additions over design_binder_hotspot; wire it into the
chroma_designer prepare/conditioner steps for a full campaign.

Usage:
    python design_subsequence.py <target.pdb> <backbone_weights> <design_weights> \
        "<hotspot resids>" <binder_len> "<MOTIF>" <motif_start> <out_dir>

motif_start is the 0-indexed offset of the motif within the binder segment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from chroma import Protein, conditioners, constants

from chroma_designer import hotspot_vector, load_model, sample, save_pdbs
from _device import resolve_device


def prepare_with_motif(pdb: Path, len_binder: int, motif: str, motif_start: int,
                       device: str):
    """Extend the target with a binder segment whose motif residues are fixed."""
    protein = Protein(str(pdb), device=device)
    X, C, S = protein.to_XCS()

    # Build the binder sequence tensor with the motif placed at motif_start.
    alphabet = constants.AA20
    binder_S = torch.zeros(1, len_binder, dtype=S.dtype)
    binder_mask = [0] * len_binder
    for i, aa in enumerate(motif):
        pos = motif_start + i
        binder_S[0, pos] = alphabet.index(aa)
        binder_mask[pos] = 1
    binder_S = binder_S.to(device)

    X = torch.cat([X, torch.zeros(1, len_binder, 4, 3).to(device)], dim=1)
    C = torch.cat([C, torch.full((1, len_binder), 2).to(device)], dim=1)  # binder=2
    S = torch.cat([S, binder_S], dim=1)
    protein = Protein(X, C, S, device=device)
    X, C, S = protein.to_XCS()

    L_receptor = int((C == 1).sum())
    L_complex = C.shape[-1]

    # Design mask: receptor fixed to native; binder free EXCEPT the motif residues.
    full_binder_mask = [1] * L_receptor + binder_mask   # 1 => keep fixed
    mask_aa = torch.zeros(L_complex, 20)
    for i in range(L_complex):
        if not full_binder_mask[i]:
            mask_aa[i] = torch.ones(20)
    mask_aa = mask_aa[None].to(device)

    protein.sys.save_selection(gti=list(range(L_receptor)), selname='receptor')
    binder_kept = [i for i in range(L_complex) if full_binder_mask[i] and i >= L_receptor]
    protein.sys.save_selection(gti=binder_kept, selname='bindermask')
    return protein, mask_aa


def main() -> None:
    pdb, bb_w, des_w, hotspot, blen, motif, mstart, out = sys.argv[1:9]
    device = resolve_device()
    pdb, out = Path(pdb), Path(out)

    model = load_model(bb_w, des_w, device)
    vector, magnitude = hotspot_vector(pdb, hotspot, 1.0, device)
    protein, mask_aa = prepare_with_motif(pdb, int(blen), motif, int(mstart), device)

    # Build the conditioner list explicitly so we can append the subsequence one.
    substructure = conditioners.SubstructureConditioner(
        protein=protein, backbone_model=model.backbone_network,
        selection='namesel receptor',
    ).to(device)
    displacement = conditioners.InflateConditioner(vector, magnitude).to(device)
    # Subsequence conditioner over the fixed-motif selection.
    subseq = conditioners.SubsequenceConditioner(
        design_model=model.design_network, protein=protein,
        selection='namesel bindermask', n_backbones=10,
    ).to(device)
    conditioner = conditioners.ComposedConditioner(
        [displacement, substructure, subseq]
    )

    proteins = sample(model, protein, conditioner, mask_aa,
                      steps=200, n_backbones=10, n_designs=1)
    written = save_pdbs(proteins, out, pdb.stem, n_designs=1)
    print(f'Wrote {len(written)} motif-grafted backbones to {out}')


if __name__ == '__main__':
    main()
