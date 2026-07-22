"""Self-contained in-process ProteinMPNN design.

Imports ``protein_mpnn_utils`` directly (no subprocess) to avoid the
torch.xpu/CUDA fork+exec conflict that hits when a worker already initialized
the GPU and then spawns a child that re-inits it. Requires PROTEINMPNN_PATH (or
pass proteinmpnn_path) pointing at the ProteinMPNN clone.

Design contract: the binder is the LAST chain in the (cleaned) PDB; every other
chain is held fixed. `fixed_positions` names 1-indexed residues *within the
binder chain* to additionally keep constant.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

from _device import resolve_device

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
CANONICAL_AA_3 = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'MSE',
}


def _clean_pdb(pdb_path: Path, out_dir: Path) -> Path:
    """Drop ATOM/HETATM lines that aren't canonical amino acids (keep MSE)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned = out_dir / pdb_path.name
    with open(pdb_path, 'rb') as fin, open(cleaned, 'w') as fout:
        for raw in fin:
            line = raw.decode('utf-8', 'ignore')
            if line[:6].startswith(('ATOM', 'HETATM')):
                if line[17:20].strip() not in CANONICAL_AA_3:
                    continue
            fout.write(line)
    return cleaned


def _binder_chain(pdb_path: Path) -> str:
    """Last chain letter appearing in the PDB = the binder."""
    last = 'A'
    with open(pdb_path) as f:
        for line in f:
            if line[:6].startswith(('ATOM', 'HETATM')):
                last = line[21]
    return last


def _load_model(proteinmpnn_path: Path, model_name: str, model_weights: str,
                device: str):
    from protein_mpnn_utils import ProteinMPNN as _MPNN

    if str(proteinmpnn_path) not in sys.path:
        sys.path.insert(0, str(proteinmpnn_path))
    torch_device = torch.device(device)
    ckpt = torch.load(
        proteinmpnn_path / model_weights / f'{model_name}.pt',
        map_location=torch_device,
    )
    model = _MPNN(
        ca_only=False, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=ckpt['num_edges'],
    )
    model.to(torch_device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, torch_device


def design_sequences(
    pdb_path: Path,
    proteinmpnn_path: Path,
    num_seq: int = 64,
    batch_size: int = 64,
    sampling_temp: float = 0.1,
    model_name: str = 'v_48_020',
    model_weights: str = 'soluble_model_weights',
    fixed_positions: list[int] | None = None,
    device: str | None = None,
) -> list[str]:
    """Design `num_seq` (deduped) sequences for the binder chain of `pdb_path`."""
    from protein_mpnn_utils import (
        StructureDatasetPDB, _S_to_seq, _scores, parse_PDB, tied_featurize,
    )

    device = device or resolve_device()
    proteinmpnn_path = Path(proteinmpnn_path)
    fixed_positions = list(fixed_positions or [])

    cleaned = _clean_pdb(Path(pdb_path), Path(pdb_path).parent / 'clean')
    binder = _binder_chain(cleaned)
    model, torch_device = _load_model(
        proteinmpnn_path, model_name, model_weights, device,
    )

    omit_AAs_np = np.array([aa in 'X' for aa in ALPHABET], dtype=np.float32)
    bias_AAs_np = np.zeros(len(ALPHABET), dtype=np.float32)

    pdb_dict_list = parse_PDB(str(cleaned), ca_only=False)
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200_000)
    protein = dataset[0]
    name = protein['name']

    all_chains = [k[-1:] for k in protein if k.startswith('seq_chain_')]
    designed = [binder]
    fixed_chains = [c for c in all_chains if c not in designed]
    chain_id_dict = {name: (designed, fixed_chains)}

    fixed_position_dict = None
    if fixed_positions:
        fixed_position_dict = {
            name: {c: (fixed_positions if c == binder else []) for c in all_chains}
        }

    results: set[str] = set()
    with torch.no_grad():
        batch_clones = [copy.deepcopy(protein) for _ in range(batch_size)]
        (X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list,
         visible_list_list, masked_list_list, masked_chain_length_list_list,
         chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask,
         tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all,
         bias_by_res_all, tied_beta) = tied_featurize(
            batch_clones, torch_device, chain_id_dict,
            fixed_position_dict=fixed_position_dict, omit_AA_dict=None,
            tied_positions_dict=None, pssm_dict=None, bias_by_res_dict=None,
            ca_only=False,
        )

        # Sample enough batches to reach num_seq unique sequences.
        max_batches = max(1, (num_seq + batch_size - 1) // batch_size) + 3
        for _ in range(max_batches):
            randn = torch.randn(chain_M.shape, device=torch_device)
            sample = model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx, mask=mask,
                temperature=sampling_temp, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias,
                pssm_multi=0.0, pssm_log_odds_flag=False,
                pssm_log_odds_mask=(pssm_log_odds_all > 0.0).float(),
                pssm_bias_flag=False, bias_by_res=bias_by_res_all,
            )
            S_sample = sample['S']
            for b in range(batch_size):
                seq = _S_to_seq(S_sample[b], chain_M[b])
                m_lens = masked_chain_length_list_list[b]
                pieces, cur = [], 0
                for ln in m_lens:
                    pieces.append(seq[cur:cur + ln])
                    cur += ln
                results.add('/'.join(pieces))
            if len(results) >= num_seq:
                break

    return list(results)[:num_seq]


if __name__ == '__main__':
    import os

    pdb = Path(sys.argv[1])
    seqs = design_sequences(
        pdb, Path(os.environ['PROTEINMPNN_PATH']),
        num_seq=int(sys.argv[2]) if len(sys.argv) > 2 else 8,
    )
    for s in seqs:
        print(s)
