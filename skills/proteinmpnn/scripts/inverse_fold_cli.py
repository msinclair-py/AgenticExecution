"""CLI / subprocess fallback: drive ProteinMPNN via protein_mpnn_run.py.

Prefer the in-process runner (mpnn_runner.py). Use this only when you must run
the stock repo scripts. NOTE: shelling out to a process that re-inits the GPU
after an in-process GPU fold in the same worker can hit a torch device conflict —
run it in a fresh process/worker.

Pipeline (mirrors the repo's helper scripts):
  1. parse_multiple_chains.py   -> parsed_design.jsonl
  2. assign_fixed_chains.py     -> chain_B_design.jsonl (design chain B, fix rest)
  3. make_fixed_positions_dict.py -> fixed_design.jsonl (positions to keep)
  4. protein_mpnn_run.py        -> <out>/seqs/*.fa

Usage:
    PROTEINMPNN_PATH=/path/to/ProteinMPNN \
        python inverse_fold_cli.py <pdb_dir_or_file> <out_dir> [design_chain=B] [fixed csv]
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(pdb: str, out_dir: str, design_chain: str = 'B',
         fixed_csv: str = '') -> None:
    ppath = Path(os.environ['PROTEINMPNN_PATH'])
    helpers = ppath / 'helper_scripts'
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parsed = out / 'parsed_design.jsonl'
    assigned = out / 'chain_design.jsonl'
    fixed = out / 'fixed_design.jsonl'
    py = sys.executable

    subprocess.run([py, str(helpers / 'parse_multiple_chains.py'),
                    '--input_path', pdb, '--output_path', str(parsed)], check=True)
    subprocess.run([py, str(helpers / 'assign_fixed_chains.py'),
                    '--input_path', str(parsed), '--output_path', str(assigned),
                    '--chain_list', design_chain], check=True)
    if fixed_csv:
        subprocess.run([py, str(helpers / 'make_fixed_positions_dict.py'),
                        '--input_path', str(parsed), '--output_path', str(fixed),
                        '--chain_list', design_chain,
                        '--position_list', ' '.join(fixed_csv.split(','))], check=True)

    cmd = [py, '-u', str(ppath / 'protein_mpnn_run.py'),
           '--jsonl_path', str(parsed),
           '--chain_id_jsonl', str(assigned),
           '--out_folder', str(out),
           '--num_seq_per_target', '32',
           '--sampling_temp', '0.1',
           '--batch_size', '32',
           '--model_name', 'v_48_020',
           '--path_to_model_weights', str(ppath / 'soluble_model_weights')]
    if fixed_csv:
        cmd += ['--fixed_positions_jsonl', str(fixed)]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    print('Sequences written under', out / 'seqs')


if __name__ == '__main__':
    chain = sys.argv[3] if len(sys.argv) > 3 else 'B'
    fixed = sys.argv[4] if len(sys.argv) > 4 else ''
    main(sys.argv[1], sys.argv[2], chain, fixed)
