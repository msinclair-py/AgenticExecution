# ProteinMPNN API reference

## In-process runner (`scripts/mpnn_runner.py`)

```python
design_sequences(
    pdb_path,                 # Path to a folded PDB (target chains + binder last)
    proteinmpnn_path,         # Path to the ProteinMPNN clone (utils + weights)
    num_seq=64,               # target count of unique deduped sequences
    batch_size=64,            # samples per batch (VRAM-bound; keep <= ~225)
    sampling_temp=0.1,        # 0.1 conservative .. 0.3 diverse
    model_name='v_48_020',    # noise level: v_48_002 / _010 / _020 / _030
    model_weights='soluble_model_weights',  # or 'vanilla_model_weights'
    fixed_positions=None,     # 1-indexed binder residues to KEEP fixed
    device=None,              # None -> resolve_device() (xpu/cuda/cpu)
) -> list[str]                # deduped designed sequences (binder chain)
```

Internals worth knowing:
- **Cleaning:** non-canonical residues (glycans/PTM/ligand/solvent) are stripped
  before parsing; `MSE` is kept (treated as MET).
- **Binder chain:** the *last* chain letter seen in the cleaned PDB is the design
  chain; all others are fixed via ProteinMPNN's `chain_id_dict`.
- **fixed_positions:** applied only to the binder chain (via
  `fixed_position_dict`); other chains are wholly fixed anyway.
- **Sampling:** `model.sample(...)` per batch; sequences accumulate in a set
  until `num_seq` unique are collected (or the batch budget is exhausted).
- **Output:** multi-chain sequences are '/'-joined, but only the binder is
  designed → one segment per sequence.

## Weight files layout

```
$PROTEINMPNN_PATH/
    protein_mpnn_utils.py
    protein_mpnn_run.py
    helper_scripts/{parse_multiple_chains,assign_fixed_chains,make_fixed_positions_dict}.py
    soluble_model_weights/v_48_{002,010,020,030}.pt
    vanilla_model_weights/v_48_{002,010,020,030}.pt
```

A checkpoint is loaded from `<proteinmpnn_path>/<model_weights>/<model_name>.pt`;
its `num_edges` sets the model's `k_neighbors`.

## CLI flags (`protein_mpnn_run.py`, used by `inverse_fold_cli.py`)

| Flag | Meaning |
|------|---------|
| `--jsonl_path` | parsed chains JSONL (from `parse_multiple_chains.py`) |
| `--chain_id_jsonl` | which chains are designed vs fixed (`assign_fixed_chains.py`) |
| `--fixed_positions_jsonl` | positions to keep (`make_fixed_positions_dict.py`) |
| `--out_folder` | output dir; sequences land in `<out>/seqs/*.fa` |
| `--num_seq_per_target` | sequences to sample per backbone |
| `--sampling_temp` | e.g. `'0.1'` (string) |
| `--batch_size` | per-batch sample count |
| `--model_name` | `v_48_020`, ... |
| `--path_to_model_weights` | `$PROTEINMPNN_PATH/soluble_model_weights` (or vanilla) |
| `--device` | `xpu` / `cuda` / `cpu` |

CLI `.fa` output: sequences are on lines `raw[3::2]` (header/score line then
sequence line per design); dedup them.

## The remodel_positions naming trap

The `bindify` reasoner config exposes `remodel_positions`, but the value is
passed to ProteinMPNN's **`fixed_positions`** — it lists residues to hold
**constant**. Everything *not* listed is redesigned. If you want to remodel a
small motif and keep the rest, you must list the *complement*.
