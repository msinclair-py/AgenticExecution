---
name: inverse-folding
description: >-
  Redesign a protein sequence for a fixed backbone/complex structure using
  ProteinMPNN in-process — given a PDB (e.g. a Chai-1 target+binder complex),
  generate new sequences predicted to adopt that fold, keeping the target fixed
  and redesigning the last (binder) chain. Use to sequence a de novo backbone or
  to redesign a binder against a fixed target; feed the sequences back to the
  folding skill to validate. GPU-heavy; needs $PROTEINMPNN_PATH pointing to a
  ProteinMPNN repo checkout with its .pt weights.
---

# Inverse folding / sequence redesign (ProteinMPNN)

Given a backbone or complex structure, generate new amino-acid sequences
predicted to fold into it. This runs ProteinMPNN **in-process** (deliberately —
avoids a fork+exec conflict on Intel XPU) against the model's small `.pt`
checkpoints. In a design campaign this is the cheap counterpart to co-folding:
propose sequences here, then re-fold them with the **folding** skill to check
they actually adopt the target complex.

This is a **self-contained, instruction-only skill** — no wrapper script and no
`bindify` dependency. You (the agent) construct the model and run sampling
directly over ProteinMPNN's own python modules.

## When to use it

- You have a **backbone or complex PDB** (e.g. from **peptide-diffusion** or from
  **folding**) and want sequences that adopt that fold.
- You want to **redesign a binder** against a fixed target — keep the target
  chain fixed, resample the binder chain.

Feed the generated sequences back to the **folding** skill (as the binder
sequence) to validate them structurally.

## Setup — build the environment (once per project)

ProteinMPNN is not a pip package; it is a **repo checkout** whose python modules
you import directly. Build the env once, on a login node, into shared project
storage — see `../ENVIRONMENTS.md` for the cross-cutting HPC rules.

```bash
conda create -n <proj>-mpnn python=3.11 -y && conda activate <proj>-mpnn
pip install torch numpy
git clone https://github.com/dauparas/ProteinMPNN.git $PROTEINMPNN_PATH
export PROTEINMPNN_PATH=/path/to/ProteinMPNN     # skill imports protein_mpnn_utils from here
```

- The `.pt` model checkpoints **ship inside the repo**
  (`vanilla_model_weights/` and `soluble_model_weights/`) — there is **no
  separate weights download**, so this tool works on an offline compute node once
  the repo is cloned on a login node.
- The in-process import needs `$PROTEINMPNN_PATH` on `sys.path` (the procedure
  does `sys.path.insert(0, ...)`). The repo also ships a CLI,
  `protein_mpnn_run.py`, if you prefer a subprocess.
- A GPU (CUDA or Intel XPU) with torch is expected; match torch's CUDA to the
  node (see `../ENVIRONMENTS.md`).

Gate on the env var + importability; fail gracefully with a message pointing at
the required checkout:

```bash
python - <<'PY'
import os, sys
p = os.environ.get("PROTEINMPNN_PATH")
if not p or not os.path.isdir(p):
    sys.exit("inverse-folding unavailable: set $PROTEINMPNN_PATH to a ProteinMPNN "
             "repo checkout (git clone https://github.com/dauparas/ProteinMPNN). "
             "It is NOT pip-installable.")
sys.path.insert(0, p)
try:
    import torch
    from protein_mpnn_utils import (ProteinMPNN, parse_PDB, StructureDatasetPDB,
                                     tied_featurize, _S_to_seq, _scores)
except Exception as e:
    sys.exit("inverse-folding unavailable: $PROTEINMPNN_PATH set but modules/torch "
             "not importable: %r" % (e,))
print("ready")
PY
```

If this prints anything other than `ready`, stop and report that the environment
is not provisioned.

## Procedure

### Key algorithm decisions (these make the result correct)

- **Clean the PDB to canonical amino acids only** — strip glycans, ligands, and
  solvent before featurizing (ProteinMPNN handles standard residues only).
- **The designed chain is the LAST remaining chain.** Chai-1 writes the binder
  last, so after cleaning, redesign that final chain and keep the rest fixed.
- **`fixed_positions` keeps residues you list.** Build
  `fixed_positions = {designed_chain: remodel_positions}` where
  `remodel_positions` are **1-indexed** residue numbers to hold FIXED; the
  unlisted positions get resampled.
- Defaults: `model_name='v_48_020'`, `model_weights='soluble_model_weights'`,
  `sampling_temp=0.1`, `batch_size=250`.

### Construct the model

```python
import os, sys, torch
sys.path.insert(0, os.environ["PROTEINMPNN_PATH"])
from protein_mpnn_utils import (ProteinMPNN, parse_PDB, StructureDatasetPDB,
                                tied_featurize, _S_to_seq, _scores)

device = torch.device("cuda" if torch.cuda.is_available() else "xpu")
model_name    = "v_48_020"
model_weights = "soluble_model_weights"
mpnn_path = os.environ["PROTEINMPNN_PATH"]

checkpoint = torch.load(os.path.join(mpnn_path, model_weights, f"{model_name}.pt"),
                        map_location=device)
model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                    augment_eps=0.0, k_neighbors=checkpoint["num_edges"])
model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Sample sequences

Inputs you supply: `structure_path` (cleaned PDB), `remodel_positions`
(1-indexed residues to keep fixed on the designed chain), `num_seq`,
`batch_size` (default 250), `sampling_temp` (default 0.1).

```python
import numpy as np

sampling_temp = 0.1
batch_size    = 250
num_seq       = 1000

# 1. parse the cleaned PDB; the designed chain = the last chain present
pdb = parse_PDB(structure_path)          # after stripping non-canonical residues
designed_chain = <last chain id in the cleaned structure>
fixed_positions = {designed_chain: remodel_positions}   # 1-indexed, kept FIXED

dataset = StructureDatasetPDB(pdb, truncate=None, max_length=200000)
protein = dataset[0]

# 2. featurize with the fixed-position mask, then sample in batches
omit_AAs_np = np.array([aa in "X" for aa in "ACDEFGHIKLMNPQRSTVWYX"]).astype(np.float32)

designed = set()
num_batches = max(1, num_seq // batch_size)
for _ in range(num_batches):
    (X, S, mask, *_rest, chain_encoding_all, *_more) = tied_featurize(
        [protein], device, ..., fixed_positions_dict=fixed_positions, ...)
    sample = model.sample(..., temperature=sampling_temp, omit_AAs_np=omit_AAs_np, ...)
    S_sample = sample["S"]
    for b in range(S_sample.shape[0]):
        seq = _S_to_seq(S_sample[b], mask[b])   # chains joined with '/'
        designed.add(seq)

designed_sequences = sorted(designed)   # de-duplicated
```

`_S_to_seq` returns multi-chain sequences with chains joined by `'/'`; split on
`'/'` and take the designed (last) chain to get the redesigned binder sequence.

## Interpreting the result

- Output is a **de-duplicated list of designed sequences**. Lower
  `sampling_temp` → more conservative, near-native sequences; higher → more
  diverse and riskier.
- These are *predictions*, not validated designs. Re-fold each candidate with the
  **folding** skill (as the binder sequence against the same target) and rank by
  the folding interface metrics.

## Cost / fidelity

Cheap relative to folding — sampling is fast on a GPU and the weights are small.
Fidelity is only a sequence-level prediction; structural validation requires
re-folding.

## Notes

- Deciding which structures to redesign, how many sequences to draw, which
  positions to fix, and how to filter the outputs is your job as the agent.
- Cross-reference: **folding** (validate the generated sequences),
  **peptide-diffusion** (generates the backbones this skill sequences).
