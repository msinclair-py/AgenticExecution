---
name: protein-language-model-design
description: Sequence design with a protein language model (ESM-2). Given an existing amino-acid sequence, generate variants by masked-LM sampling — infill chosen interior positions, or extend the N/C termini with inside-out beam search — and rank arbitrary candidate sequences by ESM-2 pseudo-log-likelihood. Covers the two operating modes (infill vs terminal extension), deterministic vs stochastic-beam sampling, restart-based diversification, optional Gibbs refinement, model-size choice, and the score/rank half of PLM design. Self-contained torch + transformers (no ProteinMPNN, no backbone/structure needed) — use to redesign a loop, diversify a hit, grow a peptide, or triage candidates before folding. Complements proteinmpnn (structure-conditioned sequence design); validate designs with chai-folding.
license: MIT (ESM-2 weights are MIT, Meta/FAIR; this skill text is project-local)
metadata:
    skill-author: local project skill (AgenticExecution)
    verified-against: transformers AutoModelForMaskedLM (facebook/esm2_t33_650M_UR50D); inside-out MLM sampler ported from bindify ESMExplorer
---

# Protein Language Model Sequence Design (ESM-2)

## Overview

A protein language model (pLM) like **ESM-2** is an encoder-only *masked* LM: it
was trained to fill in `<mask>` tokens from surrounding sequence. This skill uses
that directly to **design sequence** — no structure, no backbone required (that's
what separates it from `proteinmpnn`, which conditions on a PDB). Two things a pLM
does well:

1. **Generate** variants by masking positions and resampling them
   (`explore`) — either *infill* interior positions or *extend* a terminus.
2. **Score/rank** any candidate sequences by pseudo-log-likelihood
   (`score_sequences`) — a cheap fitness proxy to triage before folding.

Everything is plain `torch` + `transformers`. Import the helpers from `scripts/`
and call them from an `execute_code` snippet; the harness puts the skill's
`scripts/` on `PYTHONPATH` via `extra_paths`.

### Why inside-out (the key idea)

An MLM's masked predictions are only weakly conditioned on each other, so
predicting all masks in one shot is weak. Instead we commit **one position at a
time** and re-run the forward pass. For extensions we unmask **inside-out** — the
new residue nearest the known sequence first — so every generated residue sees
the strongest possible context. This is why extension quality holds for short
flanks but **degrades sharply past ~10 de-novo residues**; keep total extension
in [3, 10].

Sampling is restricted to the **20 canonical amino acids** (all other vocab ids
forced to `-inf`), so a designed position can never become `<eos>`/`<unk>`/a
special token.

---

## 1. Infill — resample interior positions (the workhorse)

Redesign a loop, diversify a hit, or relax a motif while keeping the rest fixed.
Positions are **0-indexed in the input sequence**; length is preserved.

```python
from infill import infill                 # scripts/infill.py

out = infill(
    'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ',
    masked_positions=[10, 11, 12, 13],     # contiguous or scattered
    beam_width=8,
)
for seq, score in zip(out['sequences'], out['scores']):
    print(round(score, 3), seq)           # best (least-negative) first
```

Returns `{sequences, scores, extension_spans, source_sequence}` (see
`references/api_reference.md`). `scores` is cumulative `log P` over the masked
positions only.

---

## 2. Extend — grow the N/C termini inside-out

```python
from extend_termini import extend         # scripts/extend_termini.py

out = extend('GSHMLEDPQRAA', n_length=0, c_length=6, beam_width=8)
```

- Pick the mode implicitly: nonzero `n_length` and/or `c_length` selects
  `extend_n` / `extend_c` / `extend_both`.
- **Keep `n_length + c_length` in [3, 10].** It warns above 10.
- Optionally polish the *original* residues after the flank is committed:
  pass `positions_to_vary=[...]` (0-indexed in the original) and
  `num_refinement_passes=2` for a short Gibbs pass; leave both default to freeze
  the original sequence and design only the new flank.

---

## 3. Diversify — harvest many distinct designs

Deterministic beam search returns the *same* beams every run. To explore
sequence space, switch to stochastic beam + restarts:

```python
from diversify import diversify           # scripts/diversify.py

out = diversify(
    'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ',
    mode='infill', masked_positions=[5, 6, 7, 8],
    beam_width=10, num_restarts=8, seed=0, temperature=1.2,
)
print(len(out['sequences']), 'unique designs')
```

Up to `num_restarts * beam_width` unique sequences come back (deduped,
best-first). Raise `temperature` (≈1.2–1.5) and/or `num_restarts` for more
spread; lower them to stay near the model's confident region. Works for
extension too (pass `extension_*_length` and the matching `mode`).

---

## 4. Score / rank candidates (the other half of pLM design)

Given a pool of candidates from anywhere (this skill, ProteinMPNN, a mutagenesis
sweep), rank them by ESM-2 **pseudo-log-likelihood** before spending fold/MD
compute:

```python
from score_sequences import rank_sequences   # scripts/score_sequences.py

ranked = rank_sequences(['MKTAY...', 'MRTAY...', 'MKSAY...'])
for r in ranked:
    print(round(r['pll_per_residue'], 4), r['sequence'])   # best first
```

PLL masks each position in turn and sums `log P(true residue | rest)`. **Rank
variable-length pools by `pll_per_residue`** — raw `pll` grows with length. This
whole-sequence score is *independent of* and not comparable to `explore`'s
`scores` (which cover only one run's sampled positions).

---

## 5. Efficient batch use — load the model once

`explore`/`infill`/`extend`/`diversify`/`pseudo_log_likelihood` all accept a
preloaded `(model, tokenizer, device)` tuple. For many calls, load once:

```python
from esm_explorer import load_esm, explore

model, tok, dev = load_esm('facebook/esm2_t33_650M_UR50D')   # ~2.5 GB, once
results = [
    explore(seq, mode='infill', masked_positions=[3, 4, 5],
            model=model, tokenizer=tok, device=dev)
    for seq in many_sequences
]
```

Reloading the 650M model per call otherwise dominates runtime. High throughput
across a campaign comes from fanning **independent** `execute_code` snippets —
the Parsl runtime spreads them across GPUs/tiles (see the deploy-polaris /
deploy-aurora skills); within one snippet, load once and reuse.

---

## Choosing the model

`model_id` defaults to `facebook/esm2_t33_650M_UR50D` (650M, the quality/speed
sweet spot). Smaller (`t30_150M`, `t12_35M`) for speed/smoke tests; `t36_3B` for
top quality if VRAM allows. Full table in `references/api_reference.md`.

## Best practices

- **Device-agnostic:** `_device.resolve_device()` picks `xpu` (Aurora) → `cuda`
  (Polaris/Lambda) → `cpu`. A 650M model runs on one GPU/tile comfortably, on CPU
  slowly.
- **Load the model once** per snippet and thread the tuple through (§5).
- **0-indexed positions** everywhere (`masked_positions`, `positions_to_vary`).
- **Short extensions only** (≤10 de-novo residues); prefer infill when you can.
- **`explore` scores rank within one run; PLL ranks across a pool.** Don't
  compare the two, and don't compare `explore` scores across runs with different
  mask counts — re-score with `rank_sequences`.
- **Validate structurally.** A high pLM score is a *prior*, not a fold — pass
  designs to `chai-folding` and score the interface (ipSAE), then optionally
  `md-simulation`. In the binder pipeline this skill sits alongside `proteinmpnn`
  as a sequence-design source feeding `chai-folding`.
- **Offline weights on HPC.** Compute nodes are often offline/proxied. Pre-fetch
  on a login node and pin the cache:
  ```bash
  export HF_HOME=/lus/flare/projects/FRAME-IDP/$USER/hf_cache
  python -c "from transformers import AutoModelForMaskedLM, AutoTokenizer; \
             m='facebook/esm2_t33_650M_UR50D'; AutoTokenizer.from_pretrained(m); \
             AutoModelForMaskedLM.from_pretrained(m)"
  # then on compute nodes:  export HF_HOME=...   HF_HUB_OFFLINE=1
  ```

## Common pitfalls

1. **1-indexed positions.** All position args are 0-indexed; off-by-one masks the
   wrong residue.
2. **Long extensions.** `n_length + c_length > 10` → unreliable de-novo sequence.
   Infill is far safer than long extension.
3. **`num_restarts` with deterministic beam.** Redundant — every restart is
   identical. Restarts only diversify under `strategy='stochastic_beam'`.
4. **Comparing incomparable scores.** `explore` scores (sampled-position log-prob,
   run-specific) ≠ `pseudo_log_likelihood` (whole-sequence). Rank a mixed pool
   with `rank_sequences` on `pll_per_residue`.
5. **Reloading the model per call** → runtime dominated by I/O. Preload (§5).
6. **First-call download on an offline node** → hang/error. Pre-fetch weights.
7. **Treating a pLM score as validation.** It isn't structure — always fold-check.

## Resources

### references/
- `api_reference.md` — full signatures, return schema, coordinate conventions,
  score semantics, and the ESM-2 model-size table.

### scripts/
- `_device.py` — `resolve_device()` (xpu → cuda → cpu).
- `esm_explorer.py` — **core.** `load_esm()` + `explore()` (inside-out masked-LM
  sampler: infill / extend, beam & stochastic-beam, Gibbs refinement, dedup).
- `infill.py` — resample chosen interior positions.
- `extend_termini.py` — grow N/C termini (+ optional refinement).
- `diversify.py` — stochastic-beam + restarts to harvest many designs.
- `score_sequences.py` — `pseudo_log_likelihood()` / `rank_sequences()` for
  triaging arbitrary candidate pools.
