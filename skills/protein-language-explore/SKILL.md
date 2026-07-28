---
name: protein-language-explore
description: >-
  Grow or fill a binder/peptide sequence using ESM-2's learned amino-acid
  distribution — masked-language-model sequence GENERATION via beam search, not
  mutant-scanning and not embeddings. Use to EXTEND a peptide at its N- or
  C-terminus, or to INFILL masked positions, with high-likelihood residues.
  Modes: infill, extend_n, extend_c, extend_both, with optional Gibbs
  refinement. Needs HuggingFace transformers + torch and the ESM-2 650M weights;
  GPU strongly recommended.
---

# Protein language exploration (HuggingFace transformers + torch)

Generate binder/peptide sequence variants from the learned distribution of the
ESM-2 650M masked language model (`facebook/esm2_t33_650M_UR50D`). This is
**inside-out masked-LM generation**, not mutation scanning and not embedding
extraction: you place `<mask>` tokens (either over existing positions to
resample them, or as new N-/C-terminal positions to grow the peptide) and let
the model fill them one at a time with a top-k beam search restricted to the 20
canonical amino acids. Cost/fidelity: one forward pass per masked position per
beam; a handful of masks with a modest beam width is seconds-to-minutes on GPU,
much slower on CPU. Fidelity is the language model's opinion of sequence
likelihood — good for plausible peptides, not a binding-affinity guarantee.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) write and run the sampler yourself
over `transformers`. NEVER import bindify.

## When to use it

- You have a binder/peptide sequence and want to **extend** it — grow new
  residues at the N-terminus, C-terminus, or both, from the language model's
  distribution.
- You want to **infill** — resample a set of interior positions you mark as
  masked, keeping the rest fixed.
- You want several ranked candidate sequences (by model log-likelihood) to feed
  downstream, e.g. to fold and evaluate.

Do **not** use it for embeddings, per-residue mutant ΔΔ-scoring, or structure
prediction. To turn generated sequences into structures, cross-reference the
folding skill.

## Setup — build the environment (once per project)

Build the env once, on an internet-capable login node, into shared project
storage — see `../ENVIRONMENTS.md` for the cross-cutting HPC rules.

```bash
conda create -n <proj>-esm python=3.11 -y && conda activate <proj>-esm
pip install torch transformers
```

**Pre-cache the model on an internet node** into a shared `HF_HOME` so offline
compute nodes reuse it:

```bash
export HF_HOME=$SHARED_WEIGHTS/hf
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esm2_t33_650M_UR50D')"
```

On offline compute nodes set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (and
re-export the same `HF_HOME`). GPU is strongly recommended; on CPU generation is
slow but works.

**CAUTION (real finding):** a cluster's prebuilt / "module" torch or a repurposed
env can fail to import with `libcudnn.so.8: cannot open shared object file`.
Prefer a clean env with a torch whose CUDA matches the node, or a
`python -m venv --system-site-packages` venv over a known-good base torch, rather
than fighting a mismatched module build. See the CUDA-lib matching notes in
`../ENVIRONMENTS.md`.

A quick gate for an already-built env (install only if missing):

```bash
python -c "import torch, transformers" 2>/dev/null && echo ready || pip install torch transformers
```

## Procedure

Load the model once, build a token "tape", and unmask positions **inside-out** —
the masked position adjacent to known sequence is decided first, so each new
residue is conditioned on the residues nearest it.

Key mechanics:

- **Token tape.** For an extension, build `<cls> [N-mask]* <binder tokens>
  [C-mask]* <eos>`. For infill, tokenize the binder normally, then overwrite the
  `masked_positions` with the tokenizer's `<mask>` id.
- **Vocab mask.** Restrict logits to the 20 canonical AAs with an additive
  `-inf` mask over every other vocab id (special tokens, `X`, `B`, `U`, `Z`,
  etc.), so the model can only emit real residues.
- **Per-position beam step.** For each still-masked position (chosen inside-out),
  run `model(input_ids).logits[0, pos]`, add the vocab mask, `log_softmax`, and
  keep the top `beam_width` `(token, logprob)` continuations. Extend each beam,
  carry the cumulative log-prob, prune back to `beam_width`.
- **Score.** Cumulative log-prob **over only the sampled (masked) positions**
  (≤ 0; higher = more likely under the model).
- **Strategy.** `beam` is deterministic (pure top-k). `stochastic_beam` adds
  Gumbel noise to the logits with a `temperature`, optional nucleus `top_p`
  filtering, `num_restarts` independent runs, and a `seed` for reproducibility.
- **Gibbs refinement (optional).** After a full pass, optionally re-mask and
  resample already-filled positions for a few passes to let residues relax
  against each other.

Inputs: `binder_sequence`, `mode` (`infill` | `extend_n` | `extend_c` |
`extend_both`), `masked_positions` (infill) or extension lengths (extend),
`beam_width`, `temperature`, `top_p`, `strategy`, `num_restarts`, `seed`.
Output: `{sequences, scores, extension_spans, source_sequence}`.

Compact sketch of the load + vocab-mask + one inside-out beam step (this shows
the core idea; wrap it in your own tape-builder and beam bookkeeping):

```python
import torch, torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "facebook/esm2_t33_650M_UR50D"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device).eval()

# Additive vocab mask: allow only the 20 canonical amino acids.
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
allowed = torch.full((model.config.vocab_size,), float("-inf"), device=device)
for aa in CANONICAL:
    allowed[tok.convert_tokens_to_ids(aa)] = 0.0

def top_beams(input_ids, pos, beam_width, temperature=1.0, stochastic=False):
    """One masked-position step: return top-k (token_id, logprob)."""
    with torch.no_grad():
        logits = model(input_ids).logits[0, pos]      # (vocab,)
    logits = logits + allowed                          # forbid non-AA tokens
    if stochastic:
        g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-9)))
        logits = logits / temperature + g              # Gumbel noise
    logp = F.log_softmax(logits, dim=-1)
    vals, idx = torch.topk(logp, beam_width)
    return list(zip(idx.tolist(), vals.tolist()))

# Inside-out ordering: fill the masked position nearest the known sequence
# first, so each new residue conditions on its neighbours. Maintain a list of
# (input_ids, cumulative_logprob) beams; at each masked position expand every
# beam with top_beams(...), keep the best `beam_width`, and continue until no
# <mask> remains. Final score = cumulative logprob over the sampled positions.
```

## Interpreting the result

- **Higher (closer to 0) score = more likely** under ESM-2. Rank candidates by
  score; the top ones are the model's most confident completions.
- `extend_both` grows both termini; `extension_spans` records which residue
  indices are newly generated vs. inherited from `source_sequence`.
- `stochastic_beam` with higher `temperature`/`top_p` and more `num_restarts`
  gives more diverse (less likely) candidates; `beam` gives the single most
  likely completion set. Use `seed` to reproduce a stochastic run.
- These are sequence-likelihood judgements only. Validate binding/structure
  downstream — fold the top sequences and evaluate them.

## Notes

- Deciding which binder to grow, how far to extend, which interior positions to
  infill, and how many candidates to carry forward is your job as the agent.
  This skill only generates and scores sequences.
- Natural next step: fold the top-ranked sequences (folding skill) and, for a
  complex, evaluate the contact footprint (contact-eval skill).
