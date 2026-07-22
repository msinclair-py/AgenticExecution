"""Rank arbitrary candidate sequences by ESM-2 pseudo-log-likelihood (PLL).

The other half of PLM sequence design: given a pool of candidates (from infill /
extension / ProteinMPNN / anywhere), score how "natural" each looks to ESM-2 and
rank them before spending compute on folding/MD.

Pseudo-log-likelihood for a masked LM: mask each position one at a time, run the
model, read log P(true residue | rest of sequence), and sum over positions. It's
the standard MLM fitness proxy (higher = more model-preferred). ``pll_per_residue``
(PLL / length) is length-normalized so sequences of different lengths compare
fairly — always rank variable-length pools by the per-residue value.

This scores WHOLE sequences and is independent of ``explore``'s ``scores``, which
are cumulative log-probs over only the *sampled* positions of one design run.

    from score_sequences import rank_sequences
    ranked = rank_sequences(['MKT...', 'MRT...', 'MKS...'])
    for r in ranked:
        print(round(r['pll_per_residue'], 4), r['sequence'])

CLI:  python score_sequences.py SEQ1 SEQ2 SEQ3 ...
"""
from __future__ import annotations

from typing import Any

from esm_explorer import load_esm


def pseudo_log_likelihood(
    sequence: str,
    model=None,
    tokenizer=None,
    device=None,
    model_id: str = 'facebook/esm2_t33_650M_UR50D',
) -> float:
    """Sum of log P(residue_i | sequence \\ i) over all positions of ``sequence``.

    One forward pass per position (batched over positions). Pass a preloaded
    ``load_esm(...)`` tuple to score many sequences without reloading the model.
    """
    import torch
    import torch.nn.functional as F

    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_esm(model_id, device)

    mask_id = tokenizer.mask_token_id

    enc = tokenizer(sequence, return_tensors='pt')
    ids = enc['input_ids'][0].to(device)          # <cls> ... <eos>
    positions = list(range(1, ids.shape[0] - 1))  # skip special tokens

    # One masked copy of the tape per position -> a single batched forward pass.
    batch = ids.unsqueeze(0).repeat(len(positions), 1).clone()
    for row, tok in enumerate(positions):
        batch[row, tok] = mask_id

    total = 0.0
    with torch.no_grad():
        logits = model(input_ids=batch).logits
    for row, tok in enumerate(positions):
        true_id = int(ids[tok].item())  # the real residue's token id at this pos
        log_probs = F.log_softmax(logits[row, tok].to(torch.float32), dim=-1)
        total += float(log_probs[true_id].item())
    return total


def rank_sequences(
    sequences: list[str],
    model_id: str = 'facebook/esm2_t33_650M_UR50D',
    device=None,
) -> list[dict[str, Any]]:
    """Score every sequence once (shared model) and return them best-first.

    Each item: ``{'sequence', 'pll', 'pll_per_residue'}``. Rank on
    ``pll_per_residue`` for length-fair comparison.
    """
    model, tokenizer, device = load_esm(model_id, device)
    scored = []
    for seq in sequences:
        pll = pseudo_log_likelihood(seq, model, tokenizer, device)
        scored.append({
            'sequence': seq,
            'pll': pll,
            'pll_per_residue': pll / max(1, len(seq)),
        })
    scored.sort(key=lambda d: d['pll_per_residue'], reverse=True)
    return scored


if __name__ == '__main__':
    import sys

    seqs = sys.argv[1:] or ['MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ']
    for r in rank_sequences(seqs):
        print(f"{r['pll_per_residue']:8.4f}  (PLL {r['pll']:9.3f})  {r['sequence']}")
