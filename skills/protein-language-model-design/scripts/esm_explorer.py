"""Inside-out ESM-2 masked-LM sampler for protein sequence design.

Self-contained: uses only ``torch`` + ``transformers``. No pydantic, no Parsl,
no project framework — import ``explore`` (and ``load_esm``) and call them, or
run this file directly for a quick infill demo.

Protein language models like ESM-2 are **encoder-only masked LMs**: they were
trained to fill in ``<mask>`` tokens given surrounding sequence, not to generate
left-to-right. This module exploits exactly that — it designs sequence by
masking positions and sampling replacements — in two modes:

**infill**   — replace chosen interior positions of an existing sequence with
               ``<mask>`` and resample them (redesign a loop, diversify a hit,
               relax a motif).
**extend_n / extend_c / extend_both**
             — prepend/append ``<mask>`` tokens to a terminus and unmask them
               *inside-out* (the residue nearest the known sequence first), so
               every new residue conditions on the strongest possible context.

Why inside-out (not all-at-once): an MLM's masked predictions are only weakly
conditioned on each other, so committing one position at a time and re-running
the forward pass beats predicting every mask in a single shot. Sampling is
restricted to the 20 canonical amino acids (all other vocab ids get -inf) so the
model can't emit ``<eos>``/``<unk>``/special tokens at a designed position.

Quality note: ESM-2 MLM sampling is reliable for infill and for *short*
extensions; de-novo quality degrades sharply past ~10 newly generated residues.
Keep total extension in [3, 10] and validate every design by folding it back
(see the chai-folding skill) — a high log-prob is not a guarantee it folds.
"""
from __future__ import annotations

from itertools import zip_longest
from typing import Any

from _device import resolve_device

_AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'


def load_esm(
    model_id: str = 'facebook/esm2_t33_650M_UR50D',
    device: str | None = None,
):
    """Load an ESM-2 masked-LM + tokenizer onto the resolved device.

    Returns ``(model, tokenizer, torch_device)``. Load once and pass the tuple
    into ``explore`` / ``pseudo_log_likelihood`` when scoring or sampling many
    sequences, so the ~650M-parameter model is read from disk only once.
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = device or resolve_device()
    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(torch_device).eval()
    return model, tokenizer, torch_device


def explore(
    sequence: str,
    mode: str = 'infill',
    masked_positions: list[int] | None = None,
    extension_n_length: int = 0,
    extension_c_length: int = 0,
    positions_to_vary: list[int] | None = None,
    num_refinement_passes: int = 0,
    beam_width: int = 5,
    temperature: float = 1.0,
    top_p: float = 0.9,
    strategy: str = 'beam',
    num_restarts: int = 1,
    seed: int | None = None,
    model_id: str = 'facebook/esm2_t33_650M_UR50D',
    model=None,
    tokenizer=None,
    device=None,
) -> dict[str, Any]:
    """Design variants of ``sequence`` by inside-out masked-LM sampling.

    Arguments
    ---------
    sequence : the starting amino-acid sequence (one-letter, 20 canonical AAs).
    mode : 'infill' | 'extend_n' | 'extend_c' | 'extend_both'.
    masked_positions : 0-indexed positions in ``sequence`` to mask & resample
        (required for 'infill').
    extension_n_length / extension_c_length : residues to grow on each terminus
        (required for the matching extend mode; keep the sum in [3, 10]).
    positions_to_vary : 0-indexed positions in the ORIGINAL sequence allowed to
        change during optional post-extension Gibbs refinement.
    num_refinement_passes : Gibbs passes over ``positions_to_vary`` after the new
        flank(s) are committed (0 = original sequence held fixed).
    beam_width : beams kept per step (each fans out into <=20 AA successors).
    temperature : logit temperature before softmax (higher = flatter).
    top_p : nucleus-sampling cutoff used only during Gibbs refinement.
    strategy : 'beam' (deterministic top-k) or 'stochastic_beam' (Gumbel noise
        on each selection => sampling-without-replacement; use with num_restarts
        to harvest many distinct designs).
    num_restarts : independent runs with fresh RNG (restart i uses seed+i). Only
        meaningful under 'stochastic_beam'.
    seed : optional base RNG seed for reproducibility.
    model / tokenizer / device : a preloaded ``load_esm(...)`` tuple; if omitted,
        the model is loaded from ``model_id`` on the resolved device.

    Returns
    -------
    dict with:
      sequences        : designed sequences, best-scoring first (deduped).
      scores           : cumulative log-prob over the SAMPLED positions for each
                         (higher/less-negative is better; magnitude depends on how
                         many positions were sampled, so cutoffs don't transfer
                         across designs with different mask counts).
      extension_spans  : per-sequence list of (start, end) spans (in
                         extended-sequence coords) that were designed.
      source_sequence  : the input ``sequence``.
    """
    import torch
    import torch.nn.functional as F

    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_esm(model_id, device)
    torch_device = device

    masked_positions = list(masked_positions or [])
    positions_to_vary = list(positions_to_vary or [])
    n_ext, c_ext = int(extension_n_length), int(extension_c_length)
    binder_len = len(sequence)
    pfx = 1  # ESM-2 prepends <cls>

    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    aa_to_id = {a: tokenizer.convert_tokens_to_ids(a) for a in _AA_LETTERS}
    aa_token_ids = torch.tensor(
        list(aa_to_id.values()), dtype=torch.long, device=torch_device,
    )
    # Additive mask: 0 on the 20 AAs, -inf everywhere else. Added to logits so
    # only canonical residues can ever be sampled at a designed position.
    vocab_mask = torch.full((len(tokenizer),), float('-inf'), device=torch_device)
    vocab_mask[aa_token_ids] = 0.0

    # Initial token tape: <cls> [N-mask...] <sequence> [C-mask...] <eos>
    initial: list[int] = [cls_id]
    initial.extend([mask_id] * n_ext)
    initial.extend(aa_to_id.get(c, tokenizer.unk_token_id) for c in sequence)
    initial.extend([mask_id] * c_ext)
    initial.append(eos_id)

    if mode == 'infill':
        if not masked_positions:
            raise ValueError('infill mode requires masked_positions')
        for p in masked_positions:
            initial[pfx + n_ext + p] = mask_id

    # Inside-out unmask schedule (extended-sequence coords, excl. <cls>).
    n_range = list(range(0, n_ext))
    c_range = list(range(n_ext + binder_len, n_ext + binder_len + c_ext))
    if mode == 'infill':
        schedule = sorted(n_ext + p for p in set(masked_positions))
    elif mode == 'extend_c':
        if c_ext <= 0:
            raise ValueError('extend_c requires extension_c_length > 0')
        schedule = list(c_range)
    elif mode == 'extend_n':
        if n_ext <= 0:
            raise ValueError('extend_n requires extension_n_length > 0')
        schedule = list(reversed(n_range))  # inside-out: nearest known first
    elif mode == 'extend_both':
        if n_ext <= 0 or c_ext <= 0:
            raise ValueError('extend_both requires both extension lengths > 0')
        schedule = []
        for c, n in zip_longest(c_range, list(reversed(n_range)), fillvalue=None):
            if c is not None:
                schedule.append(c)
            if n is not None:
                schedule.append(n)
    else:
        raise ValueError(f'unknown mode: {mode}')

    def _logits_at(seq_ids: torch.Tensor, tok: int) -> torch.Tensor:
        with torch.no_grad():
            out = model(input_ids=seq_ids).logits[0, tok]
        return out.to(torch.float32)

    # Everything but the 20 AAs is pinned to -inf, so a beam expands into at most
    # 20 successors; the pool still holds up to beam_width beams.
    expand_k = min(beam_width, len(_AA_LETTERS))

    if strategy not in ('beam', 'stochastic_beam'):
        raise ValueError(f'unknown strategy: {strategy}')

    all_beams: list[tuple[Any, float]] = []
    for restart in range(max(1, num_restarts)):
        if seed is not None:
            torch.manual_seed(seed + restart)

        init_tensor = torch.tensor([initial], dtype=torch.long, device=torch_device)
        # (token tensor, real cum log-prob, perturbed cum score). We SELECT on
        # the perturbed score (== real under deterministic beam) but REPORT the
        # real log-prob to the caller.
        beams: list[tuple[Any, float, float]] = [(init_tensor, 0.0, 0.0)]

        for pos in schedule:
            tok = pfx + pos
            candidates: list[tuple[Any, float, float]] = []
            for seq_ids, real_lp, pert_lp in beams:
                logits = _logits_at(seq_ids, tok) / temperature + vocab_mask
                log_probs = F.log_softmax(logits, dim=-1)

                if strategy == 'stochastic_beam':
                    # Gumbel(0,1) = -log(-log(U)). Clamp U away from 0 and
                    # -log(U) away from 0 so neither log() blows up. The parens
                    # on (-torch.log(u)) matter: unary minus binds looser than
                    # the method call, so without them clamp_min would run on the
                    # negative value and the outer log would NaN.
                    u = torch.empty_like(log_probs).uniform_().clamp_min(1e-20)
                    gumbel = -torch.log((-torch.log(u)).clamp_min(1e-20))
                    scores = log_probs + gumbel
                else:
                    scores = log_probs

                top_scores, top_ids = torch.topk(scores, expand_k)
                real_lps = log_probs[top_ids]
                for s_val, lp_val, tid in zip(
                    top_scores.tolist(), real_lps.tolist(), top_ids.tolist(),
                ):
                    new_ids = seq_ids.clone()
                    new_ids[0, tok] = tid
                    candidates.append((new_ids, real_lp + lp_val, pert_lp + s_val))
            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = candidates[:beam_width]

        all_beams.extend((b[0], b[1]) for b in beams)

    # Optional Gibbs refinement of original-sequence positions after extension.
    if num_refinement_passes > 0 and positions_to_vary:
        vary_extended = [n_ext + p for p in positions_to_vary]
        refined: list[tuple[Any, float]] = []
        for seq_ids, cum_lp in all_beams:
            cur = seq_ids.clone()
            for _ in range(num_refinement_passes):
                for pos in vary_extended:
                    tok = pfx + pos
                    masked = cur.clone()
                    masked[0, tok] = mask_id
                    logits = _logits_at(masked, tok) / temperature + vocab_mask
                    probs = F.softmax(logits, dim=-1)
                    sorted_p, sorted_i = torch.sort(probs, descending=True)
                    cum_p = torch.cumsum(sorted_p, dim=-1)
                    cutoff = cum_p <= top_p
                    if not cutoff.any():
                        cutoff[0] = True
                    keep = sorted_p[cutoff]
                    keep = keep / keep.sum()
                    pick = sorted_i[cutoff][torch.multinomial(keep, 1).item()]
                    cur[0, tok] = pick.item()
            refined.append((cur, cum_lp))
        all_beams = refined

    # Decode + dedupe by sequence string, keeping the best score per unique seq.
    seen: dict[str, float] = {}
    for seq_ids, lp in all_beams:
        ids = seq_ids[0, 1:-1].tolist()  # strip <cls> / <eos>
        seq = ''.join(tokenizer.convert_ids_to_tokens(ids))
        lp_f = float(lp)
        if seq not in seen or lp_f > seen[seq]:
            seen[seq] = lp_f

    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    sequences = [s for s, _ in ranked]
    scores = [lp for _, lp in ranked]

    spans: list[tuple[int, int]] = []
    if mode == 'infill':
        spans = [(n_ext + p, n_ext + p + 1) for p in sorted(set(masked_positions))]
    if mode in ('extend_n', 'extend_both'):
        spans.append((0, n_ext))
    if mode in ('extend_c', 'extend_both'):
        spans.append((n_ext + binder_len, n_ext + binder_len + c_ext))

    return {
        'sequences': sequences,
        'scores': scores,
        'extension_spans': [spans] * len(sequences),
        'source_sequence': sequence,
    }


if __name__ == '__main__':
    # Quick demo: mask three interior positions of a toy sequence and resample.
    seq = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ'
    out = explore(
        seq, mode='infill', masked_positions=[5, 6, 7], beam_width=5,
    )
    print('source :', out['source_sequence'])
    for s, sc in zip(out['sequences'], out['scores']):
        print(f'{sc:8.3f}  {s}')
