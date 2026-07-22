"""Extend: grow the N- and/or C-terminus with inside-out ESM-2 sampling.

New ``<mask>`` tokens are added to a terminus and unmasked closest-to-known
first, so each new residue conditions on the residue just committed next to it —
the strongest context a masked LM can give for de-novo generation.

Keep the total added length in [3, 10]: ESM-2 MLM quality falls off sharply
past ~10 de-novo residues. Optionally polish the ORIGINAL residues afterward
with a short Gibbs pass (``positions_to_vary`` + ``num_refinement_passes``).

    from extend_termini import extend
    out = extend('GSHMLEDPQRAA', n_length=0, c_length=6, beam_width=8)
    for seq, score in zip(out['sequences'], out['scores']):
        print(round(score, 3), seq)

CLI:  python extend_termini.py <SEQUENCE> <n_length> <c_length>
"""
from __future__ import annotations

from typing import Any

from esm_explorer import explore


def extend(
    sequence: str,
    n_length: int = 0,
    c_length: int = 0,
    positions_to_vary: list[int] | None = None,
    num_refinement_passes: int = 0,
    beam_width: int = 8,
    temperature: float = 1.0,
    model_id: str = 'facebook/esm2_t33_650M_UR50D',
    model=None,
    tokenizer=None,
    device=None,
) -> dict[str, Any]:
    """Grow ``n_length`` residues on the N-terminus and/or ``c_length`` on the C.

    ``positions_to_vary`` (0-indexed in the ORIGINAL sequence) with
    ``num_refinement_passes > 0`` runs Gibbs refinement over those original
    positions after the flanks are committed; leave both default to freeze the
    original sequence and only design the new flank(s).
    """
    total = int(n_length) + int(c_length)
    if total <= 0:
        raise ValueError('extend needs n_length > 0 and/or c_length > 0')
    if total > 10:
        # Not fatal, but the caller should know quality degrades here.
        print(f'[warn] total extension {total} > 10; ESM-2 de-novo quality '
              f'degrades sharply past ~10 residues.')

    if n_length and c_length:
        mode = 'extend_both'
    elif n_length:
        mode = 'extend_n'
    else:
        mode = 'extend_c'

    return explore(
        sequence,
        mode=mode,
        extension_n_length=int(n_length),
        extension_c_length=int(c_length),
        positions_to_vary=positions_to_vary,
        num_refinement_passes=num_refinement_passes,
        beam_width=beam_width,
        temperature=temperature,
        strategy='beam',
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


if __name__ == '__main__':
    import sys

    seq = sys.argv[1]
    n_len = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    c_len = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out = extend(seq, n_length=n_len, c_length=c_len)
    print('source:', out['source_sequence'])
    for s, sc in zip(out['sequences'], out['scores']):
        print(f'{sc:8.3f}  {s}')
