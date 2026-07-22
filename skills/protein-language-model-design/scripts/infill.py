"""Infill: mask chosen interior positions and resample them with ESM-2.

The workhorse mode. Use it to redesign a loop, diversify a hit, or relax a
motif while keeping the rest of the fold-defining sequence intact.

    from infill import infill
    out = infill(
        'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ',
        masked_positions=[10, 11, 12, 13],   # 0-indexed, contiguous or scattered
        beam_width=8,
    )
    for seq, score in zip(out['sequences'], out['scores']):
        print(round(score, 3), seq)

CLI:  python infill.py <SEQUENCE> <comma,separated,0indexed,positions>
"""
from __future__ import annotations

from typing import Any

from esm_explorer import explore, load_esm


def infill(
    sequence: str,
    masked_positions: list[int],
    beam_width: int = 8,
    temperature: float = 1.0,
    model_id: str = 'facebook/esm2_t33_650M_UR50D',
    model=None,
    tokenizer=None,
    device=None,
) -> dict[str, Any]:
    """Resample ``masked_positions`` (0-indexed) of ``sequence``.

    Returns the ``explore`` result dict (``sequences`` best-first, ``scores``,
    ``extension_spans``, ``source_sequence``). All designs keep the length and
    every unlisted position of the input.
    """
    if not masked_positions:
        raise ValueError('infill needs at least one masked position')
    return explore(
        sequence,
        mode='infill',
        masked_positions=masked_positions,
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
    positions = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0]
    out = infill(seq, positions)
    print('source:', out['source_sequence'])
    for s, sc in zip(out['sequences'], out['scores']):
        print(f'{sc:8.3f}  {s}')
