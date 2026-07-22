"""Diversify: harvest MANY distinct designs via stochastic-beam + restarts.

Deterministic beam search returns the same beams every run, so to explore
sequence space you switch to ``strategy='stochastic_beam'`` (Gumbel noise on
each per-beam selection = sampling-without-replacement) and repeat with
``num_restarts`` independent RNG seeds. Up to ``num_restarts * beam_width``
distinct sequences come back after dedup, best-scoring first.

Works for infill or extension — pass through the same mask/extension args.

    from diversify import diversify
    out = diversify(
        'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ',
        mode='infill', masked_positions=[5, 6, 7, 8],
        beam_width=10, num_restarts=8, seed=0,
    )
    print(len(out['sequences']), 'unique designs')

CLI:  python diversify.py <SEQUENCE> <comma,positions> [num_restarts]
"""
from __future__ import annotations

from typing import Any

from esm_explorer import explore


def diversify(
    sequence: str,
    mode: str = 'infill',
    masked_positions: list[int] | None = None,
    extension_n_length: int = 0,
    extension_c_length: int = 0,
    beam_width: int = 10,
    temperature: float = 1.0,
    num_restarts: int = 8,
    seed: int | None = 0,
    model_id: str = 'facebook/esm2_t33_650M_UR50D',
    model=None,
    tokenizer=None,
    device=None,
) -> dict[str, Any]:
    """Stochastic-beam sampling with restarts to maximize distinct designs.

    Raise ``temperature`` (e.g. 1.2–1.5) and/or ``num_restarts`` for more
    diversity; lower them to stay near the model's high-confidence region.
    """
    return explore(
        sequence,
        mode=mode,
        masked_positions=masked_positions,
        extension_n_length=extension_n_length,
        extension_c_length=extension_c_length,
        beam_width=beam_width,
        temperature=temperature,
        strategy='stochastic_beam',
        num_restarts=num_restarts,
        seed=seed,
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


if __name__ == '__main__':
    import sys

    seq = sys.argv[1]
    positions = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0]
    restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out = diversify(seq, mode='infill', masked_positions=positions, num_restarts=restarts)
    print(f'{len(out["sequences"])} unique designs (source: {out["source_sequence"]})')
    for s, sc in zip(out['sequences'], out['scores']):
        print(f'{sc:8.3f}  {s}')
