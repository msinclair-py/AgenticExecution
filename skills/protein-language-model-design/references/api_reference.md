# API reference — protein-language-model-design

Self-contained ESM-2 sequence design. Every function takes plain Python types
and returns plain dicts/lists — no project framework, no pydantic. All scripts
live in `scripts/`; add that dir to the subprocess `PYTHONPATH` (the harness
`execute_code` tool does this via `extra_paths`) so `from esm_explorer import …`
resolves.

## Model loading — `esm_explorer.load_esm`

```python
load_esm(model_id='facebook/esm2_t33_650M_UR50D', device=None)
    -> (model, tokenizer, torch_device)
```

Loads an ESM-2 masked-LM + tokenizer onto the resolved device (`xpu`→`cuda`→`cpu`
via `_device.resolve_device()`). **Load once** and thread the returned tuple into
`explore(...)`/`pseudo_log_likelihood(...)` when doing many calls — the model is
~650M parameters (2.5 GB) and reloading per call dominates runtime.

### ESM-2 model sizes (`model_id`)

| model_id | params | layers | notes |
|----------|--------|--------|-------|
| `facebook/esm2_t6_8M_UR50D` | 8M | 6 | tiny; smoke tests only |
| `facebook/esm2_t12_35M_UR50D` | 35M | 12 | fast, low quality |
| `facebook/esm2_t30_150M_UR50D` | 150M | 30 | light workhorse |
| `facebook/esm2_t33_650M_UR50D` | 650M | 33 | **default** — best quality/speed trade |
| `facebook/esm2_t36_3B_UR50D` | 3B | 36 | highest quality; needs more VRAM |

Weights come from the Hugging Face Hub on first use. Pre-fetch on a login node
(compute nodes may be offline / proxied) and pin the cache — see the SKILL's
offline note.

## Generation — `esm_explorer.explore`

```python
explore(
    sequence,                       # str, one-letter, 20 canonical AAs
    mode='infill',                  # 'infill'|'extend_n'|'extend_c'|'extend_both'
    masked_positions=None,          # list[int], 0-indexed; required for infill
    extension_n_length=0,           # int; required for extend_n/extend_both
    extension_c_length=0,           # int; required for extend_c/extend_both
    positions_to_vary=None,         # list[int], 0-indexed in ORIGINAL seq (refinement)
    num_refinement_passes=0,        # Gibbs passes over positions_to_vary post-extend
    beam_width=5,                   # beams kept per step
    temperature=1.0,                # logit temperature
    top_p=0.9,                      # nucleus cutoff, used only in Gibbs refinement
    strategy='beam',                # 'beam'|'stochastic_beam'
    num_restarts=1,                 # independent RNG runs (stochastic_beam only)
    seed=None,                      # base seed; restart i uses seed+i
    model_id='facebook/esm2_t33_650M_UR50D',
    model=None, tokenizer=None, device=None,   # preloaded load_esm() tuple
) -> {
    'sequences':       list[str],   # best-scoring first, deduped
    'scores':          list[float], # cumulative log-prob over SAMPLED positions
    'extension_spans': list[list[tuple[int,int]]],  # designed spans per sequence
    'source_sequence': str,
}
```

**Coordinates.** `masked_positions` and `positions_to_vary` are **0-indexed in
the input `sequence`**. `extension_spans` and the internal schedule are in
*extended-sequence* coordinates (after N-flank masks are prepended), so a span
`(0, n)` is the new N-flank and `(n+len, n+len+c)` is the new C-flank.

**`scores` semantics.** Cumulative `log P` over only the positions this run
sampled. Higher (less negative) is better. **Magnitude scales with how many
positions were sampled** — a cutoff tuned for an 8-mask infill will not transfer
to a 3-mask infill, and these scores are not comparable to `score_sequences`'
whole-sequence PLL. To rank designs of differing mask counts against each other,
re-score them with `score_sequences.rank_sequences`.

### Modes

- **infill** — masks `masked_positions`, resamples them left-to-right (each
  commit conditions the next). Length preserved.
- **extend_n / extend_c** — prepend/append `extension_*_length` masks, unmask
  *inside-out* (residue nearest the known sequence first).
- **extend_both** — interleaves the C and reversed-N schedules.

### Strategy

- **beam** (default) — deterministic top-`k`. Same input ⇒ same output;
  `num_restarts>1` is redundant (warned in the source tool).
- **stochastic_beam** — Gumbel-perturbed selection = sampling-without-replacement
  from the joint over (beam, vocab). Pair with `num_restarts` and a `seed` to
  harvest up to `num_restarts * beam_width` distinct designs. See `diversify.py`.

Sampling is always restricted to the 20 canonical amino acids (all other vocab
ids masked to `-inf`), so no `<eos>`/`<unk>`/special token can appear at a
designed position.

## Scoring — `score_sequences`

```python
pseudo_log_likelihood(sequence, model=None, tokenizer=None, device=None,
                      model_id='facebook/esm2_t33_650M_UR50D') -> float
rank_sequences(sequences, model_id='...', device=None)
    -> list[{'sequence', 'pll', 'pll_per_residue'}]   # best pll_per_residue first
```

PLL = Σ_i log P(residue_i | sequence with position i masked), one batched forward
pass per sequence. It's the standard masked-LM fitness proxy. **Rank
variable-length pools by `pll_per_residue`** (length-normalized); raw `pll` grows
with length and will always favour longer sequences.

## Thin condition wrappers

- `infill.infill(sequence, masked_positions, ...)` → `explore(mode='infill', …)`.
- `extend_termini.extend(sequence, n_length, c_length, ...)` → picks the
  `extend_*` mode and warns if `n_length+c_length > 10`.
- `diversify.diversify(sequence, mode, ..., num_restarts, seed)` →
  `explore(strategy='stochastic_beam', …)`.

All accept and forward a preloaded `(model, tokenizer, device)` tuple.
