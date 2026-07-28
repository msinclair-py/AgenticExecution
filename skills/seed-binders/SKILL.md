---
name: seed-binders
description: >-
  Assemble a clean, canonical starter pool of candidate binder/peptide sequences
  at the very start of a campaign — parse peptipedia-style FASTA(s), drop
  sequences with non-canonical residues, length-filter, dedupe (preferring
  experimentally-labeled over predicted when a sequence repeats), and
  random-sample each pool down to a cap. Pure Python stdlib (re, random), no
  dependencies, no setup. Use to bootstrap the initial binder pool before folding
  / redesign / evaluation. This is a BOOTSTRAP HELPER, not a dispatchable design
  tool.
---

# Seed binders — campaign bootstrap pool (Python stdlib)

**This is a bootstrap helper, not a dispatchable design tool.** It does not design
or optimize anything — it provides the *starter* set of candidate binder sequences
at the beginning of a campaign, before any folding / redesign / evaluation runs.
It was originally a supervisor helper (not a BaseTool), and its whole job is to
turn raw peptipedia-style FASTA(s) into a clean, canonical, sanity-filtered,
size-capped pool of sequences to seed the binder pool with.

This is a **self-contained, instruction-only skill**. There is no project
dependency — you (the agent) run a few lines of the Python standard library
directly. The procedure below is everything you need.

## When to use it

- At **campaign start**, to assemble an initial, sanity-filtered set of candidate
  binder sequences.
- When you have one or more FASTA files of candidate peptides/binders and need a
  clean canonical pool to hand to the next steps.

Do **not** reach for this mid-campaign as a way to generate new designs — it only
cleans and samples an existing pool. Downstream, peptide-descriptors profiles the
seeds and folding folds them.

## Setup

None. Pure Python standard library (`re`, `random`) — nothing to install, no gate
needed.

## Procedure

Parse each FASTA record, keep only canonical sequences, length-filter, dedupe
(preferring experimentally-labeled over predicted), and random-sample down to a
per-pool cap. Details that make the pool clean:

- **Canonical residues only** — reject any sequence containing a character outside
  the 20 standard amino acids via the regex `[^ACDEFGHIKLMNPQRSTVWY]`.
- **Length filter** — keep `min_len <= len(seq) <= max_len`.
- **Dedupe, preferring experimental labels** — when the same sequence appears more
  than once, keep the experimentally-labeled copy over a predicted one (inspect
  the FASTA header for the label).
- **Cap per pool** — random-sample down to `sample_cap`; set a `seed` for
  reproducibility.
- You may also seed the pool with a **small hardcoded set of known-good starter
  binders** if the campaign has them.

```python
import re, random

# --- edit these ---
FASTA_PATHS = ["/path/to/peptipedia.fasta"]  # one or more peptipedia-style FASTAs
MIN_LEN, MAX_LEN = 5, 50
SAMPLE_CAP = 500
SEED = 0
KNOWN_GOOD = []                              # optional hardcoded starter binders
# ------------------

NONCANONICAL = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

def parse_fasta(path):
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line.strip().upper())
    if header is not None:
        yield header, "".join(chunks)

def is_experimental(header):
    h = header.lower()
    return ("experiment" in h) or ("verified" in h)   # prefer labeled-experimental

# seq -> (is_experimental?) ; dedupe preferring experimental over predicted
best = {}
for path in FASTA_PATHS:
    for header, seq in parse_fasta(path):
        if not seq or NONCANONICAL.search(seq):        # canonical residues only
            continue
        if not (MIN_LEN <= len(seq) <= MAX_LEN):        # length filter
            continue
        exp = is_experimental(header)
        if seq not in best or (exp and not best[seq]):  # keep experimental copy
            best[seq] = exp

pool = list(best.keys())
random.Random(SEED).shuffle(pool)
pool = pool[:SAMPLE_CAP]                                # cap per pool

seeds = list(dict.fromkeys(KNOWN_GOOD + pool))         # prepend known-good, dedupe
print(len(seeds), "seed binders")
```

## Interpreting the result

- `seeds` is a list of clean, canonical, deduped, length-filtered sequences ready
  to initialize the campaign's binder pool.
- A tiny pool after filtering usually means the length bounds or the canonical
  filter were too strict for the source FASTA — loosen `MIN_LEN`/`MAX_LEN` or
  check the file for modified/non-standard residues before concluding the source
  is empty.

## Notes

- Inputs you supply: FASTA path(s), length bounds (`min_len`/`max_len`), the
  per-pool `sample_cap`, and an optional random `seed`.
- This only assembles the starter pool. **What to do next with the seeds —
  profile them with peptide-descriptors, fold them, or feed them to redesign — is
  your job as the agent.** This skill is the campaign's first step, not a design
  step.
