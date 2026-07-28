---
name: interactant-folding
description: >-
  Co-fold a campaign target with a KNOWN natural interacting protein (an
  interactant), rather than a designed binder, using the same Chai-1 pipeline as
  the folding skill. Produces a reference complex structure plus interface
  metrics (ipSAE headline; ipTM, pDockQ, pDockQ2, LIS) you can feed to
  contact-eval to validate target hotspots against a real binding partner. Use
  when you have a target sequence and a natural partner's sequence. GPU-heavy;
  needs the same pre-provisioned Chai-1 environment as the folding skill.
---

# Co-folding target + natural interactant (chai_lab / Chai-1)

This is the **folding** skill applied to a natural binding partner instead of a
designed binder. The algorithm, model weights, parameters, output, scoring, and
headline confidence scalar (**ipSAE**, falling back to ipTM then pDockQ) are
**identical**. The only difference is what goes in the second protein chain:
here it is a **known interactant sequence**, so the resulting complex is a
*reference* structure for validating the target's hotspots against a real
partner.

This is a **self-contained, instruction-only skill** — no wrapper script and no
`bindify` dependency.

## When to use it

- You have the target sequence and the sequence of a **natural interacting
  protein** and want a reference complex + structure-validated interface.
- You want to confirm literature/inferred target hotspots by seeing which target
  residues a real partner actually contacts (feed the output to **contact-eval**).

If the second chain is a *designed* binder, use the **folding** skill instead.

## How it differs from `folding`

**It doesn't, except for one input.** Follow the entire **folding** skill
procedure verbatim — setup, gated import check, FASTA construction, optional
`constraints.txt`, `run_inference(..., use_esm_embeddings=True,
num_diffn_timesteps=100, full_data=True)`, the 5-model output, ipSAE scoring,
best-model selection by mean ipSAE, and CIF→PDB with gemmi.

The single substitution:

```
binder_sequence  :=  interactant_sequence
```

So in the FASTA, the last protein record is the interactant rather than a
designed binder (keep it **last** — contact-eval / inverse-folding conventions
still expect the partner chain last):

```
>protein|target
<TARGET_SEQUENCE>
>protein|interactant
<INTERACTANT_SEQUENCE>
```

Obtain `<INTERACTANT_SEQUENCE>` via the **interactant-sequence** skill.

## Setup, procedure, interpretation, cost

See the **folding** skill — all identical. It requires the same pre-provisioned
GPU environment (msinclair-py `chai_lab` fork, `molecular_simulations`, `gemmi`,
local Chai-1/ESM weights, `/dev/shm`) and the same gated import check.

## Notes

- Deciding which target/interactant pairs to fold and how to use the validated
  interface is your job as the agent.
- Cross-references: **folding** (the full procedure this skill defers to),
  **interactant-sequence** (fetch the partner sequence), **contact-eval**
  (validate hotspots on the resulting complex).
