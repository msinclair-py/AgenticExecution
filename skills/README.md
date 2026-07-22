# Skills

On-demand skills for the AgenticExecution code-execution harness. Each
subdirectory holds a `SKILL.md` (name + description frontmatter the agent loads
when a task matches), plus `references/` (deep-dive docs loaded as needed) and
`scripts/` (runnable, importable templates). Skills teach the **underlying
libraries** so an LLM can author `execute_code` snippets against the project
environment — device-agnostic (`resolve_device()` → xpu/cuda/cpu) throughout.

## Biomolecular design skills

| Skill | Use it for | Underlying lib |
|-------|-----------|----------------|
| [chroma-diffusion](chroma-diffusion/SKILL.md) | De novo binder **backbone** design against a target hotspot | `chroma` |
| [proteinmpnn](proteinmpnn/SKILL.md) | **Inverse folding** — design sequences for a backbone | `protein_mpnn_utils` |
| [protein-language-model-design](protein-language-model-design/SKILL.md) | **pLM sequence design** — infill / extend / rank sequences (no structure) | `transformers` (ESM-2) |
| [chai-folding](chai-folding/SKILL.md) | **Fold** sequence → structure (mono/complex/ligand/glycan), ipSAE scoring | `chai_lab` |
| [md-simulation](md-simulation/SKILL.md) | **MD + MM-PBSA** — stability & binding free energy | `molecular_simulations` |
| [rag](rag/SKILL.md) | **Literature RAG** — mine hotspots / PTMs / known binders | `distllm` + `faiss` |

### Typical binder-design pipeline

```
rag (mine target: hotspots, PTMs, known binders)
        │
        ▼
chroma-diffusion  ──backbones──►  proteinmpnn  ──sequences──►  chai-folding
   (design bb at hotspot)          (inverse fold)              (fold + ipSAE)
                                                                    │
                                                              best PDB │
                                                                    ▼
                                                            md-simulation
                                                        (RMSD gate → MM-PBSA ΔG)
```

Each stage's output feeds the next: Chroma backbones → ProteinMPNN sequences →
Chai co-folds + ipSAE-scores them → MD checks the binder stays bound and
estimates ΔG_binding. RAG runs upstream to seed the campaign (epitope residues,
glycosylation sites, existing binders).

`protein-language-model-design` is a second, **structure-free** sequence-design
source that sits alongside `proteinmpnn`: given an existing hit it infills or
extends the sequence with ESM-2 and can rank any candidate pool by
pseudo-log-likelihood — a fast way to diversify or triage sequences before
folding them with `chai-folding`.

## Other skills

- [pyscf](pyscf/SKILL.md) — quantum chemistry (DFT/SCF, ESP/RESP, torsion scans).
- [rdkit](rdkit/SKILL.md) — cheminformatics (I/O, SMARTS, conformers, reactions).
- [deploy-polaris](deploy-polaris/SKILL.md) — scale the code-execution MCP server
  across an ALCF Polaris PBS allocation with Parsl (NVIDIA A100, 4 GPUs/node).
- [deploy-aurora](deploy-aurora/SKILL.md) — the same on ALCF Aurora (Intel Max,
  12 tiles/node, `frameworks` module, PBS Pro).
