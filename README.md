# AgenticExecution

A package of **skills** that connects an HPC execution layer to an LLM harness.

The execution layer is [Parsl](https://parsl-project.org) (fans work across every
GPU on every node of an allocation), [academy-py](https://github.com/proxystore/academy)
(long-lived, stateful agents that keep models resident in GPU memory across
calls), and [FastMCP](https://github.com/jlowin/fastmcp) (exposes that compute as
MCP tools). The `globus-compute-mcp` skill is the bridge: an MCP server runs
locally next to the harness and dispatches to Globus Compute endpoints inside the
facility — no SSH per call, no tunnel, no inbound port, no MFA prompt.

On top of that bridge sits a library of **scientific skills** for probing biology
and biophysics — folding, diffusion-based design, inverse folding, protein
language models, MD + free energy, QM, cheminformatics, literature RAG — each a
self-contained procedure the harness reads and runs.

```
   LLM harness (Claude Code / Codex / any MCP client)
        │
        │  ── reads SKILL.md, authors the code ──
        ▼
   MCP surface  ── FastMCP ──►  academy agent  ──►  Parsl HTEX
   (local, or in-allocation)     (stateful,          (all GPUs,
        │                         model resident)     all nodes)
        │
        └── globus-compute-mcp: local server → Globus Compute endpoint → PBS
```

## Repository layout

Everything lives under `skills/`. Each skill is a directory with a `SKILL.md`
(frontmatter + procedure) and optional `scripts/` and `references/`.

- `skills/README.md` — the full skill catalog and how they chain into a campaign.
- `skills/ENVIRONMENTS.md` — how to pre-provision the heavy GPU environments on a
  cluster (login-node installs, shared weight caches, CUDA/torch matching,
  offline compute nodes). **Read this before running anything GPU-heavy.**
- `skills/references/site-profiles.md` — per-machine facts (nodes, queues,
  pinning vars, filesystems) for the HPC execution skills.

## Skills are instruction-only

A skill carries the *know-how*, not a wrapper: what it does, when to use it,
gated setup (check-before-install), the real procedure and parameters over a
**base library** (raw `requests`, MDAnalysis, `peptides`, ProteinMPNN, chai_lab,
Chroma, OpenMM, PySCF, RDKit, faiss, …), and how to read the result. None of them
import this package. Any harness that can read a `SKILL.md` and run code can use
them; a harness with a code sandbox and the base library installed needs nothing
else. Deciding *which* inputs to run on and *what* to do with the outputs is the
agent's orchestration job, not the skill's.

## Connecting the execution layer

| Skill | Base | Role |
|---|---|---|
| `globus-compute-mcp` | globus-compute + academy + fastmcp | **The bridge.** FastMCP runs locally, dispatching to Globus Compute endpoints that dial outbound and acquire PBS allocations on demand |
| `polaris-jobs` | ssh + PBS | Operational fallback: one persistent MFA'd SSH connection, `qsub`/`qstat`, and the `mpiexec` affinity lines jobs get wrong |

Two ways to reach the compute, in increasing decoupling:

1. **`polaris-jobs`** — *operational*. One multiplexed SSH connection
   (ControlMaster, authenticated once with an MFA token because ALCF requires a
   passcode per login), then drive `qsub`/`qstat` and file staging over it
   non-interactively.
2. **`globus-compute-mcp`** — *federated*. The MCP server is local and endpoints
   dial outbound, so nothing has to be reachable inside the facility network, the
   model stays loaded across calls, and the endpoint survives a job hitting
   walltime.

## Scientific skills

Narrow, single-step skills — the units an orchestrating agent chains into a
campaign.

**Knowledge / lookup** (remote HTTP, light)

| Skill | Base | Purpose |
|---|---|---|
| `uniprot` | `requests` → rest.uniprot.org | Target sequence / metadata / known interactors |
| `interactant-sequence` | `requests` → rest.uniprot.org | Accession → sequence for interacting partners |
| `fetch-literature` | `requests` → PubMed / Europe PMC | Gather a literature corpus about a target |

**Retrieval-augmented memory**

| Skill | Base | Purpose |
|---|---|---|
| `rag-build-index` | faiss + datasets + MiniLM | Build a searchable vector store over a corpus |
| `rag-query` | faiss + MiniLM + LLM endpoint | Retrieve + answer / structured extraction over a store |
| `experiments-index` | faiss + datasets (fcntl append) | Incrementally index a campaign's own results |

**Analysis / triage** (CPU, light)

| Skill | Base | Purpose |
|---|---|---|
| `contact-eval` | MDAnalysis | Interface residues on a target from a folded complex |
| `peptide-descriptors` | `peptides` | Physicochemical descriptors of a peptide/binder |
| `developability` | sequence scan + structure check | Liabilities, charge/pI, aggregation, immunogenicity flags |
| `filter-binders` | none (pure logic) | Bucket a pool into priority tiers with a survivor guard |

**Generation / structure** (GPU, pre-provisioned env)

| Skill | Base | Purpose |
|---|---|---|
| `folding` | chai_lab + ipSAE | Co-fold a target+binder complex, interface confidence |
| `interactant-folding` | chai_lab | Co-fold a target with a natural partner |
| `cofactor-ligand-setup` | text only (writes a Chai FASTA) | Add metals / cofactors / ligands so the fold is HOLO |
| `inverse-folding` | ProteinMPNN | Redesign sequences for a fixed backbone/target |
| `peptide-diffusion` | Chroma | De novo binder backbones conditioned on a hotspot |
| `peptide-clamp-design` | Chroma (isolated-peptide trick) | Binders that clamp a linear/disordered epitope |
| `protein-language-explore` | transformers / ESM-2 650M | Extend or infill a sequence via a protein LM |

**Ensembles & simulation** (GPU, heavy)

| Skill | Base | Purpose |
|---|---|---|
| `bioemu-ensemble` | BioEmu | Equilibrium-like conformational ensemble — fast ML alternative to MD |
| `md-simulation` | OpenMM + AmberTools + MDAnalysis | Stability (RMSD) + binding free energy of a complex |

**Screening / bootstrap**

| Skill | Base | Purpose |
|---|---|---|
| `seed-binders` | stdlib | Sanity-filtered starter binder pool (a helper, not a dispatch tool) |
| `peptide-repurposing-screen` | chai_lab at scale | Screen a known-peptide database against a target; benchmark de novo designs |

**Working with the user**

| Skill | Base | Purpose |
|---|---|---|
| `supervisor` | none (decision lens) | Pressure-test a decision against PI-level scientific standards |
| `ntfy-notify` | ntfy HTTP | Two-way phone messaging when the user is away from the terminal |

### Reference skills (broad, library-level)

Wider-scope skills covering a whole library rather than one campaign step. Reach
for these when you need a capability the narrow skill doesn't expose — an unusual
folding condition, a different ProteinMPNN weight set, a QM calculation.

| Skill | Base | Scope |
|---|---|---|
| `chai-folding` | chai_lab | Superset of `folding` — multimers, ligands/SMILES, glycans/PTMs, restraint files |
| `proteinmpnn` | ProteinMPNN | Superset of `inverse-folding` — in-process + CLI, fixed positions, soluble weights |
| `chroma-diffusion` | Chroma | Superset of `peptide-diffusion` — Substructure / Inflate / Subsequence conditioners |
| `protein-language-model-design` | ESM-2 | Superset of `protein-language-explore` — infill, extension, pLL ranking |
| `rag` | distllm + faiss | Spans `rag-build-index` + `rag-query`, incl. generator-backend choice |
| `pyscf` | PySCF / gpu4pyscf | DFT/SCF, geometry optimization, ESP grids for RESP charges, implicit solvent |
| `rdkit` | RDKit | Molecular I/O, SMARTS search, 2D/3D coords, conformers, reactions |

## A typical campaign flow (agent-orchestrated)

```
uniprot → (fetch-literature → rag-build-index → rag-query) to find hotspots
        → interactant-sequence → interactant-folding → contact-eval  (validate hotspots)
seed-binders / peptide-diffusion / peptide-clamp-design
        → inverse-folding → folding (+ cofactor-ligand-setup for holo targets)
        → contact-eval + peptide-descriptors + developability → filter-binders
        → md-simulation on survivors → filter-binders
        → experiments-index (remember everything)
```

The agent drives the loop; the skills are the steps; the execution layer is what
makes the expensive steps land on a supercomputer instead of a laptop.
`supervisor` is consulted at the decision points between them, and
`peptide-repurposing-screen` supplies the baseline the designs are judged against.

## Installing the skills

Skills are read from the harness's skill directory. Symlink the ones you want:

```bash
ln -s "$PWD/skills/folding" ~/.claude/skills/folding

# ...or install the whole set
for s in skills/*/; do
  [ "$(basename "$s")" = references ] && continue
  ln -s "$PWD/$s" "$HOME/.claude/skills/$(basename "$s")"
done
```

## Notes and caveats

- The heavy GPU/sim skills (`folding`, `peptide-diffusion`, `md-simulation`,
  `inverse-folding`, `protein-language-explore`, `bioemu-ensemble`) require
  pre-provisioned environments with model weights and env vars
  (`$PROTEINMPNN_PATH`, `$AMBERHOME`, `$CHAI_DOWNLOADS_DIR`, `$HF_HOME`). Their
  setup sections say so honestly rather than pretending an inline `pip install`
  suffices — see `skills/ENVIRONMENTS.md`.
- On most clusters only login nodes reach the internet. Build envs and
  pre-download weights on a login node into shared project storage, then set
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` in the job.
- Where a campaign skill and a reference skill overlap, prefer the campaign skill
  for the standard step and drop to the reference skill when you need a knob it
  doesn't expose.

## License

Apache-2.0 — see [LICENSE](LICENSE).
