# Skills

Portable, **instruction-only** skills for running scientific campaigns on HPC.
Each skill is a self-contained procedure over a **base library** (raw `requests`,
MDAnalysis, `peptides`, ProteinMPNN, chai_lab, Chroma, OpenMM, PySCF, RDKit,
faiss, …) — **none require this package**. Any agentic
harness that can read a `SKILL.md` and run code (Claude Code, Codex, Hermes, …)
can use them; a harness with a code sandbox and the base library installed needs
nothing else.

- `ENVIRONMENTS.md` — how to pre-provision the heavy GPU environments on a
  cluster (login-node installs, shared weight caches, CUDA/torch matching,
  offline compute nodes). Read this before running anything GPU-heavy.
- `references/site-profiles.md` — per-machine facts (nodes, queues, pinning
  vars, filesystems) for the HPC execution skills.

## What "instruction-only" means here

A skill carries the *know-how*, not a wrapper: what it does, when to use it,
gated setup (check-before-install), the real procedure/parameters over the base
library, and how to read the result. The agent writes and runs the few lines
itself. Deciding which inputs to run on and what to do with the outputs is the
agent's orchestration job, not the skill's.

## Campaign skills

Narrow, single-step skills — the units an orchestrating agent chains into a
binder-design campaign.

### Knowledge / lookup (remote HTTP, light)
| Skill | Base | Purpose |
|---|---|---|
| `uniprot` | `requests` → rest.uniprot.org | Fetch a target's sequence / metadata / known interactors |
| `interactant-sequence` | `requests` → rest.uniprot.org | Accession → amino-acid sequence for interacting partners |
| `fetch-literature` | `requests` → PubMed / Europe PMC | Gather a literature corpus about a target |

### Retrieval-augmented memory
| Skill | Base | Purpose |
|---|---|---|
| `rag-build-index` | faiss + datasets + MiniLM | Build a searchable vector store over a corpus |
| `rag-query` | faiss + MiniLM + LLM endpoint | Retrieve + LLM-answer / structured extraction over a store |
| `experiments-index` | faiss + datasets (fcntl append) | Incrementally index a campaign's own results |

### Analysis / triage (CPU, light)
| Skill | Base | Purpose |
|---|---|---|
| `contact-eval` | MDAnalysis | Interface residues on a target from a folded complex |
| `peptide-descriptors` | `peptides` | Physicochemical descriptors of a peptide/binder |
| `developability` | sequence scan + structure check | Is a binder a realistic lab candidate — liabilities, charge/pI, aggregation, immunogenicity flags |
| `filter-binders` | none (pure logic) | Bucket a binder pool into priority tiers with a survivor guard |

### Generation / structure (GPU, pre-provisioned env)
| Skill | Base | Purpose |
|---|---|---|
| `folding` | chai_lab + ipSAE | Co-fold a target+binder complex, interface confidence |
| `interactant-folding` | chai_lab (defers to `folding`) | Co-fold target with a natural partner |
| `cofactor-ligand-setup` | text only (writes a Chai FASTA) | Add metals / cofactors / ligands so the fold is HOLO |
| `inverse-folding` | ProteinMPNN (`$PROTEINMPNN_PATH`) | Redesign sequences for a fixed backbone/target |
| `peptide-diffusion` | Chroma | De novo binder backbones conditioned on a hotspot |
| `peptide-clamp-design` | Chroma (isolated-peptide trick) | Binders that clamp a linear/disordered epitope |
| `protein-language-explore` | transformers / ESM-2 650M | Extend/infill a sequence via a protein LM |

### Ensembles & simulation (GPU, heavy)
| Skill | Base | Purpose |
|---|---|---|
| `bioemu-ensemble` | BioEmu | Equilibrium-like conformational ensemble — fast ML alternative to MD |
| `md-simulation` | OpenMM + AmberTools + MDAnalysis | Stability (RMSD) + binding free energy of a complex |

### Screening / bootstrap helpers
| Skill | Base | Purpose |
|---|---|---|
| `seed-binders` | stdlib | Assemble a sanity-filtered starter binder pool (not a dispatch tool) |
| `peptide-repurposing-screen` | chai_lab at scale | Screen a known-peptide database against a target; benchmark de novo designs |

### Working with the user
| Skill | Base | Purpose |
|---|---|---|
| `supervisor` | none (decision lens) | Pressure-test a decision against PI-level scientific standards before committing |
| `ntfy-notify` | ntfy HTTP | Two-way phone messaging when the user is away from the terminal |

## Reference skills (broad, library-level)

Wider-scope skills covering a whole library rather than one campaign step. Use
these when you need a capability the narrow skill doesn't cover — an unusual
folding condition, a different ProteinMPNN weight set, a QM calculation.

| Skill | Base | Purpose |
|---|---|---|
| `chai-folding` | chai_lab | Full range of Chai-1 folding — multimers, ligands/SMILES, glycans/PTMs, restraint files; superset of `folding` |
| `proteinmpnn` | ProteinMPNN | In-process + CLI paths, fixed positions, noise-level/soluble weight variants; superset of `inverse-folding` |
| `chroma-diffusion` | Chroma | Conditioner composition (Substructure / Inflate / Subsequence); superset of `peptide-diffusion` |
| `protein-language-model-design` | ESM-2 | Infill, terminal extension, and pseudo-log-likelihood ranking; superset of `protein-language-explore` |
| `rag` | distllm + faiss | End-to-end corpus → index → query, incl. generator-backend choice; spans `rag-build-index` + `rag-query` |
| `pyscf` | PySCF / gpu4pyscf | DFT/SCF, geometry optimization, ESP grids for RESP charges, implicit solvent |
| `rdkit` | RDKit | Molecular I/O, SMARTS substructure search, 2D/3D coords, conformers, reactions |

## HPC execution

| Skill | Base | Purpose |
|---|---|---|
| `polaris-jobs` | ssh + PBS | Submit/monitor A100 jobs on ALCF Polaris over one persistent MFA'd SSH connection; correct mpiexec + `set_affinity_gpu_polaris.sh` GPU binding |
| `globus-compute-mcp` | globus-compute + academy + fastmcp | FastMCP runs *locally*, dispatching to Globus Compute endpoints — no SSH per call, no tunnel, no inbound port, no MFA prompt |
| `hpc-deployment` (repo root, not `skills/`) | the same, plus optional `ensemble_launcher` | Campaign-grade deployment of that bridge: `HPC_EXECUTION=direct` gives one agent per pinned GPU/tile, `cluster` hands placement to an Ensemble Launcher master and adds `run_batch` over a JSONL of inputs, a frozen batch registry, a versioned tool library, and a self-healing pool |

Three ways to reach the compute, in increasing decoupling:

1. **`polaris-jobs`** — *operational*. Open a single multiplexed SSH connection
   (ControlMaster, authenticated once with an MFA token because ALCF requires a
   one-time passcode per login), then drive `qsub`/`qstat` and file staging over
   it non-interactively. It documents the machine's node hardware, queues, and
   — the part jobs get wrong — the exact `mpiexec` affinity lines.
2. **`globus-compute-mcp`** — *federated*. The MCP server is local and endpoints
   dial outbound to the Globus service, so nothing has to be reachable inside the
   facility network, the model stays loaded across calls, and the endpoint
   survives a job hitting walltime.
3. **`hpc-deployment`** — *federated, campaign-scoped*. The same outbound
   dispatch with the state a long campaign needs: a campaign ID that isolates
   agent adoption, a pool that probes and repairs itself, and batches frozen on
   disk so a crashed run resumes rather than restarting. Its two styles differ in
   who places work on GPUs — the endpoint (one worker per pin) in `direct` mode,
   or an Ensemble Launcher master spanning every tile of the allocation in
   `cluster` mode, which is also what makes `run_batch` possible. Only `cluster`
   needs `ensemble_launcher` installed on the worker.

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

The agent drives this loop; the skills are the steps. `supervisor` is consulted
at the decision points between them, and `peptide-repurposing-screen` supplies
the baseline the designs are judged against.

## Notes and caveats

- The heavy GPU/sim skills (`folding`, `peptide-diffusion`, `md-simulation`,
  `inverse-folding`, `protein-language-explore`, `bioemu-ensemble`) require
  pre-provisioned environments with model weights and env vars
  (`$PROTEINMPNN_PATH`, `$AMBERHOME`, `$CHAI_DOWNLOADS_DIR`, `$HF_HOME`); their
  setup sections say so honestly rather than pretending an inline `pip install`
  suffices. See `ENVIRONMENTS.md`.
- `seed-binders` is a bootstrap helper, not a dispatchable tool — flagged as such
  in its own frontmatter.
- `experiments-index` genuinely wraps heavy vector-store I/O (faiss + datasets);
  the skill documents the append procedure but it is not "lightweight."
- Where a campaign skill and a reference skill overlap, prefer the campaign skill
  for the standard step and drop to the reference skill when you need a knob it
  doesn't expose.
