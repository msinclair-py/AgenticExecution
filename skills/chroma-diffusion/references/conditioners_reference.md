# Chroma conditioners & sampling reference

## Protein tensors (`Protein.to_XCS()`)

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `X` | `[1, L, 4, 3]` | backbone coords (N, CA, C, O per residue) |
| `C` | `[1, L]` | chain id per residue — **1 = receptor, 2 = binder** (negative = masked) |
| `S` | `[1, L]` | amino-acid index per residue (index into `constants.AA20`) |

Extend all three by `len_binder` to add the binder segment (zeros for `X`, `2`
for `C`, motif indices or `0` for `S`), then rebuild `Protein(X, C, S)`.

## Design mask (`design_selection`)

Shape `[1, L, 20]`. Row = 20-way amino-acid allowance per residue. Set binder
rows to all-ones (free) and leave receptor rows zero (locked). For a grafted
motif, also zero the motif rows so those residues stay fixed.

## Conditioners

- **`SubstructureConditioner(protein, backbone_model, selection)`** — holds the
  substructure named by `selection` (e.g. `'namesel receptor'`) rigid, so
  diffusion only remodels the rest. `backbone_model=model.backbone_network`.
- **`InflateConditioner(vector, magnitude)`** — translates the diffusing segment
  along `vector` (a `torch.FloatTensor`) by `magnitude`. Used to aim the binder
  at a hotspot: `vector = hotspot_COM − target_COM`, `magnitude = |vector| ×
  vector_scaling`.
- **`SubsequenceConditioner(design_model, protein, selection, n_backbones)`** —
  biases the binder sequence toward a fixed motif at `selection`
  (`'namesel bindermask'`). `design_model=model.design_network`.
- **`ComposedConditioner([...])`** — combine conditioners; order in the code is
  `[displacement, substructure]` (+ subsequence when grafting).

Save named selections on the protein before building conditioners:
`protein.sys.save_selection(gti=<residue indices>, selname='receptor')`.

## `model.sample(...)` arguments

| Arg | Typical | Meaning |
|-----|---------|---------|
| `protein_init` | the extended `Protein` | starting complex |
| `conditioner` | `ComposedConditioner` | the composed constraints |
| `design_selection` | `mask_aa` | which residues are designed |
| `steps` | 200–500 | reverse-diffusion steps |
| `samples` | `n_backbones` | independent backbones per call |
| `num_designs` | `n_designs` | sequence designs per backbone |
| `sde_func` | `'langevin'` | integrator |
| `langevin_factor` | 2 | Langevin dynamics factor |
| `langevin_isothermal` | True | isothermal Langevin |
| `inverse_temperature` | 8.0 | higher = colder = more conservative geometry |
| `full_output` | True | also return trajectories |

Returns `(proteins, trajectories)`. `proteins` may be a list or nested list
(`samples × num_designs`); flatten before saving. Index mapping in the reference:
`backbone = i // num_designs`, `design = i % num_designs`.

## Saving

`protein.to('out.cif')` writes CIF. Convert to PDB for downstream ProteinMPNN /
Chai: `Bio.PDB.MMCIFParser` → `PDBIO`, or `gemmi.read_structure(cif).write_pdb`.
