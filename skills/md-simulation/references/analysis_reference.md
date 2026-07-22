# Trajectory / structure analysis reference

Beyond RMSD (§4 of the skill) and MM-PBSA (§5), `molecular_simulations.analysis`
provides several structure/trajectory analyses. Signatures below; the
`molecular-simulations` repo's `analyze-*` skills cover them in depth.

## ipSAE — interface confidence of a predicted complex

```python
from molecular_simulations.analysis import ipSAE
scorer = ipSAE(
    structure_file='model.pdb',   # or .cif
    plddt_file='scores.npz',      # npz with pLDDT array
    pae_file='scores.npz',        # npz with PAE (NxN) array
    out_path='scores/',           # optional; default = plddt_file parent
    skip_chains={'C'},            # optional: drop ligand/glycan chains
)
scorer.run()
scorer.scores                     # Polars DF: ipSAE, ipTM, pDockQ, pDockQ2, LIS per chain pair
# also writes ipSAE_scores.parquet
```
This is the same scorer the chai-folding skill uses to rank models. pLDDT/PAE
token counts must match the structure's residues (after `skip_chains`).

## RMSD (binder stability gate)

`MDAnalysis.analysis.rms.RMSD(solute, select='backbone and resid <target>',
groupselections=[binder_sel])` — align on target backbone, read binder RMSD from
the last column. See `scripts/rmsd_stability.py`.

## SASA — solvent-accessible surface area

`molecular_simulations.analysis.sasa` computes absolute SASA and `RelativeSASA`
(fractional exposure) per residue over a trajectory — useful for interface
burial. See the `analyze-sasa` skill.

## Interaction energy / fingerprints

`molecular_simulations.analysis.interaction_energy` and `.fingerprinter` provide
per-residue interaction-energy decomposition, linear interaction energy
(static/dynamic), and KMeans clustering of per-frame features. See the
`analyze-interactions` skill.

## Contact residues (quick, no trajectory)

For a single structure, residues on one chain within a cutoff of another
(interface epitope) via MDAnalysis `distance_array` — see the `bindify` material
`contact_analysis.compute_contact_residues(structure, target_chain, partner_chain,
cutoff_angstroms=5.0)`, returning labels like `['R47', 'K112']`.
