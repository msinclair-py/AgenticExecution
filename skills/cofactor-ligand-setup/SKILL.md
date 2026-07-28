---
name: cofactor-ligand-setup
description: >-
  Build a Chai-1 folding input that includes cofactors, metal ions, and small-
  molecule ligands (e.g. Fe²⁺ + 2-oxoglutarate for an Fe/2OG dioxygenase, Zn²⁺,
  heme, NAD, a substrate analog) alongside the protein chains, so a binder-target
  complex is folded HOLO with physiological pocket geometry. Use whenever the
  target is a metalloenzyme / cofactor-dependent protein and you want the fold to
  reflect the real active site. Needs only text (writes a Chai FASTA); the fold
  itself is the `folding` skill.
---

# Cofactor / ligand setup for holo folding (Chai-1)

Chai-1 accepts a multi-entity FASTA where each record header declares an entity
type. Beyond `>protein|`, you can add **ligands** (small molecules, metal ions,
cofactors) and **glycans**, so the complex folds with its cofactors present —
essential for metalloenzymes whose pocket only makes sense holo.

This is a **self-contained, instruction-only skill** — you write the FASTA and
pass it to the `folding` skill's `run_inference`. No new dependency.

## When to use it

- The target is a **metalloenzyme or cofactor-dependent** protein (Fe/2OG
  dioxygenase, Zn hydrolase, heme protein, kinase+ATP/Mg, NAD(P) enzyme…) and you
  want binder–target complexes folded with the metal/cofactor in place.
- You want a **substrate analog** in the pocket to shape a substrate-groove binder.

## Entity types in a Chai FASTA

```
>protein|target
<TARGET_SEQUENCE>
>protein|binder
<BINDER_SEQUENCE>
>ligand|<name>
<SMILES>
```

- **Metal ions** are ligands given as a SMILES for the charged atom:
  `[Fe+2]`, `[Zn+2]`, `[Mg+2]`, `[Mn+2]`, `[Ca+2]`, `[Ni+2]`.
- **Cofactors / substrates** are ligands given by SMILES (or a CCD code where the
  build supports it). Examples:
  - 2-oxoglutarate (α-ketoglutarate): `O=C(O)CCC(=O)C(=O)O`
  - succinate (product): `O=C(O)CCC(=O)O`
  - ATP, NAD, heme, SAM, etc. — use the canonical SMILES.
- One `>ligand|` record per molecule; repeat for multiple copies.

## Procedure

Assemble the holo FASTA (target + binder + cofactors), then fold with the
`folding` skill. Example for an **Fe(II)/2-oxoglutarate dioxygenase** (the ALKBH1
case — Fe²⁺ + αKG in the catalytic cleft):

```python
def write_holo_fasta(path, target_seq, binder_seq,
                     metal_smiles="[Fe+2]", cofactor_smiles="O=C(O)CCC(=O)C(=O)O",
                     cofactor_name="2OG"):
    with open(path, "w") as f:
        f.write(f">protein|target\n{target_seq}\n")
        f.write(f">protein|binder\n{binder_seq}\n")
        f.write(f">ligand|metal\n{metal_smiles}\n")
        f.write(f">ligand|{cofactor_name}\n{cofactor_smiles}\n")
    return path
# then: run_inference(fasta_file=path, output_dir=<empty dir>, device=..., use_esm_embeddings=True)
```

To **steer the metal/cofactor to the real site**, add distance/pocket constraints
via Chai's `constraint_path` CSV (contact between the metal and the coordinating
residues — e.g. the His/Asp/His triad — and between the cofactor and its anchoring
residues). Constraint columns: `chainA,res_idxA,chainB,res_idxB,connection_type,
confidence,min_distance,max_distance,comment,restraint_id`.

## Interpreting the result

- The output complex now contains the protein chains + ligand entities; the
  binder is folded against a **holo** pocket, so interface metrics and
  `contact-eval` reflect a binder that must accommodate the occupied active site
  (realistic for an inhibitor that caps the cleft).
- If the metal/cofactor drifts from the catalytic residues, add or tighten the
  constraints, or seed from an experimental holo structure's coordinates.

## Notes

- Get the cofactor identity + coordinating residues from the `uniprot` skill
  (COFACTOR / Binding-site features) or an experimental holo PDB.
- Deciding which cofactors/ions/substrate to include, and their coordinating
  residues, is your job as the agent — this skill assembles the fold input.
- Downstream: `folding` (run the holo fold), then `contact-eval` (does the binder
  hit the catalytic cleft?) and `md-simulation`.
