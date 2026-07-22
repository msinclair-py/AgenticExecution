---
name: rdkit
description: Cheminformatics toolkit for fine-grained molecular control. Emphasis on molecular I/O & parsing (SMILES/SDF/MOL/InChI, sanitization, batch suppliers), substructure search & SMARTS queries, and 2D/3D coordinate generation, conformers & reaction transforms. Also covers descriptors, fingerprints, similarity, and drawing. Use rdkit for advanced control, custom sanitization, and specialized algorithms; for simpler standard workflows consider datamol (a wrapper around RDKit).
license: BSD-3-Clause license
metadata:
    skill-author: adapted from K-Dense Inc. scientific-skills
    adapted-by: local project skill (helicon/agentic)
---

# RDKit Cheminformatics Toolkit

## Overview

RDKit is a comprehensive cheminformatics library providing Python APIs for molecular analysis and manipulation. This skill emphasizes three areas:

1. **Molecular I/O & parsing** — reading/writing SMILES, SDF, MOL, InChI; sanitization & validation; batch processing.
2. **Substructure search & SMARTS** — pattern matching, query construction, library filtering, highlighting.
3. **2D/3D coordinates & reactions** — depiction coordinates, 3D embedding & conformers, force-field optimization, reaction transforms.

Descriptors, fingerprints/similarity, and other capabilities are covered more briefly toward the end and in the bundled references. Use this skill for drug discovery, computational chemistry, and cheminformatics research tasks.

Every `Chem.MolFrom*` parser returns `None` on failure (printing an error to stderr). **Always check for `None` before using a molecule** — this is the single most common source of bugs.

---

## 1. Molecular I/O and Parsing

### Reading molecules

```python
from rdkit import Chem

mol = Chem.MolFromSmiles('Cc1ccccc1')          # SMILES -> Mol or None
mol = Chem.MolFromMolFile('path/to/file.mol')  # MOL file
mol = Chem.MolFromMolBlock(mol_block_string)   # MOL block (string)
mol = Chem.MolFromInchi('InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')
mol = Chem.MolFromSmarts('[#6]1ccccc1')        # query mol (for matching, not full chemistry)
```

### Writing molecules

```python
smiles    = Chem.MolToSmiles(mol)                       # canonical SMILES
smiles    = Chem.MolToSmiles(mol, isomericSmiles=False) # drop stereo
mol_block = Chem.MolToMolBlock(mol)
inchi     = Chem.MolToInchi(mol)
inchikey  = Chem.MolToInchiKey(mol)                     # hashable identifier
```

### Batch processing with Suppliers / Writers

```python
import gzip

# SDF in memory (random access, NOT thread-safe to share)
suppl = Chem.SDMolSupplier('molecules.sdf')
for mol in suppl:
    if mol is not None:        # parsing failures yield None
        ...

# SMILES files
suppl = Chem.SmilesMolSupplier('molecules.smi', titleLine=False)

# Streaming / large or compressed files (no random access, lower memory)
with gzip.open('molecules.sdf.gz') as f:
    for mol in Chem.ForwardSDMolSupplier(f):
        ...

# Parallel parsing of large datasets
suppl = Chem.MultithreadedSDMolSupplier('large.sdf')

# Writing
writer = Chem.SDWriter('output.sdf')
for mol in molecules:
    writer.write(mol)
writer.close()
```

**Notes**
- All `MolFrom*` functions return `None` on failure — check before using.
- Molecules are sanitized on import by default (valence check, aromaticity perception, stereo).
- `MolSupplier` objects are **not** safe to share across threads.

## 2. Sanitization and Validation

RDKit runs ~13 sanitization steps on import (valence checking, aromaticity perception, ring finding, chirality). Control it when working with unusual or malformed input.

```python
# Skip automatic sanitization, then inspect problems
mol = Chem.MolFromSmiles('C1=CC=CC=C1', sanitize=False)

problems = Chem.DetectChemistryProblems(mol)
for p in problems:
    print(p.GetType(), p.Message())

# Manual / partial sanitization
Chem.SanitizeMol(mol)
Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
```

**Common issues:** explicit valence exceeding the allowed maximum (raises), invalid aromatic rings causing kekulization errors, radicals not assigned without explicit specification. Use `DetectChemistryProblems()` to debug before `SanitizeMol()`.

### Inspecting structure

```python
for atom in mol.GetAtoms():
    print(atom.GetSymbol(), atom.GetIdx(), atom.GetDegree(), atom.IsInRing())
for bond in mol.GetBonds():
    print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())

ring_info = mol.GetRingInfo()
ring_info.NumRings()
ring_info.AtomRings()        # tuples of atom indices

# Stereochemistry
from rdkit.Chem import FindMolChiralCenters
FindMolChiralCenters(mol, includeUnassigned=True)

# Fragments / scaffold
frags = Chem.GetMolFrags(mol, asMols=True)
from rdkit.Chem.Scaffolds import MurckoScaffold
scaffold = MurckoScaffold.GetScaffoldForMol(mol)
```

---

## 3. Substructure Searching and SMARTS

### Basic matching

```python
query = Chem.MolFromSmarts('C(=O)[OH]')   # carboxylic acid

mol.HasSubstructMatch(query)              # bool
mol.GetSubstructMatch(query)              # first match: tuple of atom indices
mol.GetSubstructMatches(query)            # all matches: tuple of tuples
mol.GetSubstructMatches(query, uniquify=True, maxMatches=1000)
```

### Building queries

- **SMARTS** is the query language (`MolFromSmarts`); it is more expressive than SMILES (atom/bond logic, recursion, ring/charge/H constraints).
- Prefer SMARTS over SMILES for queries — a SMILES parsed as a query keeps default valence/aromaticity assumptions that can surprise you.

```python
primary_alcohol = Chem.MolFromSmarts('[CH2][OH1]')
amide           = Chem.MolFromSmarts('C(=O)N')
aromatic_n      = Chem.MolFromSmarts('[nR]')        # aromatic N in a ring
macrocycle      = Chem.MolFromSmarts('[r{12-}]')    # ring atom in ring >= 12
recursive       = Chem.MolFromSmarts('[$([CX3]=[OX1]),$([CX3+]-[OX1-])]')  # carbonyl (either form)
```

### Matching rules (gotchas)

- Unspecified properties in the query match **any** value in the target.
- Hydrogens are ignored unless explicitly specified (`[CH2]`, `[OH1]`).
- A charged query atom won't match an uncharged target atom.
- An aromatic query atom won't match an aliphatic target atom (and vice versa).

See `references/smarts_patterns.md` for a large library of functional-group and structural patterns. The `scripts/substructure_filter.py` helper filters molecule lists by one or more SMARTS patterns.

---

## 4. 2D / 3D Coordinates and Reactions

### 2D coordinates (depiction)

```python
from rdkit.Chem import AllChem

AllChem.Compute2DCoords(mol)              # generate 2D coords for drawing

# Align depiction to a template (consistent orientation across a series)
template = Chem.MolFromSmiles('c1ccccc1')
AllChem.Compute2DCoords(template)
AllChem.GenerateDepictionMatching2DStructure(mol, template)
```

### 3D coordinates and conformers (ETKDG)

```python
mol = Chem.AddHs(mol)                      # add Hs BEFORE embedding for good geometry

AllChem.EmbedMolecule(mol, randomSeed=42)              # single conformer
conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)

# Force-field optimization
AllChem.MMFFOptimizeMolecule(mol)         # MMFF94 (preferred when params available)
AllChem.UFFOptimizeMolecule(mol)          # UFF fallback
for cid in conf_ids:
    AllChem.MMFFOptimizeMolecule(mol, confId=cid)

# Compare / align conformers
rms = AllChem.GetConformerRMS(mol, conf_ids[0], conf_ids[1])
AllChem.AlignMol(probe_mol, ref_mol)

# Constrained embedding (keep a core fixed)
AllChem.ConstrainedEmbed(mol, core_mol)

mol = Chem.RemoveHs(mol)                   # optional: strip Hs afterward
```

**Tip:** always `AddHs()` before 3D embedding/optimization, then `RemoveHs()` if you only need heavy-atom coordinates.

### Chemical reactions (reaction SMARTS)

```python
from rdkit.Chem import AllChem

# reactants >> products, with atom maps to track atoms
rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[OH]>>[C:1](=[O:2])[O][CH3]')  # methyl esterification

products = rxn.RunReactants((mol,))        # tuple of product tuples
for product_set in products:
    for product in product_set:
        Chem.SanitizeMol(product)          # products are NOT auto-sanitized
        print(Chem.MolToSmiles(product))
```

- Atom mapping (`:1`, `:2`) carries atoms from reactants to products.
- Products from `RunReactants` are unsanitized — call `Chem.SanitizeMol()` (and dedupe canonical SMILES) yourself.
- Reaction fingerprints: `AllChem.CreateDifferenceFingerprintForReaction(rxn)` for reaction similarity.

The `scripts/transform_and_embed.py` helper applies a reaction SMARTS to input molecules and generates optimized 3D conformers for each product.

### Drawing / visualization

```python
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

Draw.MolToFile(mol, 'molecule.png', size=(300, 300))
img = Draw.MolsToGridImage([m1, m2, m3], molsPerRow=2, subImgSize=(200, 200))

# Highlight a substructure match
match = mol.GetSubstructMatch(Chem.MolFromSmarts('c1ccccc1'))
Draw.MolToImage(mol, highlightAtoms=match)

# Fine control (atom indices, stereo annotations, etc.)
drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
drawer.drawOptions().addStereoAnnotation = True
drawer.DrawMolecule(mol)
drawer.FinishDrawing()
open('molecule.png', 'wb').write(drawer.GetDrawingText())
```

In Jupyter: `from rdkit.Chem.Draw import IPythonConsole` enables automatic inline rendering.

---

## 5. Descriptors, Fingerprints & Similarity (condensed)

These areas are de-emphasized in this skill; see `references/descriptors_reference.md` and `references/api_reference.md` for full coverage. Quick reference:

```python
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

# Descriptors
Descriptors.MolWt(mol); Descriptors.MolLogP(mol); Descriptors.TPSA(mol)
Descriptors.NumHDonors(mol); Descriptors.NumHAcceptors(mol)
all_desc = Descriptors.CalcMolDescriptors(mol)   # dict of every descriptor

# Morgan (ECFP-like) fingerprints via the modern generator API
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp1, fp2 = gen.GetFingerprint(mol1), gen.GetFingerprint(mol2)
DataStructs.TanimotoSimilarity(fp1, fp2)
DataStructs.BulkTanimotoSimilarity(fp1, [gen.GetFingerprint(m) for m in mols])
```

### Common modification utilities

```python
Chem.AddHs(mol); Chem.RemoveHs(mol)
Chem.Kekulize(mol); Chem.SetAromaticity(mol)
Chem.ReplaceSubstructs(mol, query, replacement)[0]

from rdkit.Chem.MolStandardize import rdMolStandardize
mol_neutral = rdMolStandardize.Uncharger().uncharge(mol)
```

---

## Best Practices

**Always check for `None`:**

```python
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print(f"Failed to parse: {smiles}")
    continue
```

**Performance**
- Pickle parsed molecules instead of re-parsing (`pickle.dump(mols, f)`).
- Use `Bulk*` similarity functions and reuse a single fingerprint generator.
- Use `ForwardSDMolSupplier` / `MultithreadedSDMolSupplier` for large files.

**Thread safety:** molecule I/O (strings), coordinate generation, fingerprints, descriptors, substructure search, reactions, and drawing are generally thread-safe. **`MolSupplier` objects are not** — don't share them across threads.

## Common Pitfalls

1. Not checking for `None` after parsing.
2. Sanitization failures — debug with `DetectChemistryProblems()`.
3. Forgetting `AddHs()` before 3D embedding or H-dependent properties.
4. Confusing 2D vs 3D coordinates before visualization/analysis.
5. SMARTS: unspecified query properties match anything; aromatic≠aliphatic.
6. Forgetting to `SanitizeMol()` reaction products from `RunReactants`.
7. Sharing a `MolSupplier` across threads.

## Resources

### references/
- `api_reference.md` — RDKit modules, functions, and classes by functionality.
- `descriptors_reference.md` — full list of molecular descriptors.
- `smarts_patterns.md` — functional-group and structural SMARTS patterns (key reference for §3).

### scripts/
- `substructure_filter.py` — filter molecule lists by SMARTS substructure patterns (§3).
- `transform_and_embed.py` — apply a reaction SMARTS and generate 3D conformers for products (§4).

Run scripts directly or use them as templates. Load the references when you need specific API details or pattern examples.
