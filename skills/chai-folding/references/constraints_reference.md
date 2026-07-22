# Chai-1 FASTA & constraint reference

## FASTA entity grammar

Each record is a header line `>ENTITY|CHAINLETTER` followed by one sequence
line. Chai assigns chain letters **by order of appearance**, so name each record
by the letter it will receive and lay chains out in the intended order.

| Entity | Value on the sequence line | Notes |
|--------|----------------------------|-------|
| `protein` | one-letter amino-acid sequence | Standard 20 AAs. |
| `ligand` | a **SMILES** string | Small molecules / cofactors. |
| `glycan` | a glycan-tree string (see below) | Each glycan is its own chain. |
| `rna` | ribonucleotide sequence (ACGU) | |
| `dna` | deoxyribonucleotide sequence (ACGT) | |

Canonical layout for a binder campaign:

```
>protein|A      <- target chain 1
>protein|B      <- target chain 2 (if multi-chain)
>glycan|C       <- glycan(s), one record each, next free letters
>ligand|D       <- ligand(s) if any
>protein|E      <- binder, ALWAYS last so it is the final chain
```

### Glycan tree strings

A glycan chain's sequence encodes the tree with parenthesised linkages, e.g. a
common core `NAG(4-1 NAG(4-1 BMA(6-1 MAN)(3-1 MAN)))`. The root sugar is linked
to the protein via a **covalent** constraint from the protein acceptor atom to
the glycan's `C1` (see below). The `bindify` toolbelt derives these strings from
known glycosylation sites; when hand-authoring, follow Chai's glycan syntax.

## Constraint CSV schema

`constraint_path` points at a CSV with exactly this header:

```
chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id
```

- `chainA`, `chainB` — chain letters (must exist in the FASTA).
- `res_idxA`, `res_idxB` — residue token, optionally `@atom` (e.g. `N343@ND2`,
  `@C1`). A bare residue (`N9`) names a residue; empty means "any residue on the
  chain" (used on the constrained side of a pocket).
- `connection_type` — `pocket`, `contact`, or `covalent`.
- `confidence` — restraint weight; the project writes `1.0`.
- `min_distance_angstrom` / `max_distance_angstrom` — the writer sets min `0.0`,
  max = the `distance` you supply.
- `comment` — free text.
- `restraint_id` — unique id per row (`restraint_0`, ...).

### Dict shorthand → CSV

The project authors constraints as `{int: {...}}` and expands them. Each entry:

```python
{
  'chainA': 'A', 'resA': 'N343@ND2',   # side A (chain letter, residue[@atom])
  'chainB': 'C', 'resB': '@C1',        # side B
  'const_type': 'covalent',            # pocket | contact | covalent
  'distance': 1.5,                     # -> max_distance_angstrom
}
```

`scripts/write_constraints.py` maps the dict onto the CSV template
`$chainA,$resA,$chainB,$resB,$const_type,1.0,0.0,$distance,comment,restraint_$i`.

### Constraint types

- **pocket** — one side is a specific target residue, the other is a chain with
  an empty residue → that chain is pulled to bind within `distance` Å of the
  residue, non-specifically. Good for "bind near this epitope residue".
- **contact** — a specific residue-residue proximity restraint.
- **covalent** — a bond between two named atoms (`res@atom`). Used for:
  - **Glycan attachment:** `resA='N343@ND2'` (N-linked Asn side-chain N) ↔
    `resB='@C1'` (glycan root C1), `distance≈1.5`.
  - **Disulfide:** `resA='C55@SG'` ↔ `resB='C90@SG'`, `distance≈2.05`.
  - Other non-canonical covalent links (ligand-protein bonds, etc.).

### Glycan acceptor atoms

For an N-/O-linked glycan the target-side atom the glycan C1 bonds to:

| Linkage | Acceptor residue(s) | Atom |
|---------|---------------------|------|
| N-linked | Asn (N) | `ND2` |
| O-linked | Ser (S) | `OG` |
| O-linked | Thr (T) | `OG1` |

A residue that is not a valid acceptor for the requested linkage has no defined
atom — skip it rather than guessing. When a target has multiple chains, an
explicit chain prefix (`B:N343`) disambiguates which chain's residue 343 the
site sits on.
