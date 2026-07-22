"""Device-agnostic Chroma binder designer.

A distillation of the project's ChromaDesigner: load the model once, extend the
target with an empty binder segment, build the design mask, compose the
Inflate + Substructure conditioners, sample, and save backbones as PDB.

Chain labels: receptor = 1, binder = 2 (Chroma convention).
"""
from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import torch
from Bio import PDB
from chroma import Chroma, Protein, conditioners

from _device import resolve_device


def load_model(backbone_weights: str, design_weights: str, device: str) -> Chroma:
    return Chroma(weights_backbone=str(backbone_weights),
                  weights_design=str(design_weights), device=device)


def hotspot_vector(pdb: Path, hotspot_resids: str, vector_scaling: float, device: str):
    """Displacement from target COM to hotspot COM, scaled. hotspot_resids is a
    space-separated 1-indexed resid string, e.g. '47 48 49'."""
    u = mda.Universe(str(pdb))
    hs = u.select_atoms(f'protein and resid {hotspot_resids}')
    vec = hs.center_of_mass() - u.atoms.center_of_mass()
    magnitude = float(np.linalg.norm(vec) * vector_scaling)
    return torch.tensor(vec, dtype=torch.float), magnitude


def prepare_protein(pdb: Path, len_binder: int, device: str):
    """Append an empty binder segment and build the design mask (binder-only)."""
    protein = Protein(str(pdb), device=device)
    X, C, S = protein.to_XCS()

    X = torch.cat([X, torch.zeros(1, len_binder, 4, 3).to(device)], dim=1)
    C = torch.cat([C, torch.full((1, len_binder), 2).to(device)], dim=1)  # binder=2
    S = torch.cat([S, torch.full((1, len_binder), 0).to(device)], dim=1)
    protein = Protein(X, C, S, device=device)
    X, C, S = protein.to_XCS()

    L_receptor = int((C == 1).sum())
    L_complex = C.shape[-1]

    mask_aa = torch.zeros(L_complex, 20)
    for i in range(L_complex):
        if i >= L_receptor:                       # binder residue: allow all 20
            mask_aa[i] = torch.ones(20)
    mask_aa = mask_aa[None].to(device)

    protein.sys.save_selection(gti=list(range(L_receptor)), selname='receptor')
    return protein, mask_aa


def make_conditioner(model: Chroma, protein: Protein, vector, magnitude, device: str):
    substructure = conditioners.SubstructureConditioner(
        protein=protein, backbone_model=model.backbone_network,
        selection='namesel receptor',
    ).to(device)
    displacement = conditioners.InflateConditioner(vector, magnitude).to(device)
    return conditioners.ComposedConditioner([displacement, substructure])


def sample(model: Chroma, protein: Protein, conditioner, mask_aa,
           steps: int, n_backbones: int, n_designs: int):
    proteins, _ = model.sample(
        protein_init=protein, conditioner=conditioner, design_selection=mask_aa,
        langevin_factor=2, langevin_isothermal=True, inverse_temperature=8.0,
        sde_func='langevin', full_output=True,
        steps=steps, samples=n_backbones, num_designs=n_designs,
    )
    return proteins


def save_pdbs(proteins, out_dir: Path, name: str, n_designs: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(proteins, list):
        proteins = [proteins]
    if proteins and isinstance(proteins[0], list):
        proteins = [p for row in proteins for p in row]

    written: list[Path] = []
    for i, protein in enumerate(proteins):
        cif = out_dir / f'{name}_bb{i // n_designs}_d{i % n_designs}.cif'
        protein.to(str(cif))
        pdb = _cif_to_pdb(cif)
        cif.unlink(missing_ok=True)
        written.append(pdb)
    return written


def _cif_to_pdb(cif: Path) -> Path:
    structure = PDB.MMCIFParser(QUIET=True).get_structure('s', str(cif))
    io = PDB.PDBIO()
    io.set_structure(structure)
    pdb = cif.with_suffix('.pdb')
    io.save(str(pdb))
    return pdb


def design(pdb, backbone_weights, design_weights, hotspot_resids, len_binder,
           out_dir, steps=200, n_backbones=10, n_designs=1, vector_scaling=1.0):
    """One-call convenience wrapper. Returns the list of written backbone PDBs."""
    device = resolve_device()
    pdb, out_dir = Path(pdb), Path(out_dir)
    model = load_model(backbone_weights, design_weights, device)
    vector, magnitude = hotspot_vector(pdb, hotspot_resids, vector_scaling, device)
    protein, mask_aa = prepare_protein(pdb, len_binder, device)
    conditioner = make_conditioner(model, protein, vector, magnitude, device)
    proteins = sample(model, protein, conditioner, mask_aa, steps, n_backbones, n_designs)
    return save_pdbs(proteins, out_dir, pdb.stem, n_designs)
