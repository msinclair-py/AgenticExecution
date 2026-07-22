"""Compute MM-P(G)BSA binding free energy from a complex trajectory.

Derives the receptor/ligand cpptraj selections from the target/binder fragment
split, runs GB+PB energies frame-parallel, and prints ΔG_binding [mean, std]
(PB). Requires $AMBERHOME.

Usage:
    AMBERHOME=/path/to/amber \
        python run_mmpbsa.py <system.prmtop> <prod.dcd> [n_cpus]
"""
from __future__ import annotations

import sys

import MDAnalysis as mda

from mda_utils import format_for_cpptraj, split_target_binder


def main(prmtop: str, dcd: str, n_cpus: int = 8) -> None:
    from molecular_simulations.simulate import MMPBSA

    u = mda.Universe(prmtop, dcd)
    target, binder = split_target_binder(u)
    selections = [
        format_for_cpptraj(target.residues.resids),   # receptor
        format_for_cpptraj(binder.residues.resids),   # ligand (binder)
    ]
    print('MM-PBSA selections:', selections)

    mmpbsa = MMPBSA(
        top=prmtop, dcd=dcd, selections=selections,
        n_cpus=n_cpus, parallel_mode='frame',
    )
    mmpbsa.run()
    mean, std = mmpbsa.free_energy
    print(f'ΔG_binding (PB) = {mean:.2f} +/- {std:.2f} kcal/mol')
    print('Per-term stats in statistics.json; summary in deltaG.txt')


if __name__ == '__main__':
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(sys.argv[1], sys.argv[2], n)
