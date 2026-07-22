"""End-to-end binder pipeline: PDB -> build -> simulate -> RMSD gate -> MM-PBSA.

Ties the other scripts together. Auto-detects glycoproteins and routes them
through the explicit + binder-last prep. Skips MM-PBSA when the binder
dissociates (RMSD > cutoff), matching the project's gate.

Usage:
    AMBERHOME=/path/to/amber \
        python fold_to_dG_pipeline.py <complex.pdb> <binder_sequence> <work_dir> \
            [ns=25] [rmsd_cutoff=5.0]
"""
from __future__ import annotations

import sys
from pathlib import Path

from _device import resolve_md_platform
from prep_glycoprotein import _has_sugars, reorder_binder_last
from rmsd_stability import binder_rmsd
from run_mmpbsa import main as run_mmpbsa


def main(pdb: str, binder_seq: str, work_dir: str,
         ns: float = 25.0, rmsd_cutoff: float = 5.0) -> None:
    from molecular_simulations.build import ExplicitSolvent, convert_cif_with_gemmi
    from molecular_simulations.simulate import Simulator

    pdb_path, work = Path(pdb), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if pdb_path.suffix == '.cif':
        pdb_path = Path(convert_cif_with_gemmi(pdb_path))

    # 1. Build (glycoprotein-aware).
    if _has_sugars(pdb_path):
        build_pdb = reorder_binder_last(pdb_path, binder_seq, work)
        ExplicitSolvent(path=work, pdb=build_pdb, glycans=True).build()
    else:
        ExplicitSolvent(path=work, pdb=pdb_path, padding=10.0).build()
    print('[1/4] build done')

    # 2. Simulate.
    prod_steps = int(ns / 0.004 * 1000)
    Simulator(path=work, prod_steps=prod_steps, platform=resolve_md_platform(),
              device_ids=[0]).run()
    print(f'[2/4] {ns} ns production done')

    # 3. RMSD gate.
    prmtop, dcd = work / 'system.prmtop', work / 'prod.dcd'
    mean, std = binder_rmsd(str(prmtop), str(dcd))
    print(f'[3/4] binder RMSD = {mean:.2f} +/- {std:.2f} A (cutoff {rmsd_cutoff})')
    if mean >= rmsd_cutoff:
        print('Binder DISSOCIATED — skipping MM-PBSA.')
        return

    # 4. MM-PBSA.
    print('[4/4] running MM-PBSA...')
    run_mmpbsa(str(prmtop), str(dcd))


if __name__ == '__main__':
    ns = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
    cutoff = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
    main(sys.argv[1], sys.argv[2], sys.argv[3], ns, cutoff)
