"""Assess binder stability: align on the target backbone, measure binder RMSD.

Returns [mean, std] of the per-frame binder heavy-atom RMSD. Because the
alignment is on the target only, this captures both conformational distortion
and rigid-body drift away from the binding site. Compare mean to an rmsd_cutoff
(default 5.0 A) to decide "stably bound" vs "dissociated".

Usage:
    python rmsd_stability.py <system.prmtop> <prod.dcd> [rmsd_cutoff]
"""
from __future__ import annotations

import sys

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.rms import RMSD

from mda_utils import split_target_binder


def binder_rmsd(prmtop: str, dcd: str) -> tuple[float, float]:
    u = mda.Universe(prmtop, dcd)
    target, binder = split_target_binder(u)
    first, last = target.residues.resids[0], target.residues.resids[-1]
    binder_sel = 'not type H and resid ' + ' '.join(
        map(str, binder.residues.resids)
    )

    solute = target + binder
    R = RMSD(solute, select=f'backbone and resid {first} to {last}',
             groupselections=[binder_sel])
    R.run()
    per_frame = R.rmsd[:, -1]            # last column = binder group RMSD
    return float(np.mean(per_frame)), float(np.std(per_frame))


def main(prmtop: str, dcd: str, cutoff: float = 5.0) -> None:
    mean, std = binder_rmsd(prmtop, dcd)
    stable = mean < cutoff
    print(f'binder RMSD = {mean:.2f} +/- {std:.2f} A  '
          f'({"STABLE" if stable else "DISSOCIATED"} vs cutoff {cutoff})')
    sys.exit(0 if stable else 1)


if __name__ == '__main__':
    cut = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    main(sys.argv[1], sys.argv[2], cut)
