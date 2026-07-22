"""Shared MDAnalysis helpers for the RMSD gate + MM-PBSA selections.

The binder is taken as the LAST solute fragment (Chai folds it last, and the
glycoprotein prep reorders it last), so target/binder split is fragment-based
and makes no assumption about chain letters.
"""
from __future__ import annotations

import numpy as np

# Water + common ions to exclude from the solute.
SOLVENT_RESNAMES = (
    'WAT', 'HOH', 'TIP3', 'TIP4', 'T3P', 'T4P', 'SOL', 'SPC', 'OPC',
    'Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+', 'Zn2+',
    'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'IP', 'IM',
)


def split_target_binder(universe):
    """Return (target AtomGroup, binder AtomGroup) — binder = last solute fragment."""
    solute = universe.atoms[~np.isin(universe.atoms.resnames, SOLVENT_RESNAMES)]
    *_, binder = solute.fragments
    target = solute - binder
    return target, binder


def format_for_cpptraj(resids: np.ndarray) -> str:
    """Collapse a sorted resid array into an AMBER/cpptraj mask, e.g. ':1-10,25-43'."""
    resids = list(map(int, resids))
    out, start, cur = ':', resids[0], resids[0] - 1
    for r in resids:
        if r - cur > 1:
            out += f'{start}-{cur},'
            start = r
        cur = r
    return out + f'{start}-{cur}'
