"""Run the minimize -> heat -> equilibrate -> NPT production pipeline.

Re-invoking this after a crash RESUMES from the last checkpoint (reads prod.log
for remaining steps). Point it at a directory that already contains
system.prmtop / system.inpcrd (from a build script).

Usage:
    python run_production.py <system_dir> [ns] [--implicit]
"""
from __future__ import annotations

import sys
from pathlib import Path

from _device import resolve_md_platform


def main(system_dir: str, ns: float = 100.0, implicit: bool = False) -> None:
    path = Path(system_dir)
    prod_dt = 0.004                                  # 4 fs (HMR)
    prod_steps = int(ns / prod_dt * 1000)            # 1 ns = 250_000 steps

    if implicit:
        from molecular_simulations.simulate import ImplicitSimulator
        from openmm.app import GBn2
        ImplicitSimulator(
            path=path, implicit_solvent=GBn2, prod_steps=prod_steps,
            solvent_dielectric=78.5, platform=resolve_md_platform(), device_ids=[0],
        ).run()
    else:
        from molecular_simulations.simulate import Simulator
        Simulator(
            path=path,
            equil_steps=1_250_000,                   # 2.5 ns @ 2 fs
            prod_steps=prod_steps,
            temperature=300.0,
            platform=resolve_md_platform(),
            device_ids=[0],                          # left at 0 under Parsl GPU masking
        ).run()
    print(f'Production complete ({ns} ns) in {path}')


if __name__ == '__main__':
    ns = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    implicit = '--implicit' in sys.argv
    main(sys.argv[1], ns, implicit)
