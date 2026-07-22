"""Shared helpers for the chai-folding scripts: FASTA assembly + a run wrapper.

Chain letters are assigned by order of appearance, so `build_fasta` takes an
ordered list of (entity, sequence) pairs and names each record by the letter it
will receive: records[i] -> ascii_uppercase[i].
"""
from __future__ import annotations

import string
from pathlib import Path

from _device import resolve_device


def build_fasta(records: list[tuple[str, str]], out: Path) -> Path:
    """Write a Chai FASTA. `records` is ordered [(entity, sequence), ...] where
    entity is 'protein' | 'ligand' | 'glycan' | 'rna' | 'dna'. Record i is named
    with chain letter i (A, B, C, ...).
    """
    letters = string.ascii_uppercase
    text = ''.join(
        f'>{entity}|{letters[i]}\n{seq}\n'
        for i, (entity, seq) in enumerate(records)
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def split_target_chains(target_sequence: str) -> list[str]:
    """Split an AF-multimer-style ':'-delimited target into its chain sequences."""
    return [c.strip() for c in target_sequence.split(':') if c.strip()]


def fold(fasta: Path, out_dir: Path, constraint_path: Path | None = None,
         num_diffn_timesteps: int = 100):
    """Run Chai-1 on `fasta`, writing 5 models + score npz into `out_dir`."""
    from chai_lab.chai1 import run_inference

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_inference(
        fasta_file=Path(fasta),
        output_dir=out_dir,
        device=resolve_device(),
        use_esm_embeddings=True,
        num_diffn_timesteps=num_diffn_timesteps,
        constraint_path=constraint_path,
        full_data=True,
    )
