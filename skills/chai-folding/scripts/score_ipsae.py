"""Select the best Chai model and score its interface with ipSAE.

Reads the 5 pred.model_idx_{i}.cif + scores.model_idx_{i}.npz that
run_inference (full_data=True) wrote, ranks by mean ipSAE, converts the winner
to PDB, and prints the interface metrics.

Usage:
    python score_ipsae.py <fold_out_dir> [out.pdb]
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(fold_dir: str, out_pdb: str = 'best.pdb', n_models: int = 5) -> None:
    import gemmi
    from molecular_simulations.analysis import ipSAE

    fold_dir = Path(fold_dir)
    best_score, best_i, best_scores = -float('inf'), 0, None
    for i in range(n_models):
        model = fold_dir / f'pred.model_idx_{i}.cif'
        npz = fold_dir / f'scores.model_idx_{i}.npz'
        scorer = ipSAE(structure_file=model, plddt_file=npz, pae_file=npz)
        scorer.run()
        s = scorer.scores['ipSAE'].mean()
        # `best_scores is None` guard: NaN on model 0 still yields a populated
        # result (NaN > -inf is False), and a legitimate ipSAE 0.0 is kept.
        if best_scores is None or s > best_score:
            best_score, best_i, best_scores = s, i, scorer.scores

    model = fold_dir / f'pred.model_idx_{best_i}.cif'
    gemmi.read_structure(str(model)).write_pdb(str(out_pdb))

    print(f'Best model: idx {best_i}  (mean ipSAE = {best_score:.3f})')
    print('Interface metrics per chain pair:')
    print(best_scores)
    print('Wrote', out_pdb)


if __name__ == '__main__':
    out = sys.argv[2] if len(sys.argv) > 2 else 'best.pdb'
    main(sys.argv[1], out)
