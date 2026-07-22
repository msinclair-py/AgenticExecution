"""Turn the compact constraint dict into a Chai-1 constraint CSV.

Dict form (one entry per restraint):
    {0: {'chainA': 'B', 'resA': '', 'chainB': 'A', 'resB': 'N9',
         'const_type': 'pocket', 'distance': 5.0}}

The CSV header Chai expects and the per-row template (confidence fixed at 1.0,
min distance 0.0, max distance = `distance`) match the project's ChaiBinder.
"""
from __future__ import annotations

from pathlib import Path
from string import Template

_HEADER = (
    'chainA,res_idxA,chainB,res_idxB,connection_type,confidence,'
    'min_distance_angstrom,max_distance_angstrom,comment,restraint_id'
)
_ROW = Template(
    '$chainA,$resA,$chainB,$resB,$const_type,1.0,0.0,$distance,comment,restraint_$i'
)


def write_constraints(constraints: dict[int, dict], out: Path) -> Path:
    """Write `constraints` to `out` as a Chai constraint CSV. Returns the path."""
    lines = [_HEADER]
    for i, params in constraints.items():
        params = dict(params)
        params['i'] = i
        lines.append(_ROW.substitute(**params))
    out = Path(out)
    out.write_text('\n'.join(lines) + '\n')
    return out


if __name__ == '__main__':
    demo = {
        0: {'chainA': 'B', 'resA': '', 'chainB': 'A', 'resB': 'N9',
            'const_type': 'pocket', 'distance': 5.0},
    }
    print(write_constraints(demo, Path('constraints.txt')).read_text())
