"""
make_all_surfaces.py — generate simulation dirs for every valid
orientation / surface combination.

Run from the DiamondEtchMD directory:
    python make_all_surfaces.py

Each surface gets its own directory under surfaces/.
Submit with:  sbatch surfaces/<name>/submit
Open impact_snaps/0.data or ML_impacts.dump in OVITO after running.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diamond_etch_md import SimSpec, compute_ml, make_sim
from diamond_etch_md.orientations import ORIENT

OUTDIR = Path("surfaces")

# Every valid (orientation, surface) combination
CASES = [
    # C(100)
    ("100", "1x1"),
    ("100", "2x1"),
    ("100", "2x1_O"),
    ("100", "O_ether"),
    # C(110)
    ("110", ""),
    # ("110", "O"),
    # C(111)
    ("111", "1x1"),
    ("111", "2x1_single"),
    ("111", "2x1_pandey"),
    ("111", "1x1_O"),
    ("111", "2x1_single_O"),
    ("111", "2x1_pandey_O"),
    # C(113)
    ("113", ""),
    ("113", "O"),
]

for orient, surface in CASES:
    cfg = ORIENT[orient]
    dx, dy, dz = cfg["default_box"]
    ml = compute_ml(orient, dx, dy)
    dirname = f"{orient}_{surface}" if surface else orient

    spec = SimSpec(
        orientation = orient,
        surface     = surface,
        species     = "O",
        energy      = 0.5,
        temperature = 300.0,
        ml          = ml,
        box_x       = dx,
        box_y       = dy,
        box_depth   = dz,
        wall_hours  = 1,
        fluence     = 0,
        name        = dirname,
    )

    make_sim(spec, OUTDIR / dirname)

print(f"\nGenerated {len(CASES)} surface directories under {OUTDIR}/")
print("Submit each with:  sbatch surfaces/<name>/submit")
print("Open impact_snaps/0.data or ML_impacts.dump in OVITO after running.")
