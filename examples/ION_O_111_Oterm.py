"""
O radical etching of C(111) Pandey-chain reconstructed, O-terminated surface.

The surface key "1x1_O" encodes no reconstruction with O termination.
Valid 111 surface keys:
  1x1, 2x1_single, 2x1_pandey, 1x1_O, 2x1_single_O, 2x1_pandey_O

python examples/ION_O_111_Oterm.py
sbatch ION_O_111_Oterm/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

nx, ny = 5, 9

spec = SimSpec(
    orientation    = "111",
    surface        = "1x1_O",
    temperature    = 300.0,

    species        = "O",
    energy         = 4.0,          # eV
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("111", nx, ny),
    box_x          = nx,
    box_y          = ny,
    box_depth      = 3,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "ION_111_Oterm_O_4eV",
)

make_sim(spec, Path("ION_O_111_Oterm"))