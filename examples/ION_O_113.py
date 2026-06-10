"""
O radical etching of O-terminated C(113) surface.

python examples/ION_O_113.py
sbatch ION_O_113/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

nx, ny = 9, 3

spec = SimSpec(
    orientation    = "113",
    surface        = "",                            # no surface termination
    temperature    = 300.0,

    species        = "O",
    energy         = 3,                             # eV
    angle          = 0.0,

    fluence        = 30,
    ml             = compute_ml("113", nx, ny),     # 108 atoms/ML for 9×3 box
    box_x          = nx,
    box_y          = ny,
    box_depth      = 3,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "ION_113_O_3eV",
)

make_sim(spec, Path("ION_O_113"))
