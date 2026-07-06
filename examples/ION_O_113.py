"""
O+ etching of the C(113) surface.

python examples/ION_O_113.py
cd ION_O_113; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate

nx, ny = 9, 3
ml = compute_ml("113", nx, ny)   # 108 atoms/ML for 9×3 box

spec = SimSpec(
    orientation    = "113",
    surface        = "",                            # no surface termination
    surface_temperature    = 300.0,

    species        = "O",
    energy         = 3,                             # eV
    ion_angle          = 0.0,

    fluence        = 30,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 3,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "ION_113_O_3eV",
)

validate(spec)
make_sim(spec, Path("ION_O_113"))
