"""
O+ etching of C(111) O-terminated surface.

Valid 111 surface keys:
  1x1, 2x1_single, 2x1_pandey, 1x1_O, 2x1_single_O, 2x1_pandey_O

python examples/ION_O_111_Oterm.py
cd ION_O_111_Oterm; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate

nx, ny = 5, 9
ml = compute_ml("111", nx, ny)

spec = SimSpec(
    orientation    = "111",
    surface        = "1x1_O",
    surface_temperature    = 300.0,

    species        = "O",
    energy         = 15.0,          # eV
    ion_angle          = 0.0,

    fluence        = 50,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 3,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "ION_111_Oterm_O_15eV",
)

validate(spec)
make_sim(spec, Path("ION_O_111_Oterm"))
