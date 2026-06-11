"""
O+ etching of the bare C(110) surface.

python examples/ION_O_110.py
sbatch ION_110_O/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate

nx, ny = 4, 6
ml = compute_ml("110", nx, ny)   # 96 atoms/ML for 4×6 box

spec = SimSpec(
    orientation = "110",
    surface     = "",              # unterminated C(110)
    temperature = 300.0,

    species     = "O",
    energy      = 5,              # eV
    angle       = 0.0,

    fluence     = 20,
    ml          = ml,
    box_x       = nx,
    box_y       = ny,
    box_depth   = 5,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours  = 24,
    account     = "dgraves",
    name        = "ION_110_bare_O_5eV",
)

validate(spec)
make_sim(spec, Path("ION_110_O"))
