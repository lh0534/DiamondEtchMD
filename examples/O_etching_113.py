"""
O radical etching of C(113) with O termination.

The 113 surface has no reconstructions (only "bare"). It uses a higher
ml_factor (4) than 100 or 111, so a 9x3 box already has 108 atoms per
monolayer. Default box dimensions are (9, 3, 3) for this orientation.

    python examples/O_etching_113.py
    sbatch O_etching_113/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "113",
    reconstruction = "bare",       # only option for 113
    termination    = "O",
    temperature    = 300.0,

    species        = "O",
    energy         = 0.5,          # eV
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("113", 9, 3),
    box_x          = 9,
    box_y          = 3,
    box_depth      = 3,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "113_O_0.5eV",
)

make_sim(spec, Path("O_etching_113"))
