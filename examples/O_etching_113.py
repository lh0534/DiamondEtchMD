"""
O radical etching of O-terminated C(113) surface.

The 113 surface has two options: "" (unterminated, default) and "O"
(O-terminated). There are no geometric reconstructions — it's purely
about surface chemistry.

    python examples/O_etching_113.py
    sbatch O_etching_113/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "113",
    surface         = "O",          # O-terminated surface state
    
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
