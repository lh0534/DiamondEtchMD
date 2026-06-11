"""
O2+ ion bombardment of C(111) Pandey-chain reconstructed surface.

python examples/ION_O2_111.py
sbatch ION_O2_111p/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

nx, ny = 5, 9

spec = SimSpec(
    orientation    = "111",
    surface         = "2x1_pandey",

    temperature    = 300.0,

    species        = "O2",
    energy         = 50.0,                          # 50 eV total -> 25 eV per O atom
    angle          = 0.0,

    fluence        = 20,
    ml             = compute_ml("111", nx, ny),
    box_x          = nx,
    box_y          = ny,
    box_depth      = 6,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "ION_111p_O2_50eV",
)

make_sim(spec, Path("ION_O2_111p"))
