"""
Ar+ physical sputtering of C(100) 2x1 surface.

python examples/ION_Ar_100.py
sbatch ION_100_Ar/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)   # 64 atoms/ML for 8×8 box

spec = SimSpec(
    orientation    = "100",
    surface        = "2x1",
    temperature    = 300.0,

    species        = "Ar",
    energy         = 50.0,        # eV
    angle          = 0.0,

    fluence        = 20,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 4,           # deep slab for 50 eV Ar+

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "ION_100_2x1_Ar_50eV",
)

validate(spec)
make_sim(spec, Path("ION_100_Ar"))
