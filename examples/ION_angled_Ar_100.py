"""
Ar+ sputtering at 45-degree off-normal incidence on C(100).

Non-zero incidence angle is specified in degrees from the surface normal.
The velocity is decomposed into normal (vz) and lateral (vy) components.
Angled bombardment affects sputter yield and surface morphology.

python examples/ION_100_2x1_angled_Ar.py
sbatch ION_100_2x1_angled_Ar/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

nx, ny = 8, 8

spec = SimSpec(
    orientation    = "100",
    surface         = "2x1",

    temperature    = 300.0,

    species        = "Ar",
    energy         = 50.0,         # eV
    angle          = 45.0,         # degrees from surface normal

    fluence        = 20,
    ml             = compute_ml("100", nx, ny),
    box_x          = nx,
    box_y          = ny,
    box_depth      = 4,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "ION_100_2x1_Ar_50eV_45deg",
)

make_sim(spec, Path("ION_100_2x1_angled_Ar"))
