"""
Ar+ sputtering at 45-degree off-normal incidence on C(100).

Non-zero incidence angle is specified in degrees from the surface normal.
The velocity is decomposed into normal (vz) and lateral (vy) components.
Angled bombardment affects sputter yield and surface morphology.

    python examples/angled_Ar_100.py
    sbatch angled_Ar_100/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

nx, ny = 8, 8

spec = SimSpec(
    orientation    = "100",
    surface         = "1x1",

    temperature    = 300.0,

    species        = "Ar",
    energy         = 100.0,        # eV
    angle          = 45.0,         # degrees from surface normal

    fluence        = 50,
    ml             = compute_ml("100", nx, ny),
    box_x          = nx,
    box_y          = ny,
    box_depth      = 10,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_Ar_100eV_45deg",
)

make_sim(spec, Path("angled_Ar_100"))
