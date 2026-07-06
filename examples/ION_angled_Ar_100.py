"""
Ar+ sputtering at 45-degree off-normal incidence on C(100) 2x1 surface.

Non-zero angle is in degrees from the surface normal; the velocity is
decomposed into normal (vz) and lateral (vy) components.

python examples/ION_angled_Ar_100.py
cd ION_100_2x1_angled_Ar; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)   # 64 atoms/ML for 8×8 box

spec = SimSpec(
    orientation    = "100",
    surface        = "2x1",
    surface_temperature    = 300.0,

    species        = "Ar",
    energy         = 50.0,         # eV
    ion_angle          = 45.0,         # degrees from surface normal

    fluence        = 20,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 4,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "ION_100_2x1_Ar_50eV_45deg",
)

validate(spec)
make_sim(spec, Path("ION_100_2x1_angled_Ar"))
