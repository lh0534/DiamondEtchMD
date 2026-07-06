"""
RIE-etch: O+ ions with O• radical co-exposure on C(100) 2x1 surface.

flux_ratio O• radicals are deposited before each ion impact, building up
surface oxygen that is driven off as volatile COx by the subsequent ion.

python examples/RIE_O_100.py
cd RIE_O_20eV_R5; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate, etch_mode

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

spec = SimSpec(
    orientation    = "100",
    surface        = "2x1",
    surface_temperature    = 300.0,

    species        = "O",
    energy         = 20.0,      # eV
    ion_angle          = 0.0,

    fluence        = 20,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 3,

    flux_ratio     = 5,         # 5 O• radicals at 0.2 eV before each O+ impact
    radical_energy = 0.2,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    inter_neutral_time  = 1000.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "RIE_100_2x1_O_20eV_R5",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")   # → "rie-etch"
make_sim(spec, Path("RIE_O_20eV_R5"))
