"""
Burst-RIE: 200 eV Ar+ on C(100) 2x1, with O• radicals.


python RIE_Ar_100.py
cd RIE_Ar_100; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate, etch_mode

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)           # 64 atoms/ML for 8x8 box

spec = SimSpec(
    orientation         = "100",
    surface             = "2x1",
    surface_temperature = 300.0,

    species             = "Ar",
    energy              = 50.0,         # eV
    ion_angle           = 0.0,

    fluence             = 1,
    ml                  = ml,
    box_x               = nx,
    box_y               = ny,
    box_depth           = 4,

    flux_ratio          = 5,             # 5 O• radicals per ion impact
    radical_temperature = 1500,          # K
    radical_angle_distribution = True,   # Lambert cosine polar angles
    radical_i_above     = 6.0,           # Å above surface to inject radical
    skip_radical_thermalization = True,  # no thermalization between chunks


    impact_time         = 500.0,
    thermalization_time = 500.0,
    max_inter_neutral_time  = 2000.0,

    dump_mode           = "all",

    wall_hours          = 6,
    nice                = 1,
    account             = "dgraves",
    name                = "RIE_Ar_100_50eV_FR5",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")
make_sim(spec, Path("RIE_Ar_100"))
