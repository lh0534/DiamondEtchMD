"""
MASK_RIE-etch:  O+ ions with O• radicals injected in a restricted
                impact-region on C(100) 2x1 surface.

python examples/MASK_RIE_O_100.py
cd MASK_RIE_O_20eV_R5; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate, etch_mode

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)                          # 64 atoms/ML for 8×8 box

spec = SimSpec(
    orientation                 = "100",
    surface                     = "2x1",
    surface_temperature         = 300.0,
    mask_type                   = "xymask",                 # mask is a box in the center of the surface
    mask_width                  = 0.3,               # fraction of box width that is masked from species injection

    species                     = "O",
    energy                      = 20.0,                     # eV
    ion_angle                   = 0.0,

    fluence                     = 50,
    ml                          = ml,
    box_x                       = nx,
    box_y                       = ny,
    box_depth                   = 5,

    flux_ratio                  = 5,                        # 5 O• radicals at 0.2 eV before each O+ impact
    radical_energy              = 0.2,
    radical_burst               = True,                     # burst mode
    radical_burst_chunk         = 5,
    radical_burst_attempt       = 100,
    skip_radical_thermalization = True,

    impact_time                 = 1000.0,
    thermalization_time         = 500.0,
    inter_neutral_time          = 1500.0,

    dump_mode                   = "all",

    wall_hours                  = 12,
    nice                        = 2,
    account                     = "dgraves",
    name                        = "maskRIE_O_20eV_R5burst",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")
make_sim(spec, Path("MASK_RIE_O_20eV_R5"))
