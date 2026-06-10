"""
RIE-etch simulation: O+ ions with O• radical pre-exposure on C(100) 1x1 surface.

RIE-etch mode (flux_ratio > 0) deposits flux_ratio O• radicals (type 3) before
each ion impact.  The radical loop uses the same 5-column ncarbon.txt format as
cycle-etch, enabling mid-loop restarts after a wall-time preemption.

python examples/RIE_O_100.py
sbatch RIE_O_20eV_R5/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate, etch_mode

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

spec = SimSpec(
    orientation    = "100",
    surface        = "2x1",
    temperature    = 300.0,

    species        = "O",
    energy         = 20.0,      # eV  (O+ ion)
    angle          = 0.0,

    fluence        = 20,        # 20 ML total ion impacts
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 3,         # adequate for 20 eV O+

    # RIE-etch parameters: 5 O• radicals at 0.2 eV before each O+ impact
    flux_ratio     = 5,
    radical_energy = 0.2,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    inter_neutral_time  = 1000.0,   # fs for each O• radical impact window

    wall_hours     = 24,
    account        = "dgraves",
    name           = "RIE_100_2x1_O_20eV_R5",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")   # → "rie-etch"
make_sim(spec, Path("RIE_O_20eV_R5"))
