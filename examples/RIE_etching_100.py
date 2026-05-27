"""
RIE-etch simulation: O+ ions with O• radical pre-exposure on C(100) 1x1 surface.

RIE-etch mode (flux_ratio > 0) deposits flux_ratio O• radicals (type 3) before
each ion impact.  The radical loop uses the same 5-column ncarbon.txt format as
cycle-etch, enabling mid-loop restarts after a wall-time preemption.

This example uses:
  - O+ ions at 20 eV
  - 5 O• radicals at 0.2 eV deposited before each O+ impact (flux_ratio=5)
  - C(100) 1×1 surface, 9×9 box

    python examples/RIE_etching_100.py
    sbatch RIE_O20eV_R5/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate, etch_mode

ml = compute_ml("100", 9, 9)   # 81 atoms/ML for 9×9 box

spec = SimSpec(
    orientation    = "100",
    surface        = "1x1",
    temperature    = 300.0,

    species        = "O",
    energy         = 20.0,      # eV  (O+ ion)
    angle          = 0.0,

    fluence        = 50,        # 50 ML total ion impacts
    ml             = ml,
    box_x          = 9,
    box_y          = 9,
    box_depth      = 5,         # adequate for 20 eV O+

    # RIE-etch parameters: 5 O• radicals at 0.2 eV before each O+ impact
    flux_ratio     = 5,
    radical_energy = 0.2,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    inter_neutral_time  = 1000.0,   # fs for each O• radical impact window

    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_1x1_O20eV_R5_RIE",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")   # → "rie-etch"
make_sim(spec, Path("RIE_O20eV_R5"))
