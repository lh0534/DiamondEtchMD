"""
Cycle-etch: O+ chemical etching alternating with O2+ etching on C(100) O-ether surface.

python examples/CYCLE_O_O2_100.py
cd cycling_O_O2; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, validate

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)   # 64 atoms/ML for 8×8 box

spec = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    surface_temperature = 300.0,

    ml          = ml,
    box_x       = nx,
    box_y       = ny,
    box_depth   = 3,

    phases = [
        CyclePhase(
            species    = "O",
            energy     = 15,
            fluence_ml = 5,
            flux_ratio = 0,
        ),
        CyclePhase(
            species        = "O2",
            energy         = 12.0,
            fluence_ml     = 5,
            flux_ratio     = 2,
            radical_energy = 0.2,
        ),
    ],
    cycles     = 3,
    wall_hours = 72,
    account    = "dgraves",
    name       = "CYCLE_100_Oether_O_15eV_O2_12eV_R2",
)

validate(spec)
make_sim(spec, Path("cycling_O_O2"))
