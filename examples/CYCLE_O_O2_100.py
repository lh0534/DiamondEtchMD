"""
Cycle-etch simulation: O+ chemical etching alternating with
O2+ chemical etching on C(100) O-ether surface.

Each cycle runs 5 ML of O+ (15 eV) followed by 5 ML of O2+ (12 eV) with
2 O• radicals per O2+ ion impact. Three cycles = 30 ML total fluence.

O+ → O2+ (2-phase)

python examples/CYCLE_O_O2_100.py
sbatch cycling_O_O2/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, validate

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)   # 64 atoms/ML for 8×8 box

# ---------------------------------------------------------------------------
# O+ → O2+ cycling
# ---------------------------------------------------------------------------

spec2 = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    temperature = 300.0,

    ml          = ml,
    box_x       = nx,
    box_y       = ny,
    box_depth   = 3,           # low-energy O+ — shallow depth is fine

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

validate(spec2)
make_sim(spec2, Path("cycling_O_O2"))
