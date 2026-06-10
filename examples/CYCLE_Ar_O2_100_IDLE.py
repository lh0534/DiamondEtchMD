"""
Cycle-etch simulation: Ar+ physical sputtering alternating with
O2+ chemical modification on C(100) O-ether surface.

Each cycle runs 20 ML of Ar+ (30 eV) followed by 2 ML of O2+ (2 eV) with
1 O• radical per O2+ ion impact. Three cycles = 150 ML total fluence.

Ar+ → O2+ (2-phase; also demonstrates ALE-etch factory make_ale())

python examples/CYCLE_Ar_O2_100_ALE.py
sbatch cycling_Ar_O2_ALE/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, validate

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)   # 64 atoms/ML for 8×8 box

# ---------------------------------------------------------------------------
# Ar+ → O2+ cycling
# ---------------------------------------------------------------------------

spec1 = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    temperature = 300.0,

    ml          = ml,
    box_x       = nx,
    box_y       = ny,
    box_depth   = 3,

    phases = [
        CyclePhase(
            species      = "Ar",
            energy       = 30.0,      # eV
            fluence_ml   = 20,        # 20 ML Ar+ per cycle
            flux_ratio   = 0,         # no radicals during Ar phase
        ),
        CyclePhase(
            species      = "O2",
            energy       = 2.0,       # eV total dimer (1 eV per O atom)
            fluence_ml   = 2,         # 2 ML O2+ per cycle
            flux_ratio   = 5,         # 5 O• at 0.2 eV before each O2+ impact
            radical_energy = 0.2,
        ),
    ],
    cycles     = 3,
    wall_hours = 24,
    account    = "dgraves",
    name       = "100_Oether_Ar_30eV_O2_2eV_R5",
)

validate(spec1)
make_sim(spec1, Path("cycling_Ar_O2_ALE"))
