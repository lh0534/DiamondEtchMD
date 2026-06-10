"""
Cycle-etch simulation: Ar+ physical sputtering alternating with
O2+ chemical modification on C(100) O-ether surface.

Each cycle runs 5 ML of Ar+ (30 eV) followed by 5 ML of O2+ (20 eV) with
10 O• radicals per O2+ ion impact. Three cycles = 30 ML total fluence.

Ar+ → O2+ (2-phase; also demonstrates ALE-etch factory make_ale())

python examples/CYCLE_Ar_O2_100_IDLE.py
sbatch cycling_Ar_O2_IDLE/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, validate

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)   # 64 atoms/ML for 8×8 box

# ---------------------------------------------------------------------------
# Ar+ → O2+ cycling (physical sputtering + chemical etching)
# ---------------------------------------------------------------------------
# Ar+ is inert — uses hybrid ReaxFF+ZBL pair style. No radicals during Ar phase.
# O2+ phase deposits 10 O• radicals at 0.2 eV before each O2+ impact (flux_ratio=10).
# Plain ReaxFF is NOT used here because Ar is present.

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
            fluence_ml   = 20,        # 5 ML Ar+ per cycle
            flux_ratio   = 0,         # no radicals during Ar phase
        ),
        CyclePhase(
            species      = "O2",
            energy       = 12.0,      # eV total dimer (6 eV per O atom)
            fluence_ml   = 30,        # 30 ML O2+ per cycle
            flux_ratio   = 1,         # 1 O• at 0.2 eV before each O2+ impact
            radical_energy = 0.2,
        ),
    ],
    cycles     = 3,
    wall_hours = 48,
    account    = "dgraves",
    name       = "100_Oether_Ar30eV_O2_12eV_R1",
)

validate(spec1)
make_sim(spec1, Path("cycling_Ar_O2_IDLE"))
