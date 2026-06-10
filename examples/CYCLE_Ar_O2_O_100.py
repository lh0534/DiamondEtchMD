"""
Cycle-etch (and ALE-etch) simulation: Ar+ physical sputtering followed by O2+ chemical etching followed by O+ chemical etching on C(100) O-ether surface.

Each cycle runs 5 ML of Ar+ (30 eV) followed by 5 ML of O2+ (20 eV) with
10 O• radicals per O2+ ion impact. Ten cycles = 100 ML total fluence.

Ar+ → O2+ → O+ (3-phase, full reactive-ion cycle)

python examples/CYCLE_Ar_O2_O_100.py
sbatch cycling_3phase_Ar_O2_O/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, validate

# ---------------------------------------------------------------------------
# Ar+ → O2+ → O+ three-phase cycling
# ---------------------------------------------------------------------------
nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box


spec3 = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    temperature = 300.0,

    ml          = ml,
    box_x       = nx,
    box_y       = ny,
    box_depth   = 4,           # 3 for 30 eV Ar+

    phases = [
        CyclePhase(species="Ar", energy=30.0, fluence_ml=5),
        CyclePhase(species="O2", energy=10.0, fluence_ml=5),
        CyclePhase(species="O",  energy=8.0,  fluence_ml=5),
    ],
    cycles     = 2,
    wall_hours = 72,
    account    = "dgraves",
    name       = "CYCLE3_100_Oether_Ar_30eV_O2_10eV_O_8eV",
)

validate(spec3)
make_sim(spec3, Path("cycling_3phase_Ar_O2_O"))
