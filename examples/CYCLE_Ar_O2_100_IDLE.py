"""
Cycle-etch simulation: Ar+ physical sputtering alternating with
sub-threshold O2+ oxidation on C(100) O-ether surface (IDLE mode).

O2+ at 12 eV is below the sputtering threshold — it oxidises the surface
without removing carbon.  The Ar+ phase then strips the oxide.  Unlike ALE,
the O2+ fluence is longer (30 ML) to ensure full monolayer oxidation.

Ar+ → O2+ (2-phase IDLE cycle)

python examples/CYCLE_Ar_O2_100_IDLE.py
cd cycling_Ar_O2_IDLE; sbatch submit
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
            species    = "Ar",
            energy     = 30.0,    # eV — above sputtering threshold
            fluence_ml = 20,      # 20 ML Ar+ per cycle
            flux_ratio = 0,       # no radicals during Ar phase
        ),
        CyclePhase(
            species        = "O2",
            energy         = 12.0,   # eV total dimer (6 eV per O atom) — sub-threshold
            fluence_ml     = 30,     # 30 ML O2+ to fully oxidise surface
            flux_ratio     = 1,      # 1 O• at 0.2 eV before each O2+ impact
            radical_energy = 0.2,
        ),
    ],
    cycles     = 3,
    wall_hours = 48,
    account    = "dgraves",
    name       = "CYCLE_100_Oether_Ar_30eV_O2_12eV_R1",
)

validate(spec)
make_sim(spec, Path("cycling_Ar_O2_IDLE"))
