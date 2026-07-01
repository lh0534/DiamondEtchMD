"""
Cycle-etch (ALE): Ar+ physical sputtering alternating with O2+ oxidation on C(100) O-ether surface.

O2+ at 2 eV per atom is near-threshold — it modifies the surface layer without deep sputtering.
The Ar+ phase then removes the modified layer, giving quasi-atomic-layer precision.

python examples/CYCLE_Ar_O2_100_ALE.py
sbatch cycling_Ar_O2_ALE/submit
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
            species      = "Ar",
            energy       = 30.0,      # eV
            fluence_ml   = 20,        # 20 ML Ar+ per cycle
            flux_ratio   = 0,         # no radicals during Ar phase
        ),
        CyclePhase(
            species        = "O2",
            energy         = 2.0,     # eV total dimer (1 eV per O atom)
            fluence_ml     = 2,       # 2 ML O2+ per cycle
            flux_ratio     = 5,       # 5 O• at 0.2 eV before each O2+ impact
            radical_energy = 0.2,
        ),
    ],
    cycles     = 3,
    wall_hours = 24,
    account    = "dgraves",
    name       = "CYCLE_100_Oether_Ar_30eV_O2_2eV_R5",
)

validate(spec)
make_sim(spec, Path("cycling_Ar_O2_ALE"))
