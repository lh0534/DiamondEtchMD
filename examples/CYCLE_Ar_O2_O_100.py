"""
3-Phase cycle-etch: Ar+ → O2+ → O+ on C(100) O-ether surface.

python examples/CYCLE_Ar_O2_O_100.py
sbatch cycling_3phase_Ar_O2_O/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, validate

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

spec = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    surface_temperature = 300.0,

    ml          = ml,
    box_x       = nx,
    box_y       = ny,
    box_depth   = 4,

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

validate(spec)
make_sim(spec, Path("cycling_3phase_Ar_O2_O"))
