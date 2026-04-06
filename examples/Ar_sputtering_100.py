"""
Ar+ physical sputtering of C(100) 1x1 surface.

Argon ions are chemically inert — they transfer kinetic energy to the surface
and are deleted after each impact. Uses a hybrid ReaxFF + ZBL pair style.

Typical Ar sputtering energies: 20-200 eV. Higher energies need a deeper slab.

    python examples/Ar_sputtering_100.py
    sbatch Ar_sputtering_100/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "100",
    surface        = "1x1",
    temperature    = 300.0,

    species        = "Ar",
    energy         = 100.0,        # eV
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("100", 9, 9),
    box_x          = 9,
    box_y          = 9,
    box_depth      = 10,           # deep slab for 100 eV Ar

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_1x1_Ar_100eV",
)

make_sim(spec, Path("Ar_sputtering_100"))
