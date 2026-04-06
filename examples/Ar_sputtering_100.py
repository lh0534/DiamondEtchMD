"""
Ar+ physical sputtering of bare C(100).

Argon ions are chemically inert — they transfer kinetic energy to the surface
and are deleted after each impact. Uses a hybrid ReaxFF + ZBL pair style for
the short-range nuclear repulsion between Ar and C/H/O atoms.

Typical Ar sputtering energies: 20–200 eV. Higher energies need a deeper slab
(box_depth) so the ion stops within the mobile region.

    python examples/Ar_sputtering_100.py
    sbatch Ar_sputtering_100/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "100",
    surface         = "1x1",
                # no termination (bare surface for physical sputtering)
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
    name           = "100_bare_Ar_100eV",
)

make_sim(spec, Path("Ar_sputtering_100"))
