"""
High-energy O+ ion bombardment of C(100) 1x1 surface.

At higher energies the ion penetrates deeper into the lattice, so box_depth
must be increased. Empirical guidance:
    <= 20 eV  -> box_depth = 5
       50 eV  -> box_depth = 6
      100 eV  -> box_depth = 10
      200 eV  -> box_depth = 12

    python examples/high_energy_O_100.py
    sbatch high_energy_O_100/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "100",
    surface        = "1x1",
    temperature    = 300.0,

    species        = "O",
    energy         = 200.0,        # eV — needs deep slab
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("100", 9, 9),
    box_x          = 9,
    box_y          = 9,
    box_depth      = 12,           # deep slab for 200 eV

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_1x1_O_200eV",
)

make_sim(spec, Path("high_energy_O_100"))
