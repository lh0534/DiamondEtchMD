"""
O radical etching of C(100) 2×1 reconstructed, O-ether terminated surface.

This is the most common use case: low-energy oxygen radicals impacting a
reconstructed diamond (100) surface. The 2×1 reconstruction forms dimer rows
along [110]; O_ether bridges O atoms between adjacent surface C.

    python examples/O_radical_100.py
    sbatch O_radical_100/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "100",
    reconstruction = "bare_2x1",
    termination    = "O_ether",
    temperature    = 300.0,

    species        = "O",
    energy         = 0.5,          # eV — thermal radical energy
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("100", 9, 9),
    box_x          = 9,
    box_y          = 9,
    box_depth      = 3,            # shallow slab is fine for sub-eV radicals

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_2x1_Oether_O_0.5eV",
)

make_sim(spec, Path("O_radical_100"))
