"""
O radical etching of C(111) Pandey-chain reconstructed, O-terminated surface.

The surface key "2x1_pandey_O" encodes both the reconstruction (Pandey chain)
and the O termination. Valid 111 surface keys:
  1x1, 2x1_single, 2x1_pandey, 1x1_O, 2x1_single_O, 2x1_pandey_O

    python examples/O_terminated_111_pandey.py
    sbatch O_terminated_111_pandey/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "111",
    surface        = "2x1_pandey_O",
    temperature    = 300.0,

    species        = "O",
    energy         = 1.0,          # eV
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("111", 5, 9),
    box_x          = 5,
    box_y          = 9,
    box_depth      = 3,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "111_pandey_O_1eV",
)

make_sim(spec, Path("O_terminated_111_pandey"))
