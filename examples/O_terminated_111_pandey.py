"""
O radical etching of C(111) Pandey-chain reconstructed, O-terminated surface.

On 111 surfaces, O termination is specified as just "O" regardless of the
reconstruction. The reconstruction field determines which template runs
(each has its own O placement code).

Valid 111 reconstructions: 1x1, 2x1_single, 2x1_pandey

    python examples/O_terminated_111_pandey.py
    sbatch O_terminated_111_pandey/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "111",
    surface         = "2x1_pandey_O",
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
