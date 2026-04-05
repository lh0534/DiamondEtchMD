"""
O radical etching of C(111) Pandey-chain reconstructed, O-terminated surface.

Demonstrates the reconstruction/termination coupling on 111 surfaces:
each reconstruction has its own O termination name that encodes which
reconstruction it belongs to. For Pandey chains, the termination is
"O_2x1_pandey" (not just "O").

Valid 111 combinations:
    bare_1x1        -> bare, O_1x1
    bare_2x1_single -> bare, O_2x1_single
    bare_2x1_pandey -> bare, O_2x1_pandey

    python examples/O_terminated_111_pandey.py
    sbatch O_terminated_111_pandey/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "111",
    reconstruction = "bare_2x1_pandey",
    termination    = "O_2x1_pandey",   # must match the reconstruction
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
