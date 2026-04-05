"""
O2+ ion bombardment of C(111) Pandey-chain reconstructed surface.

O2 is injected as a LAMMPS molecule (two O atoms at 1.2 A separation).
The user-specified energy is the TOTAL dimer kinetic energy; each atom
receives half internally. So 50 eV here means 25 eV per oxygen atom.

    python examples/O2_bombardment_111.py
    sbatch O2_bombardment_111/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "111",
    reconstruction = "bare_2x1_pandey",
    termination    = "bare",
    temperature    = 300.0,

    species        = "O2",
    energy         = 50.0,         # 50 eV total -> 25 eV per O atom
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("111", 5, 9),
    box_x          = 5,
    box_y          = 9,
    box_depth      = 6,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "111_pandey_O2_50eV",
)

make_sim(spec, Path("O2_bombardment_111"))
