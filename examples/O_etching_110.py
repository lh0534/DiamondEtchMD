"""
O radical etching of C(110) surface — bare and O-terminated variants.

C(110) lattice orientation: z=[110], x=[-110], y=[001].
ML factor = 4, so a 4×6 box has 96 atoms/ML.

Surface keys:
  ""  — unterminated (bare) C(110)
  "O" — O-terminated C(110)

box_depth=5 is the recommended starting point for low-energy O radicals
on C(110); the slab is slightly thicker than C(100) due to the open-channel
geometry along [110].

    python examples/O_etching_110.py
    sbatch O_etching_110_bare/submit
    sbatch O_etching_110_O/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate

# C(110) default box: 4×6 lattice units, 96 atoms/ML
ml = compute_ml("110", 4, 6)   # 96

# ---------------------------------------------------------------------------
# Bare C(110) surface
# ---------------------------------------------------------------------------

spec_bare = SimSpec(
    orientation = "110",
    surface     = "",              # unterminated C(110)
    temperature = 300.0,

    species     = "O",
    energy      = 0.5,             # eV — thermal radical
    angle       = 0.0,

    fluence     = 50,
    ml          = ml,
    box_x       = 4,
    box_y       = 6,
    box_depth   = 5,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours  = 24,
    account     = "dgraves",
    name        = "110_bare_O_0.5eV",
)

validate(spec_bare)
make_sim(spec_bare, Path("O_etching_110_bare"))

# ---------------------------------------------------------------------------
# O-terminated C(110) surface
# ---------------------------------------------------------------------------

spec_O = SimSpec(
    orientation = "110",
    surface     = "O",             # O-terminated C(110)
    temperature = 300.0,

    species     = "O",
    energy      = 0.5,
    angle       = 0.0,

    fluence     = 50,
    ml          = ml,
    box_x       = 4,
    box_y       = 6,
    box_depth   = 5,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours  = 24,
    account     = "dgraves",
    name        = "110_O_O_0.5eV",
)

validate(spec_O)
make_sim(spec_O, Path("O_etching_110_O"))
