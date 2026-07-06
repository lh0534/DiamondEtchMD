"""
Ar+ physical sputtering of a graphullerene structure.

Reads an existing 0 K graphullerene config, thermalizes it to 300 K,
then etches with 20 eV Ar+ at 45° incidence for 2 ML of fluence.
Monolayer count is computed automatically from the box XY area.

python examples/CARBON_graphullerene_Ar.py
cd graphullerene_Ar; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import (SimSpec, validate, make_sim,
                              parse_data_file_box, compute_ml_langmuir)

config = Path(__file__).parent.parent.parent / "graphullerene" / "graphullerene_start.dat"

lx, ly = parse_data_file_box(str(config))
ml = compute_ml_langmuir(lx, ly)

spec = SimSpec(
    name                   = "graphullerene_Ar",

    initial_config_file    = str(config),
    anchor_z_max           = 7.0,          # Å; freeze bottom layer
    initial_thermalization = True,         # 10 000-step NVT equil before impacts
    initial_thermalization_steps = 10000000,

    species                = "Ar",
    energy                 = 20.0,         # eV
    ion_angle              = 45.0,         # degrees

    fluence                = 2,            # ML
    ml                     = ml,

    surface_temperature    = 300.0,        # K
    impact_time            = 1000.0,
    thermalization_time    = 500.0,
    wall_hours             = 1,
    account                = "dgraves",
)

validate(spec)
make_sim(spec, Path("graphullerene_Ar"))
