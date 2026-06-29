"""
RIE-etch: Ar+ ions with Maxwell-Boltzmann O• radicals on C(100) 2x1 surface.

Radical speeds are sampled from a 500 K Maxwell-Boltzmann distribution and
angles from a Lambert cosine (flux-weighted) distribution, both on-the-fly
during the LAMMPS simulation.  Only impacts that produce an etch event or
channeled atom keep their trajectory dump (dump_mode="etch_only").

python examples/boltzmann_radical_rie/RIE_O_100_boltzmannRads.py
sbatch boltzmann_radical_rie/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate, etch_mode

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)               # 64 atoms/ML for 8×8 box

spec = SimSpec(
    orientation    = "100",
    surface        = "2x1",
    surface_temperature = 300.0,

    species        = "Ar",
    energy         = 100.0,                  # eV
    ion_angle      = 0.0,

    fluence        = 20,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 6,

    flux_ratio          = 10,                # 10 O• radicals before each Ar+ impact
    radical_temperature = 500.0,             # K — Maxwell-Boltzmann speed distribution
    radical_angle_distribution  = True,      # Lambert cosine polar angles
    radical_i_above    = 6.0,                # Å above surface to inject radical
    max_inter_neutral_time = 5000.0,         # fs — cap on per-radical halt time

    dump_mode           = "all",             # dump all impacts, not just etch events

    impact_time         = 1000.0,
    thermalization_time = 500.0,

    wall_hours     = 24,
    account        = "dgraves",
    name           = "RIE_O_100_boltzRads",
    email          = "",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")   # → "rie-etch"
make_sim(spec, Path("RIE_O_100_boltzRads"))
