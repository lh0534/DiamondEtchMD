"""
Single-impact statistics: 50 eV Ar+ on graphullerene at normal incidence.

Thermalizes the graphullerene to 300 K once, then repeats a single Ar+ impact
100 times — each trial reloads the same thermalized surface and draws fresh
thermal velocities.  Outputs:

  impact_stats.txt  — per-trial: surface_z_before, ion_z_after, ion_in_box,
                       penetration depth (Å)
  etch_products.txt — ejected clusters (composition + exit vcm_z)
  ncarbon.txt       — per-trial atom counts
  impact_snaps/     — post-impact snapshot after each trial (${trial}_0.data)
  etch_event_trajs/ — ion trajectory dumps for trials that etch (etch_only mode)

python examples/SINGLE_IMPACT_graphullerene_Ar.py
cd graphullerene_Ar_50eV_single; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, validate, make_sim

config = Path(__file__).parent.parent.parent / "graphullerene" / "graphullerene_start.dat"

spec = SimSpec(
    name                          = "balls_Ar_50eV_single",

    # Starting configuration
    initial_config_file           = str(config),
    anchor_z_max                  = 7.0,           # Å; freeze the bottom layer
    initial_thermalization        = True,          # NVT equilibration before trials
    initial_thermalization_steps  = 300_000,

    # Ion beam
    species                       = "Ar",
    energy                        = 50.0,          # eV
    ion_angle                     = 0.0,           # degrees from surface normal

    # Single-impact statistics mode
    single_impact                 = True,
    n_trials                      = 100,           # independent impact events
    randomize_velocities          = True,          # re-draw thermal velocities each trial

    # Thermodynamics
    surface_temperature           = 300.0,         # K
    impact_time                   = 1000.0,        # fs per impact event
    thermalization_time           = 500.0,         # fs post-impact thermalization

    # Output
    dump_mode                     = "all",         # save trajectories only for etching events

    # SLURM
    wall_hours                    = 1,
    account                       = "dgraves",
)

validate(spec)
make_sim(spec, Path("balls_Ar_50eV_single"))
