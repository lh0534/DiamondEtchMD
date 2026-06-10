"""
Multi-ion-etch: stochastic mix of O+ and O2+ ions on C(100) 1x1 surface.

Each impact draws an ion species at random according to the specified fractions.
No O• radicals are deposited (flux_ratio=0); this is pure ion bombardment with
a mixed beam.  Which ion struck is recorded in ion_impacts.txt.

Use IonComponent(species, fraction, energy) — fractions need not be normalized,
but the directory name will reflect normalized percentages.

Two examples are included:
  1. 50% O+ at 50 eV  +  50% O2+ at 50 eV (equal-energy mix)
  2. Energy distribution: 60% O+ at 20 eV, 30% O+ at 30 eV, 10% O+ at 50 eV

    python examples/multi_ion_etch.py
    sbatch ION_O_50p_50eV_O2_50p_50eV/submit
    sbatch ION_O_60p_20eV_O_30p_30eV_O_10p_50eV/submit
"""

from pathlib import Path
from diamond_etch_md import (
    SimSpec, IonComponent, compute_ml,
    make_sim, validate, etch_mode, multi_ion_dir_name,
)

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

# ---------------------------------------------------------------------------
# Example 1: 50/50 O+ / O2+ mixed beam at equal energy
# ---------------------------------------------------------------------------
# O2+ is injected as a dimer; its energy is the total dimer KE (halved per atom).
# No ZBL needed — neither species is Ar.

spec1 = SimSpec(
    orientation    = "100",
    surface        = "1x1",
    temperature    = 300.0,

    ion_mix = [
        IonComponent(species="O",  fraction=0.5, energy=50.0),
        IonComponent(species="O2", fraction=0.5, energy=50.0),  # 25 eV per O atom
    ],

    fluence        = 50,        # ML total ion impacts
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 6,         # adequate for 50 eV O+

    impact_time         = 2000.0,   # 1 eV O+ moves at ~0.035 Å/fs
    thermalization_time = 500.0,

    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_1x1_O50_O2_50eV_50-50_mixed",
)

validate(spec1)
print(f"etch_mode = {etch_mode(spec1)}")    # → "multi-ion-etch"
print(f"outdir    = {multi_ion_dir_name(spec1)}")
make_sim(spec1, Path(multi_ion_dir_name(spec1)))

# ---------------------------------------------------------------------------
# Example 2: O+ energy distribution — simulate a non-monoenergetic ion beam
# ---------------------------------------------------------------------------
# Three IonComponent entries with the same species at different energies
# reproduces a stepped energy distribution (e.g. from an ion energy analyser).
# Here: 60% low-energy tail at 20 eV, 30% mid at 30 eV, 10% high at 50 eV.
# Fractions are automatically normalized so they need not sum to 1.

spec2 = SimSpec(
    orientation    = "100",
    surface        = "1x1",
    temperature    = 300.0,

    ion_mix = [
        IonComponent(species="O", fraction=0.60, energy=20.0),
        IonComponent(species="O", fraction=0.30, energy=30.0),
        IonComponent(species="O", fraction=0.10, energy=50.0),
    ],

    fluence        = 50,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 6,         # sized for 50 eV tail

    impact_time         = 2000.0,
    thermalization_time = 500.0,

    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_1x1_O_20_50_100eV_energy_dist",
)

validate(spec2)
print(f"etch_mode = {etch_mode(spec2)}")    # → "multi-ion-etch"
print(f"outdir    = {multi_ion_dir_name(spec2)}")
make_sim(spec2, Path(multi_ion_dir_name(spec2)))

# To submit:
#   sbatch ION_O_50p_50eV_O2_50p_50eV/submit
#   sbatch ION_O_60p_20eV_O_30p_30eV_O_10p_50eV/submit
