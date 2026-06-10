"""
Multi-RIE-etch: stochastic O+/O2+ mixed beam with O• radical pre-exposure.

Combines the mixed-ion stochastic selection of multi-ion-etch with the radical
flooding of RIE-etch.  Before each ion impact, flux_ratio O• radicals are
deposited; then one ion is drawn at random from the mix.

The output directory name encodes both the ion mix and the flux ratio:
  RIE_{ion0}_{pct0}p_{e0}eV_{ion1}_{pct1}p_{e1}eV_R{FR}

Two examples are included:
  1. 50% O+ at 50 eV  +  50% O2+ at 100 eV, 5 O• radicals per impact
  2. Ar+/O2+ sputtering mix with O• radicals (requires ZBL hybrid potential)

    python examples/multi_RIE_etch.py
    sbatch RIE_O_50p_50eV_O2_50p_100eV_R5/submit
    sbatch RIE_Ar_30p_100eV_O2_70p_50eV_R3/submit
"""

from pathlib import Path
from diamond_etch_md import (
    SimSpec, IonComponent, compute_ml,
    make_sim, validate, etch_mode, multi_ion_dir_name,
)

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

# ---------------------------------------------------------------------------
# Example 1: 50% O+ at 50 eV + 50% O2+ at 100 eV, 5 O• radicals per impact
# ---------------------------------------------------------------------------
# Radicals use radical_energy regardless of which ion was selected.
# O2+ energy is the total dimer KE (100 eV total → 50 eV per O atom).

spec1 = SimSpec(
    orientation    = "100",
    surface        = "1x1",
    temperature    = 300.0,

    ion_mix = [
        IonComponent(species="O",  fraction=0.5, energy=50.0),
        IonComponent(species="O2", fraction=0.5, energy=100.0),  # 50 eV per O atom
    ],

    flux_ratio     = 5,         # 5 O• radicals at 0.2 eV before each ion impact
    radical_energy = 0.2,       # eV  (0.2 eV O• reaches surface in ~240 fs)

    fluence        = 50,        # ML total ion impacts
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 6,         # adequate for 100 eV O2+ (50 eV/atom)

    impact_time         = 2000.0,   # 1 eV O+ moves at ~0.035 Å/fs
    thermalization_time = 500.0,
    inter_neutral_time  = 1000.0,   # fs for each O• radical impact window

    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_1x1_O50_O2_100eV_R5_multi_RIE",
)

validate(spec1)
print(f"etch_mode = {etch_mode(spec1)}")    # → "multi-rie-etch"
print(f"outdir    = {multi_ion_dir_name(spec1)}")
make_sim(spec1, Path(multi_ion_dir_name(spec1)))

# ---------------------------------------------------------------------------
# Example 2: Ar+/O2+ mix with O• radicals
# ---------------------------------------------------------------------------
# Ar requires the ZBL hybrid pair style — the generator detects this
# automatically when any IonComponent has species="Ar".
# O• radicals still use plain ReaxFF (type 3); only the Ar impact events
# trigger the ZBL correction.

spec2 = SimSpec(
    orientation    = "100",
    surface        = "1x1",
    temperature    = 300.0,

    ion_mix = [
        IonComponent(species="Ar", fraction=0.30, energy=100.0),
        IonComponent(species="O2", fraction=0.70, energy=50.0),  # 25 eV per O atom
    ],

    flux_ratio     = 3,
    radical_energy = 0.2,

    fluence        = 50,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 10,        # sized for 100 eV Ar+ tail

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    inter_neutral_time  = 1000.0,

    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_1x1_Ar100_O2_50eV_R3_multi_RIE",
)

validate(spec2)
print(f"etch_mode = {etch_mode(spec2)}")    # → "multi-rie-etch"
print(f"outdir    = {multi_ion_dir_name(spec2)}")
make_sim(spec2, Path(multi_ion_dir_name(spec2)))

# To submit:
#   sbatch RIE_O_50p_50eV_O2_50p_100eV_R5/submit
#   sbatch RIE_Ar_30p_100eV_O2_70p_50eV_R3/submit
