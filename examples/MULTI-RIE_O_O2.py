"""
Multi-RIE-etch: stochastic O+/O2+ mixed beam with O• radical pre-exposure.

Combines the mixed-ion stochastic selection of multi-ion-etch with the radical
flooding of RIE-etch.  Before each ion impact, flux_ratio O• radicals are
deposited; then one ion is drawn at random from the mix.

The output directory name encodes both the ion mix and the flux ratio:
RIE_{surf}_{ion0}_{pct0}p_{e0}eV_{ion1}_{pct1}p_{e1}eV_R{FR}

50% O+ at 50 eV  +  50% O2+ at 100 eV, 5 O• radicals per impact

python examples/multi_RIE_etch.py
sbatch MULTI_RIE_100_Oether_O_50p_20eV_O2_50p_40eV_R5/submit
"""

from pathlib import Path
from diamond_etch_md import (
    SimSpec, IonComponent, compute_ml,
    make_sim, validate, etch_mode,
)

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

# ---------------------------------------------------------------------------
# 50% O+ at 50 eV + 50% O2+ at 100 eV, 5 O• radicals per impact
# ---------------------------------------------------------------------------

spec1 = SimSpec(
    orientation    = "100",
    surface        = "O_ether",
    temperature    = 300.0,

    ion_mix = [
        IonComponent(species="O",  fraction=0.5, energy=20.0),
        IonComponent(species="O2", fraction=0.5, energy=40.0),  # 20 eV per O atom
    ],

    flux_ratio     = 2,             # 2 O• radicals at 0.2 eV before each ion impact
    radical_energy = 0.2,           # eV

    fluence        = 20,            # ML total ion impacts
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 4,

    impact_time         = 1000.0,   # 1 eV O+ moves at ~0.035 Å/fs
    thermalization_time = 500.0,
    inter_neutral_time  = 1000.0,   # fs for each O• radical impact window (0.2 eV O• reaches surface in ~385 fs)

    wall_hours     = 24,
    account        = "dgraves",
    name           = "MULTI_RIE_100_Oether_O_50p_20eV_O2_50p_40eV_R5",
)

validate(spec1)
print(f"etch_mode = {etch_mode(spec1)}")    # → "multi-rie-etch"
make_sim(spec1, Path("MULTI_RIE_100_Oether_O_50p_20eV_O2_50p_40eV_R2"))
