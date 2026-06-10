"""
Multi-RIE-etch: stochastic Ar+/O2+ mixed beam with O• radical pre-exposure.

Combines the mixed-ion stochastic selection of multi-ion-etch with the radical
flooding of RIE-etch. Before each ion impact, flux_ratio O• radicals are
deposited; then one ion is drawn at random from the mix.

The output directory name encodes both the ion mix and the flux ratio:
RIE_{surf}_{ion0}_{pct0}p_{e0}eV_{ion1}_{pct1}p_{e1}eV_R{FR}

Ar+/O2+ sputtering mix with O• radicals (requires ZBL hybrid potential)

python examples/multi_RIE_etch.py
sbatch MULTI_RIE_100_Oether_Ar_O2/submit
"""

from pathlib import Path
from diamond_etch_md import (
    SimSpec, IonComponent, compute_ml,
    make_sim, validate, etch_mode,
)

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

# ---------------------------------------------------------------------------
# Ar+/O2+ mix with O• radicals
# ---------------------------------------------------------------------------

spec2 = SimSpec(
    orientation    = "100",
    surface        = "O_ether",
    temperature    = 300.0,

    ion_mix = [
        IonComponent(species="Ar", fraction=0.30, energy=50.0),
        IonComponent(species="O2", fraction=0.70, energy=50.0),  # 25 eV per O atom
    ],

    flux_ratio     = 2,             # 2 O• radicals at 0.2 eV before each ion impact
    radical_energy = 0.2,

    fluence        = 50,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 5,             # sized for 50 eV Ar+

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    inter_neutral_time  = 1000.0,

    wall_hours     = 24,
    account        = "dgraves",
    name           = "MULTI_RIE_100_Oether_Ar_30p_50eV_O2_70p_50eV_R2",
)

validate(spec2)
print(f"etch_mode = {etch_mode(spec2)}")    # → "multi-rie-etch"
make_sim(spec2, Path("MULTI_RIE_100_Oether_Ar_O2"))
