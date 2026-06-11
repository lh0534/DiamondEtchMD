"""
Multi-RIE-etch: stochastic O+/O2+ mixed beam with O• radical co-exposure.

Before each ion impact, flux_ratio O• radicals are deposited; then one ion is
drawn at random from the mix.

python examples/MULTI-RIE_O_O2.py
sbatch MULTI_RIE_100_Oether_O_O2/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, IonComponent, compute_ml, make_sim, validate, etch_mode

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

spec = SimSpec(
    orientation    = "100",
    surface        = "O_ether",
    temperature    = 300.0,

    ion_mix = [
        IonComponent(species="O",  fraction=0.5, energy=20.0),
        IonComponent(species="O2", fraction=0.5, energy=40.0),  # 20 eV per O atom
    ],

    flux_ratio     = 2,             # 2 O• radicals at 0.2 eV before each ion impact
    radical_energy = 0.2,

    fluence        = 20,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 4,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    inter_neutral_time  = 1000.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "MULTI_RIE_100_Oether_O_50p_20eV_O2_50p_40eV_R5",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")    # → "multi-rie-etch"
make_sim(spec, Path("MULTI_RIE_100_Oether_O_O2"))
