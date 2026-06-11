"""
Multi-ion-etch: stochastic 50/50 O+/O2+ mixed beam on C(100) O-ether surface.

Each impact draws an ion at random from the mix; no O• radicals (flux_ratio=0).
O2+ is injected as a dimer — its energy is the total dimer KE (halved per atom).

python examples/MULTI-ION_O_O2.py
sbatch MULTI_ION_100_Oether_O_O2/submit
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
        IonComponent(species="O",  fraction=0.5, energy=50.0),
        IonComponent(species="O2", fraction=0.5, energy=50.0),  # 25 eV per O atom
    ],

    fluence        = 50,
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 4,

    impact_time         = 1000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "MULTI_ION_100_Oether_O_50eV_O2_50p_50eV",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")    # → "multi-ion-etch"
make_sim(spec, Path("MULTI_ION_100_Oether_O_O2"))
