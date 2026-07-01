"""
Multi-ion-etch: O+ beam with trimodal energy distribution on C(100) O-ether surface.

Three IonComponent entries with the same species at different energies sample a
trimodal beam (three delta functions): 60% at 20 eV, 30% at 30 eV, 10% at 50 eV.

python examples/MULTI-ION_O_Edist.py
sbatch MULTI_ION_100_Oether_O_Edist/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, IonComponent, compute_ml, make_sim, validate, etch_mode

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

spec = SimSpec(
    orientation    = "100",
    surface        = "O_ether",
    surface_temperature    = 300.0,

    ion_mix = [
        IonComponent(species="O", fraction=0.60, energy=20.0),
        IonComponent(species="O", fraction=0.30, energy=30.0),
        IonComponent(species="O", fraction=0.10, energy=50.0),
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
    name           = "MULTI_ION_100_Oether_O_60p_20eV_30p_30eV_10p_50eV",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")    # → "multi-ion-etch"
make_sim(spec, Path("MULTI_ION_100_Oether_O_Edist"))
