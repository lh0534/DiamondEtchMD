"""
Multi-ion-etch: stochastic mix of O+ ions at different energies on the C(100)
ether-terminated surface.

Each impact draws an ion species at random according to the specified fractions.
No O• radicals are deposited (flux_ratio=0); this is pure ion bombardment with
a mixed beam.  Which ion struck is recorded in ion_impacts.txt.

Use IonComponent(species, fraction, energy) — fractions need not be normalized,
but the directory name will reflect normalized percentages.

Energy distribution: 60% O+ at 20 eV, 30% O+ at 30 eV, 10% O+ at 50 eV

python examples/MULTI-ION_O_Edist.py
sbatch MULTI_ION_100_Oether_O_Edist/submit
"""

from pathlib import Path
from diamond_etch_md import (
    SimSpec, IonComponent, compute_ml,
    make_sim, validate, etch_mode,
)

nx, ny = 6, 6
ml = compute_ml("100", nx, ny)   # 36 atoms/ML for 6×6 box

# ---------------------------------------------------------------------------
# O+ energy distribution — simulate a non-monoenergetic ion beam
# ---------------------------------------------------------------------------
# Three IonComponent entries with the same species at different energies
# reproduces a stepped energy distribution (e.g. from an ion energy analyser).
# Here: 60% low-energy tail at 20 eV, 30% mid at 30 eV, 10% high at 50 eV.

spec2 = SimSpec(
    orientation    = "100",
    surface        = "O_ether",
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
    box_depth      = 4,         # sized for 50 eV tail

    impact_time         = 1000.0,
    thermalization_time = 500.0,

    wall_hours     = 24,
    account        = "dgraves",
    name           = "MULTI_ION_100_Oether_O_60p_20eV_30p_30eV_10p_50eV",
)

validate(spec2)
print(f"etch_mode = {etch_mode(spec2)}")    # → "multi-ion-etch"
make_sim(spec2, Path("MULTI_ION_100_Oether_O_Edist"))
