"""
Multi-ion-etch: stochastic mix of O+ and O2+ ions on C(100) ether-terminated surface.

Each impact draws an ion species at random according to the specified fractions.
No O• radicals are deposited (flux_ratio=0); this is pure ion bombardment with
a mixed beam.  Which ion struck is recorded in ion_impacts.txt.

Use IonComponent(species, fraction, energy) — fractions need not be normalized,
but the directory name will reflect normalized percentages.

50% O+ at 50 eV  +  50% O2+ at 50 eV (equal-energy mix)

python examples/MULTI-ION_O_O2.py
sbatch MULTI_ION_100_Oether_O_O2/submit
"""

from pathlib import Path
from diamond_etch_md import (
    SimSpec, IonComponent, compute_ml,
    make_sim, validate, etch_mode,
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
    surface        = "O_ether",
    temperature    = 300.0,

    ion_mix = [
        IonComponent(species="O",  fraction=0.5, energy=50.0),
        IonComponent(species="O2", fraction=0.5, energy=50.0),  # 25 eV per O atom
    ],

    fluence        = 50,        # ML total ion impacts
    ml             = ml,
    box_x          = nx,
    box_y          = ny,
    box_depth      = 4,         # sized for 50 eV O+

    impact_time         = 1000.0,   # 1 eV O+ moves at ~0.035 Å/fs
    thermalization_time = 500.0,

    wall_hours     = 24,
    account        = "dgraves",
    name           = "MULTI_ION_100_Oether_O_50eV_O2_50p_50eV",
)

validate(spec1)
print(f"etch_mode = {etch_mode(spec1)}")    # → "multi-ion-etch"
make_sim(spec1, Path("MULTI_ION_100_Oether_O_O2"))
