"""
Cycle-etch (and ALE-etch) simulation: Ar+ physical sputtering alternating with
O2+ chemical etching on C(100) O-ether surface.

Each cycle runs 5 ML of Ar+ (30 eV) followed by 5 ML of O2+ (20 eV) with
10 O• radicals per O2+ ion impact. Ten cycles = 100 ML total fluence.

Ar+ breaks surface bonds and roughens the lattice while O2+ (with O• pre-exposure)
converts sp3 carbon to volatile COx.

Three cycle-etch examples are included:
  1. Ar+ → O2+ (2-phase; also demonstrates ALE-etch factory make_ale())
  2. O+ → O2+ (2-phase, no Ar, faster plain ReaxFF)
  3. Ar+ → O+ → O2+ (3-phase, full reactive-ion cycle)

    python examples/cycling_Ar_O2_100.py
    sbatch cycling_Ar_O2/submit
    sbatch cycling_O_O2/submit
    sbatch cycling_3phase/submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, validate

ml = compute_ml("100", 8, 8)   # 64 atoms/ML for 8×8 box

# ---------------------------------------------------------------------------
# Example 1: Ar+ → O2+ cycling (physical sputtering + chemical etching)
# ---------------------------------------------------------------------------
# Ar+ is inert — uses hybrid ReaxFF+ZBL pair style. No radicals during Ar phase.
# O2+ phase deposits 10 O• radicals at 0.2 eV before each O2+ impact (flux_ratio=10).
# Plain ReaxFF is NOT used here because Ar is present.

spec1 = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    temperature = 300.0,

    ml          = ml,
    box_x       = 8,
    box_y       = 8,
    box_depth   = 5,           # 5 = adequate for ≤30 eV Ar+

    phases = [
        CyclePhase(
            species      = "Ar",
            energy       = 30.0,     # eV
            fluence_ml   = 5,        # 5 ML Ar+ per cycle
            flux_ratio   = 0,        # no radicals during Ar phase
        ),
        CyclePhase(
            species      = "O2",
            energy       = 20.0,     # eV total dimer (10 eV per O atom)
            fluence_ml   = 5,        # 5 ML O2+ per cycle
            flux_ratio   = 10,       # 10 O• at 0.2 eV before each O2+ impact
            radical_energy = 0.2,
        ),
    ],
    cycles     = 10,           # 10 × (5+5) ML = 100 ML total
    wall_hours = 48,
    account    = "dgraves",
    name       = "100_Oether_Ar30eV_O2_20eV_R10_x10",
)

validate(spec1)
make_sim(spec1, Path("cycling_Ar_O2"))

# ---------------------------------------------------------------------------
# Example 2: O+ → O2+ cycling (no Ar → faster plain ReaxFF potential)
# ---------------------------------------------------------------------------
# When no phase uses Ar, the simulation uses 3-atom-type plain ReaxFF —
# faster than the 4-type ZBL hybrid.
# O+ at low energy passivates the surface; O2+ drives chemical etching.

spec2 = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    temperature = 300.0,

    ml          = ml,
    box_x       = 8,
    box_y       = 8,
    box_depth   = 3,           # low-energy O+ — shallow depth is fine

    phases = [
        CyclePhase(
            species    = "O",
            energy     = 1.0,
            fluence_ml = 5,
            flux_ratio = 0,
        ),
        CyclePhase(
            species        = "O2",
            energy         = 20.0,
            fluence_ml     = 5,
            flux_ratio     = 10,
            radical_energy = 0.2,
        ),
    ],
    cycles     = 10,
    wall_hours = 48,
    account    = "dgraves",
    name       = "100_Oether_O1eV_O2_20eV_R10_x10",
)

validate(spec2)
make_sim(spec2, Path("cycling_O_O2"))

# ---------------------------------------------------------------------------
# Example 3: Ar+ → O+ → O2+ three-phase cycling
# ---------------------------------------------------------------------------
# Phase 1 (Ar+): physical sputtering to remove passivation layer.
# Phase 2 (O+):  low-energy oxygen to build up surface O coverage.
# Phase 3 (O2+): energetic O2+ with radical pre-exposure to etch oxidised C.

spec3 = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    temperature = 300.0,

    ml          = ml,
    box_x       = 6,
    box_y       = 6,
    box_depth   = 4,           # 4 for 50 eV Ar+

    phases = [
        CyclePhase(species="Ar", energy=50.0, fluence_ml=5),
        CyclePhase(species="O",  energy=1.0,  fluence_ml=3, flux_ratio=5),
        CyclePhase(species="O2", energy=20.0, fluence_ml=5,
                   flux_ratio=2, radical_energy=0.2),
    ],
    cycles     = 3,
    wall_hours = 48,
    account    = "dgraves",
    name       = "100_Oether_3phase_Ar50_O1_R5_O2_20_R2_x5",
)

validate(spec3)
make_sim(spec3, Path("cycling_3phase"))
