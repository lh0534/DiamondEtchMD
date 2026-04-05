"""
example_all_options.py — full SimSpec examples showing every available option.

Run from the DiamondEtchMD directory:
    python example_all_options.py

Generates three simulation directories (one per ion species) under examples/.
All surface templates and force-field files are bundled with the package;
no external data directory is needed.
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

OUTDIR = Path("examples")

# ---------------------------------------------------------------------------
# Example 1: O radical bombardment — low energy, O-ether terminated 2×1 surface
# ---------------------------------------------------------------------------

spec_O = SimSpec(
    # --- surface ---
    orientation    = "100",        # "100", "111", or "113"

    reconstruction = "bare_2x1",   # 100: "bare_1x1", "bare_2x1"
                                   # 111: "bare_1x1", "bare_2x1_single", "bare_2x1_pandey"
                                   # 113: "bare"

    termination    = "O_ether",    # 100: "bare", "O", "O_ether"
                                   # 111: "bare", "O_1x1", "O_2x1_single", "O_2x1_pandey"
                                   # 113: "bare", "O"
                                   # termination must be valid for the chosen reconstruction:
                                   #   bare_1x1  → bare, O_1x1        (111)
                                   #   bare_2x1_single → bare, O_2x1_single  (111)
                                   #   bare_2x1_pandey → bare, O_2x1_pandey  (111)

    temperature    = 300.0,        # substrate temperature (K)

    # --- bombardment ---
    species        = "O",          # single oxygen atom (O⁺ ion or O radical)
    energy         = 0.5,          # incident particle energy (eV)
    angle          = 0.0,          # incidence angle from surface normal (degrees)

    # --- simulation size ---
    fluence        = 50,           # total fluence (monolayers)
    box_x          = 9,            # lateral box size, x (lattice units)
    box_y          = 9,            # lateral box size, y (lattice units)
    box_depth      = 3,            # slab depth — lat_top (lattice units)
                                   # recommended: ≤20 eV → 5, 50 eV → 6, 100 eV → 10, 200 eV → 12

    # ml is computed automatically from orientation and box dimensions;
    # override only if you know the exact atom count from a reference data file
    ml             = compute_ml("100", 9, 9),  # 81 for 100, 9×9 box

    # --- timing ---
    impact_time         = 2000.0,  # simulation time per impact event (fs)
    thermalization_time = 500.0,   # NVT thermalisation after each impact (fs)

    # --- SLURM ---
    wall_hours     = 24,           # wall-clock limit (hours)
    account        = "dgraves",    # Della account to charge
    email          = "",           # email for END/FAIL notifications; "" = no mail
    name           = "100_bare_2x1_Oether_O_0.5eV_300K",  # job name (auto-generated if "")
)

make_sim(spec_O, OUTDIR / "O_radical")

# ---------------------------------------------------------------------------
# Example 2: Ar+ ion bombardment — high energy sputtering
# ---------------------------------------------------------------------------

spec_Ar = SimSpec(
    orientation    = "100",
    reconstruction = "bare_1x1",
    termination    = "bare",       # typically bare for physical sputtering
    temperature    = 300.0,

    species        = "Ar",         # argon ion — uses hybrid ReaxFF+ZBL pair style;
                                   # Ar is inert and deleted after each impact
    energy         = 100.0,        # eV — Ar sputtering is typically 20–200 eV
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("100", 9, 9),
    box_x          = 9,
    box_y          = 9,
    box_depth      = 10,           # deeper slab needed for high-energy Ar

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_bare_Ar_100eV_300K",
)

make_sim(spec_Ar, OUTDIR / "Ar_sputtering")

# ---------------------------------------------------------------------------
# Example 3: O2+ ion bombardment — dimer injection
# ---------------------------------------------------------------------------

spec_O2 = SimSpec(
    orientation    = "111",
    reconstruction = "bare_2x1_pandey",
    termination    = "bare",
    temperature    = 300.0,

    species        = "O2",         # oxygen dimer — injected as a LAMMPS molecule;
                                   # energy is the TOTAL dimer KE (each atom gets half)
    energy         = 50.0,         # 50 eV total → 25 eV per O atom
    angle          = 0.0,

    fluence        = 50,
    ml             = compute_ml("111", 5, 9),
    box_x          = 5,
    box_y          = 9,
    box_depth      = 6,

    impact_time         = 2000.0,
    thermalization_time = 500.0,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "111_pandey_O2_50eV_300K",
)

make_sim(spec_O2, OUTDIR / "O2_bombardment")

# To submit any of these:
#   sbatch examples/O_radical/submit
#   sbatch examples/Ar_sputtering/submit
#   sbatch examples/O2_bombardment/submit
