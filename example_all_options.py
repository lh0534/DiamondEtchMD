"""
example_all_options.py — full SimSpec example showing every available option.

Run from the DiamondEtchMD directory:
    python example_all_options.py

The generated simulation directory will be created at OUTDIR below.
All surface templates and force-field files are bundled with the package;
no external data directory is needed.
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

# Output directory for the generated simulation
OUTDIR = Path("my_sim_full_example")

# ---------------------------------------------------------------------------
# Build the spec with every option explicitly set
# ---------------------------------------------------------------------------

spec = SimSpec(
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
    species        = "O",          # "O"
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

# ---------------------------------------------------------------------------
# Generate the simulation directory
# ---------------------------------------------------------------------------

make_sim(spec, OUTDIR)

# To submit:
#   sbatch my_sim_full_example/submit
