"""
Burst-RIE: 200 eV Ar+ on C(100) 2x1, with O• radicals injected as a burst.

flux_ratio=150 radicals are injected before each ion impact in chunks of 0.5 ML
(32 atoms each for an 8x8 surface): 4 full chunks of 32 + 1 chunk of 22 = 150 total.
Each chunk is deposited at the same z-height (radical_i_above), then dynamics run
for inter_neutral_time before the next chunk.  One ncarbon.txt entry per burst.

python examples/RIE_Ar_100_burst.py
cd RIE_Ar_100_burst; sbatch submit
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate, etch_mode

nx, ny = 8, 8
ml = compute_ml("100", nx, ny)           # 64 atoms/ML for 8x8 box

spec = SimSpec(
    orientation         = "100",
    surface             = "2x1",
    surface_temperature = 300.0,

    species             = "Ar",
    energy              = 200.0,         # eV
    ion_angle           = 0.0,

    fluence             = 1,
    ml                  = ml,
    box_x               = nx,
    box_y               = ny,
    box_depth           = 8,

    flux_ratio          = 150,           # 150 O• radicals per ion impact
    radical_energy      = 0.2,           # eV per radical
    radical_burst       = True,          # burst mode
    radical_burst_chunk = 10,            # deposit burst in 10 atom chunks
    skip_radical_thermalization = True,  # no thermalization between chunks

    impact_time         = 500.0,
    thermalization_time = 500.0,
    inter_neutral_time  = 1500.0,

    dump_mode           = "all",         # dump only etch events

    wall_hours          = 1,
    nice                = 1,
    account             = "dgraves",
    name                = "RIE_Ar_100_burst_200eV_FR150",
)

validate(spec)
print(f"etch_mode = {etch_mode(spec)}")   # → "burst-rie-etch"
print(f"ML = {ml}, chunk_size = {ml // 2}, chunks = {150 // (ml // 2)} full + remainder {150 % (ml // 2)}")
make_sim(spec, Path("RIE_Ar_100_burst"))
