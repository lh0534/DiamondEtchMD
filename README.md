# DiamondEtchMD

Python package for setting up and analysing LAMMPS ReaxFF molecular-dynamics simulations of diamond surface etching. Supports O radical and ion bombardment of C(100), C(111), and C(113) surfaces. All surface templates and force-field files are bundled with the package.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9. No third-party dependencies.

## Quick start

```python
from diamond_etch_md import SimSpec, compute_ml, make_sim
from pathlib import Path

spec = SimSpec(
    orientation    = "100",
    reconstruction = "bare_1x1",
    termination    = "bare",
    species        = "O",
    energy         = 0.5,       # eV
    temperature    = 300.0,     # K
    ml             = compute_ml("100", 9, 9),  # 81
    box_x=9, box_y=9, box_depth=3,
    fluence        = 50,        # monolayers
    wall_hours     = 24,
    name           = "100_bare_1x1_O_0.5eV_300K",
)

make_sim(spec, Path("my_sim"))
# → my_sim/config.lmp, head.lmp, make_surf.lmp, submit, symlinks
```

Submit:
```bash
sbatch my_sim/submit
```

Or use the CLI:
```bash
# C(100) — bare 1×1 surface, O radical at 0.5 eV
diamond-etch-md --orientation 100 --reconstruction bare_1x1 --energy 0.5 my_sim

# C(100) — 2×1 reconstructed, O-ether terminated
diamond-etch-md --orientation 100 --reconstruction bare_2x1 --termination O_ether --energy 0.5 my_sim

# C(100) — 2×1 reconstructed, ketone-O terminated
diamond-etch-md --orientation 100 --reconstruction bare_2x1 --termination O --energy 0.5 my_sim

# C(111) — Pandey chain reconstruction, bare
diamond-etch-md --orientation 111 --reconstruction bare_2x1_pandey --energy 1.0 my_sim

# C(111) — Pandey chain reconstruction, O-terminated
diamond-etch-md --orientation 111 --reconstruction bare_2x1_pandey --termination O_2x1_pandey --energy 1.0 my_sim

# C(113) — O-terminated, angled incidence
diamond-etch-md --orientation 113 --termination O --angle 15 --energy 0.2 my_sim
```

## Package layout

```
diamond_etch_md/
  spec.py          SimSpec dataclass, compute_ml(), validate()
  orientations.py  ORIENT registry — lattice commands, ML factors, make_surf paths
  species.py       SPECIES registry — atom type indices, injection heights
  builder.py       make_sim() — writes config/head/submit, copies make_surf, symlinks
  cli.py           diamond-etch-md entry point
  lammps/
    config.py         get_config_lmp()    — per-condition LAMMPS variable file
    head.py           get_head_lmp()      — main driver script (per-impact loop)
    submit.py         get_submit_script() — SLURM batch script
    templates/
      make_surf_100.lmp           C(100) all reconstructions + terminations
      make_surf_111_1x1.lmp       C(111) 1×1
      make_surf_111_2x1_single.lmp C(111) 2×1 single chains
      make_surf_111_2x1_pandey.lmp C(111) 2×1 Pandey chains
      make_surf_113.lmp           C(113)
      sweep.lmp, thermalize.lmp, addfix.lmp, ffield.reax, lat_a.txt, lmp_env.sh
  analysis/
    etch_products.py  parse_etch_products(), etch_yield()
    ncarbon.py        parse_ncarbon(), etch_depth()
```

## SimSpec reference

| Field | Type | Default | Description |
|---|---|---|---|
| `orientation` | str | `"100"` | Crystal surface: `"100"`, `"111"`, or `"113"` |
| `reconstruction` | str | `"bare_1x1"` | Surface reconstruction (see table below) |
| `termination` | str | `"bare"` | Chemical termination (see table below) |
| `temperature` | float | `300.0` | Substrate temperature (K) |
| `species` | str | `"O"` | Incident species: `"O"` |
| `energy` | float | `0.5` | Incident particle energy (eV) |
| `angle` | float | `0.0` | Incidence angle from surface normal (degrees) |
| `fluence` | int | `50` | Total fluence (monolayers) |
| `ml` | int | `0` | Atoms per monolayer; `0` triggers `compute_ml()` |
| `box_x` | int | `9` | Lateral box size, x (lattice units) |
| `box_y` | int | `9` | Lateral box size, y (lattice units) |
| `box_depth` | int | `3` | Surface slab depth — `lat_top` (lattice units) |
| `impact_time` | float | `2000.0` | Simulation time per impact (fs) |
| `thermalization_time` | float | `500.0` | Thermalisation time after each impact (fs) |
| `wall_hours` | int | `24` | SLURM wall-clock limit (hours) |
| `name` | str | `""` | SLURM job name (auto-generated if empty) |
| `account` | str | `"dgraves"` | SLURM account to charge |
| `email` | str | `""` | Email for END/FAIL notifications; empty = no mail |

### Reconstructions and valid terminations

| Orientation | Reconstruction | Valid terminations |
|---|---|---|
| `100` | `bare_1x1` | `bare`, `O`, `O_ether` |
| `100` | `bare_2x1` | `bare`, `O`, `O_ether` |
| `111` | `bare_1x1` | `bare`, `O_1x1` |
| `111` | `bare_2x1_single` | `bare`, `O_2x1_single` |
| `111` | `bare_2x1_pandey` | `bare`, `O_2x1_pandey` |
| `113` | `bare` | `bare`, `O` |

**C(100):** reconstruction and termination are independent. A single template
handles all cases: `bare_2x1` triggers dimer-row displacements along [110]/[−1−10];
`O` places ketone O atop surface C; `O_ether` bridges O between adjacent C atoms.

**C(111):** each reconstruction has its own template. O termination names embed
the reconstruction (`O_1x1`, `O_2x1_single`, `O_2x1_pandey`) to make valid
combinations unambiguous. O atoms are placed atop surface C along [111].

**C(113):** no reconstruction; `O` places O atop surface C.

### Atoms per monolayer (`ml`)

`compute_ml(orientation, box_x, box_y)` returns `ml_factor × box_x × box_y`:

| Orientation | `ml_factor` | Example (default box) |
|---|---|---|
| `100` | 1 | 9×9 → 81 |
| `111` | 2 | 5×9 → 90 |
| `113` | 4 | 9×3 → 108 |

### Box depth and ion energy

`box_depth` (= `lat_top`) must be deep enough that the ion stops within the
mobile region above the fixed anchor layer. Empirical guidance:

| Energy | Recommended `box_depth` |
|---|---|
| ≤ 20 eV | 5 |
| 50 eV | 6 |
| 100 eV | 10 |
| 200 eV | 12 |

For low-energy radicals (< 1 eV) the default of 3 is sufficient.

## Analysis

```python
from diamond_etch_md.analysis.etch_products import parse_etch_products, etch_yield
from diamond_etch_md.analysis.ncarbon import parse_ncarbon, etch_depth

# Etch yield (C atoms ejected per impact)
records = parse_etch_products("my_sim/etch_products.txt")
y = etch_yield(records, ml=81)

# Etch depth vs impact number (in monolayers)
nc = parse_ncarbon("my_sim/ncarbon.txt")
depths = etch_depth(nc, ml=81, box_x=9, box_y=9, orientation="100")
```

### `etch_products.txt` format

One line per ejected cluster:

```
impact#  atom_type  n_C  n_H  n_O  vx  vy  vz
```

### `ncarbon.txt` format

One line per completed impact:

```
impact#  n_carbon  n_hydrogen  n_oxygen
```

## Examples

[`example_all_options.py`](example_all_options.py) — a fully annotated Python script
showing every `SimSpec` field with valid values and descriptions. Run it to generate a
sample simulation directory:

```bash
python example_all_options.py
```

## Running tests

```bash
# Unit + integration tests (no cluster required)
pytest

# Also submit a real SLURM job (requires Della)
pytest -m slurm -v
```

## Roadmap

Planned features are tracked in [`TODO.md`](TODO.md).
