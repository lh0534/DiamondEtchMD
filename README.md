# DiamondEtchMD

Python package for setting up and analysing LAMMPS ReaxFF molecular-dynamics simulations of diamond surface etching. Supports radical (O, H) and ion bombardment of C(001), C(111), and C(113) surfaces on Princeton's Della GPU cluster.

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
    orientation  = "001",
    species      = "O",
    energy       = 0.5,       # eV
    temperature  = 300.0,     # K
    ml           = compute_ml("001", 9, 9),  # 81
    box_x=9, box_y=9, box_depth=3,
    fluence      = 50,        # monolayers
    wall_hours   = 24,
    name         = "001_O_0.5eV_300K",
)

make_sim(spec, Path("my_sim"), dfiles_root=Path("/path/to/dfiles"))
# → my_sim/config.lmp, head.lmp, make_surf.lmp, submit, symlinks
```

Submit:
```bash
sbatch my_sim/submit
```

Or use the CLI:
```bash
diamond-etch-md --orientation 001 --species O --energy 0.5 --temperature 300 my_sim
diamond-etch-md --orientation 111 --reconstruction 2x1_pandey --energy 1.0 111_pandey
diamond-etch-md --orientation 113 --termination O --angle 15 --energy 0.2 113_O_15deg
```

## Package layout

```
diamond_etch_md/
  spec.py          SimSpec dataclass, compute_ml(), validate()
  orientations.py  ORIENT registry — lattice commands, ML factors, make_surf paths
  species.py       SPECIES registry — atom type indices, injection heights
  builder.py       make_sim() — writes config/head/submit, copies make_surf, symlinks shared files
  cli.py           diamond-etch-md entry point
  lammps/
    config.py      get_config_lmp()  — per-condition LAMMPS variable file
    head.py        get_head_lmp()    — main driver script (per-impact loop)
    submit.py      get_submit_script() — SLURM batch script
  analysis/
    etch_products.py  parse_etch_products(), etch_yield()
    ncarbon.py        parse_ncarbon(), etch_depth()
```

## SimSpec reference

| Field | Type | Default | Description |
|---|---|---|---|
| `orientation` | str | `"001"` | Crystal surface: `"001"`, `"111"`, or `"113"` |
| `reconstruction` | str | `"bare"` | Surface reconstruction (see below) |
| `termination` | str | `"bare"` | Chemical termination: `bare`, `H`, `O`, `O_ether` |
| `temperature` | float | `300.0` | Substrate temperature (K) |
| `species` | str | `"O"` | Incident species: `"O"` or `"H"` |
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

### Reconstructions by orientation

| Orientation | Valid reconstructions |
|---|---|
| `001` | `bare`, `2x1` (controlled via `config.lmp` flag) |
| `111` | `bare` / `1x1`, `2x1_single`, `2x1_pandey` |
| `113` | `bare`, `O` |

### Atoms per monolayer (`ml`)

`compute_ml(orientation, box_x, box_y)` returns `ml_factor × box_x × box_y`:

| Orientation | `ml_factor` | Example (default box) |
|---|---|---|
| `001` | 1 | 9×9 → 81 |
| `111` | 2 | 5×9 → 90 |
| `113` | 4 | 9×3 → 108 |

### Box depth and ion energy

`box_depth` (= `lat_top`) must be deep enough that the ion stops within the mobile region above the fixed anchor layer. Empirical values from the RIE study:

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
depths = etch_depth(nc, ml=81, box_x=9, box_y=9, orientation="001")
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

## Running tests

```bash
# Unit + integration tests (no cluster required)
pytest

# Also submit a real SLURM job (requires Della)
pytest -m slurm -v
```

## dfiles layout expected by `make_sim`

```
dfiles/
  ffield.reax          ReaxFF force field (C-H-O; ZBL for Ar)
  lat_a.txt            Equilibrium lattice constant for the chosen potential
  lmp_env.sh           Module load script
  radicals/
    make_surf.lmp      Surface builder for 001 (all reconstructions via config flags)
    sweep.lmp          Cluster detection and ejection sweep
    thermalize.lmp     NVT thermalisation block
    addfix.lmp         Carbon replenishment block
  111/
    bare_surf/
      1x1_non-reconstructed/make_surf.lmp
      2x1_Single_Chains/make_surf.lmp
      2x1_Pandey_Chains/make_surf.lmp
  113/
    bare_surf/make_surf.lmp
    O_terminated/make_surf.lmp
```
