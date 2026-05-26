# DiamondEtchMD

Python package for setting up and analysing LAMMPS ReaxFF molecular-dynamics simulations of diamond surface etching. Supports O, O₂, and Ar ion bombardment of C(100), C(110), C(111), and C(113) surfaces. All surface templates and force-field files are bundled with the package.

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
    surface        = "1x1",
    species        = "O",
    energy         = 0.5,       # eV
    temperature    = 300.0,     # K
    ml             = compute_ml("100", 9, 9),  # 81
    box_x=9, box_y=9, box_depth=3,
    fluence        = 50,        # monolayers
    wall_hours     = 24,
    name           = "100_1x1_O_0.5eV_300K",
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
# C(100) — 1×1 surface, O radical at 0.5 eV
diamond-etch-md --orientation 100 --surface 1x1 --energy 0.5 my_sim

# C(100) — 2×1 reconstructed + O terminated
diamond-etch-md --orientation 100 --surface 2x1_O --energy 0.5 my_sim

# C(100) — O-ether surface
diamond-etch-md --orientation 100 --surface O_ether --energy 0.5 my_sim

# C(111) — Pandey chain
diamond-etch-md --orientation 111 --surface 2x1_pandey --energy 1.0 my_sim

# C(111) — Pandey chain, O-terminated
diamond-etch-md --orientation 111 --surface 2x1_pandey_O --energy 1.0 my_sim

# C(113) — O-terminated
diamond-etch-md --orientation 113 --surface O --energy 0.2 my_sim

# Ar+ bombardment at 100 eV
diamond-etch-md --species Ar --energy 100 --box-depth 10 my_sim

# O2+ bombardment at 50 eV (25 eV per atom internally)
diamond-etch-md --species O2 --energy 50 my_sim
```

## Package layout

```
diamond_etch_md/
  spec.py          SimSpec dataclass, compute_ml(), validate()
  orientations.py  ORIENT registry — lattice commands, ML factors, surface definitions
  species.py       SPECIES registry — atom types, injection heights, ZBL/molecule flags
  builder.py       make_sim() — writes config/head/submit, copies make_surf, symlinks
  cli.py           diamond-etch-md entry point
  lammps/
    config.py         get_config_lmp(), get_config_lmp_cycling()
    head.py           get_head_lmp()        — single-species driver
    head_cycling.py   get_head_lmp_cycling() — cycling (multi-phase) driver
    submit.py         get_submit_script(), get_submit_script_cycling()
    templates/
      make_surf_100.lmp            C(100) all surfaces
      make_surf_110.lmp            C(110)
      make_surf_111_1x1.lmp       C(111) 1×1
      make_surf_111_2x1_single.lmp C(111) 2×1 single chains
      make_surf_111_2x1_pandey.lmp C(111) 2×1 Pandey chains
      make_surf_113.lmp           C(113)
      O2.molecule                 O₂ dimer definition for molecule injection
      sweep.lmp, thermalize.lmp, addfix.lmp, ffield.reax, lat_a.txt, lmp_env.sh
  analysis/
    etch_products.py  parse_etch_products(), etch_yield()
    ncarbon.py        parse_ncarbon(), etch_depth()
```

## SimSpec reference

| Field | Type | Default | Description |
|---|---|---|---|
| `orientation` | str | `"100"` | Crystal surface: `"100"`, `"110"`, `"111"`, or `"113"` |
| `surface` | str | `"1x1"` | Surface state — reconstruction + termination (see table below) |
| `temperature` | float | `300.0` | Substrate temperature (K) |
| `species` | str | `"O"` | Incident species: `"O"`, `"Ar"`, `"O2"` |
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
| `lammps_module` | str | `"lammps/kokkos/gpu_della9_2022"` | LAMMPS module for submit script |

### Surface states

The `surface` field is a single key that encodes both the geometric reconstruction
and the chemical termination of the surface.

| Orientation | Surface key | Description |
|---|---|---|
| `100` | `1x1` | Unreconstructed |
| `100` | `2x1` | 2×1 dimer-row reconstruction |
| `100` | `2x1_O` | 2×1 + ketone O atop surface C |
| `100` | `O_ether` | O bridging between adjacent surface C |
| `110` | `""` | Unterminated (default) |
| `110` | `O` | O-terminated |
| `111` | `1x1` | Unreconstructed |
| `111` | `2x1_single` | 2×1 single-chain |
| `111` | `2x1_pandey` | 2×1 Pandey chain |
| `111` | `1x1_O` | 1×1 + O |
| `111` | `2x1_single_O` | 2×1 single-chain + O |
| `111` | `2x1_pandey_O` | 2×1 Pandey chain + O |
| `113` | `""` | Unterminated (default) |
| `113` | `O` | O-terminated |

### Atoms per monolayer (`ml`)

`compute_ml(orientation, box_x, box_y)` returns `ml_factor × box_x × box_y`:

| Orientation | `ml_factor` | Example (default box) |
|---|---|---|
| `100` | 1 | 9×9 → 81 |
| `110` | 4 | 4×6 → 96 |
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

### Ion species

| Species | Description | Force field | Notes |
|---|---|---|---|
| `O` | Single oxygen atom (O⁺ ion or O radical) | ReaxFF (C-H-O) | Default species |
| `Ar` | Argon ion (Ar⁺) | ReaxFF + ZBL hybrid | Inert; removed after each impact |
| `O2` | Oxygen dimer (O₂⁺) | ReaxFF (C-H-O) | Injected as molecule; energy is per-dimer (halved per atom internally) |

**Ar** uses a hybrid ReaxFF + ZBL (Ziegler-Biersack-Littmark) pair style for
short-range nuclear repulsion. QEQ charges are only computed for non-Ar atoms.
Ar atoms are deleted after each impact since they do not participate in chemistry.

**O₂** is injected as a LAMMPS molecule (two O atoms at 1.2 Å separation).
The user-specified `energy` is the total dimer kinetic energy; each atom receives
half. The `O2.molecule` file is bundled with the package and automatically
symlinked into the simulation directory.

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

**Single-species:** one line per completed impact:
```
impact#  n_carbon  n_hydrogen  n_oxygen
```

**Cycling:** one line per radical and per ion impact:
```
impact#  radical#  n_carbon  n_hydrogen  n_oxygen
```
`radical# > 0` after each O• radical (1-indexed); `radical# = 0` after each ion impact. This enables mid-radical-loop restarts.

## Cycling simulations (multi-phase)

Set `phases` to a list of `CyclePhase` objects to alternate between ion species within a single run. The `cycles` field controls how many times the phase list repeats.

```python
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, validate
from pathlib import Path

ml = compute_ml("100", 8, 8)  # 64 atoms/ML for 8×8 box

# Example 1: Ar+ physical sputtering → O2+ chemical etching (10 cycles)
spec = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    ml          = ml,
    box_x=8, box_y=8, box_depth=5,
    phases=[
        CyclePhase(species="Ar", energy=30.0, fluence_ml=5),          # 5 ML Ar+, no radicals
        CyclePhase(species="O2", energy=20.0, fluence_ml=5,
                   flux_ratio=10, radical_energy=0.2),                  # 5 ML O2+, R=10 O• per O2+
    ],
    cycles     = 10,
    wall_hours = 48,
    name       = "100_Oether_Ar30eV_O2_20eV_R10_x10",
)
validate(spec)
make_sim(spec, Path("ar_o2_cycling"))
```

```python
# Example 2: O+ → O2+ cycling (no Ar → faster plain ReaxFF potential)
spec = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    ml          = ml,
    box_x=8, box_y=8, box_depth=5,
    phases=[
        CyclePhase(species="O",  energy=1.0, fluence_ml=5, flux_ratio=0),
        CyclePhase(species="O2", energy=20.0, fluence_ml=5, flux_ratio=10),
    ],
    cycles     = 10,
    name       = "100_Oether_O_O2_cycling",
)
make_sim(spec, Path("o_o2_cycling"))
```

```python
# Example 3: 3-phase — Ar sputtering → O passivation → O2 etching
spec = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    ml          = ml,
    box_x=8, box_y=8, box_depth=5,
    phases=[
        CyclePhase(species="Ar", energy=50.0, fluence_ml=5),
        CyclePhase(species="O",  energy=1.0,  fluence_ml=3, flux_ratio=5),
        CyclePhase(species="O2", energy=20.0, fluence_ml=5, flux_ratio=10),
    ],
    cycles = 5,
    name   = "100_3phase_Ar_O_O2",
)
make_sim(spec, Path("ar_o_o2_cycling"))
```

### `CyclePhase` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `species` | str | — | Ion species: `"O"`, `"O2"`, `"Ar"`, `"H"` |
| `energy` | float | — | Ion energy in eV (total dimer energy for O₂) |
| `fluence_ml` | int | — | ML of this species per cycle repetition |
| `flux_ratio` | int | `0` | O• radicals deposited before each ion impact (0 = none) |
| `radical_energy` | float | `0.2` | eV per O• radical |

**Notes:**
- If no phase uses Ar, plain ReaxFF (3 atom types) is used — faster than the ZBL hybrid.
- `ncarbon.txt` cycling format: `c cn ncarbon nhydrogen noxygen` — `cn > 0` after a radical impact, `cn = 0` after an ion impact. The submit script reads col 2 to resume mid-radical-loop after a wall-time restart.
- `O2.molecule` is automatically symlinked when any phase uses O₂.

## Examples

[`example_all_options.py`](example_all_options.py) — generates all three ion species
(O, Ar, O₂) in one shot with full annotations on every field.

The [`examples/`](examples/) directory has focused single-case scripts:

| File | What it demonstrates |
|---|---|
| [`O_radical_100.py`](examples/O_radical_100.py) | O radical on C(100) O-ether surface |
| [`Ar_sputtering_100.py`](examples/Ar_sputtering_100.py) | Ar⁺ physical sputtering of C(100) 1x1 at 100 eV |
| [`O2_bombardment_111.py`](examples/O2_bombardment_111.py) | O₂⁺ dimer on C(111) Pandey chain |
| [`O_terminated_111_pandey.py`](examples/O_terminated_111_pandey.py) | C(111) Pandey + O surface |
| [`O_etching_113.py`](examples/O_etching_113.py) | C(113) O-terminated surface |
| [`angled_Ar_100.py`](examples/angled_Ar_100.py) | 45° off-normal Ar⁺ incidence |
| [`high_energy_O_100.py`](examples/high_energy_O_100.py) | 200 eV O⁺ with deep slab (box_depth guidance) |

```bash
python examples/O_radical_100.py
sbatch O_radical_100/submit
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
