# DiamondEtchMD — Simulation Specification Reference

This document covers every field on `SimSpec`, `IonComponent`, and `CyclePhase`,
along with valid values, defaults, simulation modes, and output file formats.

---

## Simulation modes

| Mode | Condition | Description |
|------|-----------|-------------|
| `ion-etch` | `phases=None`, `ion_mix=None`, `flux_ratio=0` | Single species, ions only |
| `rie-etch` | `phases=None`, `ion_mix=None`, `flux_ratio>0` | Single species + O• radical pre-exposure |
| `multi-ion-etch` | `ion_mix` set, `flux_ratio=0` | Stochastic multi-species mix, no radicals |
| `multi-rie-etch` | `ion_mix` set, `flux_ratio>0` | Stochastic multi-species mix + O• radicals |
| `cycle-etch` | `phases` set | Multi-phase cycling (sequential species/fluence blocks) |
| `ALE-etch` | `phases` with exactly 2 entries | Atomic layer etching (validated via `make_ale()`) |

---

## `SimSpec` fields

### Surface geometry

| Field | Type | Default | Valid values / notes |
|-------|------|---------|----------------------|
| `orientation` | `str` | `"100"` | `"100"`, `"110"`, `"111"`, `"113"` |
| `surface` | `str` | `"1x1"` | See table below; depends on orientation |
| `box_x` | `int` | `9` | Lattice units along x |
| `box_y` | `int` | `9` | Lattice units along y |
| `box_depth` | `int` | `3` | Slab depth in lattice units (`lat_top`) |
| `ml` | `int` | `0` | Atoms per monolayer; `0` → computed as `ml_factor × box_x × box_y` |

**Valid `surface` values by orientation:**

| Orientation | Surface keys | Notes |
|-------------|-------------|-------|
| `100` | `1x1`, `2x1`, `2x1_O`, `O_ether` | `2x1` requires even `box_x` and `box_y` |
| `110` | `""` (bare), `O` | |
| `111` | `1x1`, `2x1_single`, `2x1_pandey`, `1x1_O`, `2x1_single_O`, `2x1_pandey_O` | |
| `113` | `""` (bare), `O` | |

**ML factors** (atoms per monolayer = `ml_factor × box_x × box_y`):

| Orientation | `ml_factor` | Example (default box) |
|-------------|------------|----------------------|
| `100` | 1 | 9×9 → 81 |
| `110` | 4 | 4×6 → 96 |
| `111` | 2 | 5×9 → 90 |
| `113` | 4 | 9×3 → 108 |

### Ion parameters (single-species / multi-ion modes)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `species` | `str` | `"O"` | `"O"`, `"H"`, `"Ar"`, `"O2"` — single-species mode only |
| `energy` | `float` | `0.5` | eV; total dimer energy for O2 (split per-atom automatically) |
| `angle` | `float` | `0.0` | Degrees from surface normal |
| `fluence` | `int` | `50` | Total monolayers to simulate (single-species mode only) |

**Species properties:**

| Species | Atom type | Mass var | Is molecule | Energy divisor | ZBL | Removed after impact |
|---------|-----------|----------|-------------|----------------|-----|----------------------|
| `O` | 3 | `M_O` | No | 1 | No | No |
| `H` | 2 | `M_H` | No | 1 | No | No |
| `Ar` | 4 | `M_Ar` | No | 1 | Yes | Yes (default) |
| `O2` | 3 | `M_O` | Yes (`O2.molecule`) | 2 | No | No |

### RIE-etch parameters (single-species with radical pre-exposure)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `flux_ratio` | `int` | `0` | O• radicals deposited before each ion impact; `0` = ion-etch |
| `radical_energy` | `float` | `0.2` | eV per O• radical |

### Timing parameters

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `temperature` | `float` | `300.0` | K; thermostat target |
| `impact_time` | `float` | `2000.0` | fs — ion impact MD window |
| `thermalization_time` | `float` | `500.0` | fs — post-impact thermalisation |
| `inter_neutral_time` | `float` | `1000.0` | fs — O• radical impact window (RIE/cycling) |

### SLURM / cluster parameters

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `wall_hours` | `int` | `24` | SLURM wall time in hours |
| `name` | `str` | `""` | SLURM job name; also used for `--dependency=singleton` serialization |
| `account` | `str` | `"dgraves"` | SLURM account (`#SBATCH --account`) |
| `email` | `str` | `""` | If non-empty, emits `#SBATCH --mail-type=END,FAIL` and `#SBATCH --mail-user=<email>` |
| `lammps_module` | `str` | `"lammps/kokkos/gpu_della9_2022"` | Module loaded before running LAMMPS |
| `nice` | `int` | `2` | `#SBATCH --nice` priority offset (must be ≥ 1) |

### Analysis / plotting parameters

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `plot_interval_hours` | `int` | `12` | Hours between auto-plot runs while LAMMPS is running; `0` = disabled |
| `cna_stride` | `int` | `0` | CNA stride for `--cna` mode; `0` = 1 analysis per ML |

### Miscellaneous

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `remove_ar` | `bool` | `True` | Delete Ar atoms after each impact; set `False` to retain for analysis |
| `seed_adjust` | `int` | `0` | Random seed offset; increment for independent replicas of the same condition |

### Multi-ion mode (`ion_mix`)

Set `ion_mix` to a list of `IonComponent` objects instead of using `species`/`energy`.

```python
ion_mix=[
    IonComponent(species="O",  fraction=0.6, energy=1.0),
    IonComponent(species="Ar", fraction=0.4, energy=2.0),
]
```

All fractions must sum to 1.0 (use `normalize_ion_mix()` to auto-normalize).
`flux_ratio` and `radical_energy` on `SimSpec` still apply when `ion_mix` is set.

### Cycling mode (`phases` / `cycles`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `phases` | `List[CyclePhase]` | `None` | At least 2 phases required; `None` = single-species mode |
| `cycles` | `int` | `1` | Number of times the full phase list repeats |

`flux_ratio` and `radical_energy` on `SimSpec` are **ignored** in cycling mode; use per-phase values instead.

---

## `IonComponent` fields

Used in `SimSpec.ion_mix` for stochastic multi-species bombardment.

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `species` | `str` | Any key in `SPECIES` | Ion species for this component |
| `fraction` | `float` | `> 0`; all fractions must sum to 1.0 | Probability of selecting this ion per impact |
| `energy` | `float` | `> 0` | Total kinetic energy in eV (for O2, this is the full dimer energy) |

---

## `CyclePhase` fields

Used in `SimSpec.phases` for cycling simulations.

| Field | Type | Default | Constraint | Description |
|-------|------|---------|-----------|-------------|
| `species` | `str` | — | Any key in `SPECIES` | Ion species for this phase |
| `energy` | `float` | — | `> 0` | Total kinetic energy in eV |
| `fluence_ml` | `int` | — | `> 0` | Monolayers of this species per cycle repetition |
| `flux_ratio` | `int` | `0` | `>= 0` | O• radicals deposited before each ion impact in this phase |
| `radical_energy` | `float` | `0.2` | `> 0` if `flux_ratio > 0` | eV per O• radical in this phase |

---

## Output file formats

### `ncarbon.txt`

Written once per impact (and at radical-loop milestones in RIE/cycling modes).
Tab-separated, no header.

| Column | Name | Description |
|--------|------|-------------|
| 0 | `step` | LAMMPS timestep |
| 1 | `neut_complete` | Radicals deposited so far in this impact cycle (used for mid-loop restart detection) |
| 2 | `n_carbon` | Current number of C atoms in the simulation box |
| 3 | `n_impacts` | Total ion impacts completed |

The submit script reads `n_complete = n_impacts` (column 3) to decide whether the simulation has reached `end_fluence × ML` and whether to re-queue.

### `etch_products.txt`

Written once per ejection event (each time a cluster leaves the box or a channeled atom is removed). Space-separated, no header.

| Column | Index | Description |
|--------|-------|-------------|
| `impact` | 0 | Impact number when this event occurred |
| `cn` | 1 | Sequential ejection event counter (`event_count`) |
| `C` | 2 | Number of C atoms in the ejected cluster |
| `H` | 3 | Number of H atoms in the ejected cluster |
| `O` | 4 | Number of O atoms in the ejected cluster |
| `Ar` | 5 | Number of Ar atoms in the ejected cluster |

### Dump files (`etch_event_trajs/`)

One dump file per ejection event.

| Filename pattern | Mode | Description |
|-----------------|------|-------------|
| `event_dump_${c}_${event_count}.dump` | ion-etch, rie-etch, multi-ion | Ion impact event trajectory |
| `event_dump_n${c}_${event_count}.dump` | rie-etch | Radical (neutral) impact event trajectory |
| `event_dump_ion${c}_${event_count}.dump` | cycle-etch | Ion phase event trajectory |

`c` = impact number; `event_count` = sequential event counter (incremented on each ejection and on each channeling event).

The submit script counts events via `ls etch_event_trajs/event_dump_*.dump | wc -l`.

### `ATOM_CHANNELED`

Presence-only flag file written when a channeled atom is detected. Checked by analysis tools to identify channeling events; does not persist between jobs (cleared at job start).

### `LAMMPS_FAILED`

Presence-only flag file written when LAMMPS exits with a non-zero return code. The submit script skips re-queuing when this file exists. Currently persists between job runs (see TODO: LAMMPS_FAILED auto-clear).

### `spec.json`

JSON snapshot of the `SimSpec` used to generate this simulation directory. Written by `make_sim()` for automatic spec recovery by `diamond-etch-md-plot`.

---

## CLI flags

All `SimSpec` fields are exposed as CLI flags on `diamond-etch-md-make` and `diamond-etch-md-submit`. Flags use `--kebab-case` matching the field names.

Key flags:

```
--orientation 100          # surface orientation
--surface 2x1_O            # surface state
--species O                # ion species (single-species mode)
--energy 0.5               # eV
--fluence 50               # monolayers
--flux-ratio 5             # O• radicals per impact (0 = ion-etch)
--radical-energy 0.2       # eV per O• radical
--box-x 9 --box-y 9        # lateral box size in lattice units
--temperature 300          # K
--wall-hours 24            # SLURM wall time
--name my_sim              # SLURM job name
--email user@example.com   # SLURM mail address (omit for no mail)
--lammps-module lammps/kokkos/gpu_della9_2022
--seed-adjust 1            # increment for independent replicas
--remove-ar / --no-remove-ar
--nice 2
```
