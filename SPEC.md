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

`etch_mode(spec)` prefixes the mode string with `carbon-` when `initial_config_file`
is set (see [Carbon-etch mode](#carbon-etch-mode-arbitrary-initial-structure) below)
— e.g. `carbon-rie-etch`. This prefix composes with any of the six modes above.

**Single-impact statistics mode** (`single_impact=True`) is a separate execution mode
that overrides the normal fluence loop — see
[Single-impact statistics mode](#single-impact-statistics-mode) below. It can be combined
with either the crystal-builder or carbon-etch initial structure, and with `flux_ratio`
(including burst mode) for radical pre-exposure before each trial's impact.

**Deposition mask** (`mask_type` set) restricts where ions/radicals are permitted to land,
independent of etch mode — see [Deposition mask](#deposition-mask) below.
⚠️ **Not yet committed / experimental** — implemented but not covered by tests, and not
yet validated against `cycle-etch` (explicitly rejected by `validate()`).

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
| `ion_angle` | `float` | `0.0` | Ion beam angle in degrees from surface normal |
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
| `radical_energy` | `float` | `0.2` | eV per O• (used when `radical_temperature` is `None`) |
| `radical_temperature` | `float\|None` | `None` | K; enables Maxwell-Boltzmann speed sampling (overrides `radical_energy`) |
| `radical_angle` | `float` | `0.0` | Degrees from surface normal for radicals in fixed-angle mode |
| `radical_angle_distribution` | `bool` | `False` | Lambert cosine (3-D) angle distribution for O• radicals |
| `max_inter_neutral_time` | `float` | `5000.0` | fs; cap on per-radical halt time in stochastic mode |
| `radical_i_above` | `float` | `6.0` | Å above surface to inject O• radical |
| `skip_radical_thermalization` | `bool` | `False` | Omit `thermalize.lmp` between successive radicals in the one-at-a-time loop |
| `dump_mode` | `str` | `"all"` | Trajectory dump mode: `"all"` \| `"etch_only"` \| `"none"` |

**Stochastic radical sampling notes:**

- When `radical_temperature` is set, three Box-Muller Gaussian deviates (σ = √(k_B T/m)) give the 3D velocity components; the speed follows the Maxwell-Boltzmann speed distribution.
- When `radical_angle_distribution=True`, the polar angle θ is sampled from the Lambert cosine (flux-weighted) distribution: θ = arcsin(√U), φ = 2π·U₂. This is the correct distribution for thermalized species hitting a surface, where the flux is weighted by the normal velocity component cos θ.
- Both flags are independent and can be combined. Fixed-angle + Boltzmann speed is also valid (e.g. normal incidence with a thermal speed distribution).
- In stochastic mode, `max_inter_neutral_time` is used as the MD window for every radical.
- Sampled velocities are logged per-radical to `radical_log.txt` (columns: `impact cn energy_eV polar_deg azimuthal_deg`) and visualised in `radical_distribution.png`.

**Burst radical injection** (`radical_burst=True`): deposits all `flux_ratio` radicals at once instead of one at a time. Mono-energetic fixed-angle only (no `radical_temperature`, no `radical_angle_distribution`); requires `ml > 0`.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `radical_burst` | `bool` | `False` | Enable burst mode |
| `radical_burst_chunk` | `int` | `0` | Atoms deposited per chunk before running dynamics; `0` = auto (0.5 ML) |
| `radical_burst_attempt` | `int` | `200` | Placement attempts per atom in burst deposit; increase if packing failures occur |

All atoms in a chunk land at the same z height (`bound(all,zmax) + radical_i_above`, evaluated once per chunk) via a narrow LAMMPS region, so the box never grows during a chunk's deposition.

### Timing parameters

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `surface_temperature` | `float` | `300.0` | K; substrate thermostat target |
| `impact_time` | `float` | `1000.0` | fs — ion impact MD window |
| `thermalization_time` | `float` | `500.0` | fs — post-impact thermalisation |
| `inter_neutral_time` | `float` | `1500.0` | fs — O• radical impact window (fixed-energy/angle mode only) |

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

### Carbon-etch mode (arbitrary initial structure)

Runs any of the six modes above on an arbitrary LAMMPS data file instead of a
generated diamond surface — e.g. graphullerene balls. Triggered by setting
`initial_config_file`. Monolayer size is computed via Langmuir ML from the box's
XY area (`ml=0` → auto), and there is no carbon replenishment (the anchor region
is frozen, but nothing extends the box downward as material is removed).

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `initial_config_file` | `str\|None` | `None` | Absolute path to a LAMMPS data file; setting this triggers carbon-etch |
| `anchor_z_max` | `float\|None` | `None` | Å; top of the frozen anchor region. **Required** for carbon-etch (validated) |
| `initial_thermalization` | `bool` | `False` | Run NVT equilibration before impacts |
| `initial_thermalization_steps` | `int` | `10000` | NVT steps for initial thermalization (only used if `initial_thermalization=True`) |

### Single-impact statistics mode

Repeats one impact event `n_trials` times, each starting from the same
thermalized surface, to build up impact statistics (e.g. penetration depth
distributions) rather than a cumulative etch simulation. Works with either the
crystal-builder path (`initial_config_file=None`) or the carbon-etch path.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `single_impact` | `bool` | `False` | Enable single-impact statistics mode (overrides the normal fluence loop) |
| `n_trials` | `int` | `100` | Number of independent single-impact trials |
| `randomize_velocities` | `bool` | `False` | Re-assign thermal velocities (`velocity create`, Gaussian) from `surface_temperature` before each trial |

**Trial mechanics:**

1. The surface is built/loaded and thermalized **once**, writing `thermalized.data`
   (crystal path: inside `head.lmp`, first run only; carbon path: via a standalone
   `thermalize_surface.lmp` run before `head.lmp`, since carbon surfaces may have
   multiple disconnected clusters and can't safely use the normal `stopclust`
   machinery during thermalization).
2. Each trial: delete all atoms, `read_data thermalized.data add 0` to reset to
   the clean surface, re-establish groups (`read_data` clears per-atom group
   membership), optionally re-randomize velocities, run one impact event
   (optionally with `flux_ratio` radical pre-exposure / burst beforehand), sweep
   etch products, log per-trial stats.
3. `flux_ratio` / radical pre-exposure work the same as in RIE-etch — each trial
   gets a fresh radical exposure before its single ion impact, not a persistent
   radical history across trials.
4. **Restart**: the submit script reads `ntrials_done.txt` (written by LAMMPS
   after each completed trial) and passes it as `n_complete` — the trial loop
   runs from `n_complete+1` to `n_trials`. `data_file` is always the initial
   surface (crystal: `impact_snaps/0.data`; carbon: `initial_config.data`),
   never the latest impact snapshot, since every trial restarts from
   `thermalized.data`.

Output is written to `impact_stats.txt` (see
[Output file formats](#output-file-formats)) and analyzed with
`python -m diamond_etch_md.analysis.single_impact [sim_dir]`, which also reads
per-trial trajectory dumps for the trajectory-based maximum penetration depth
and writes `penetration_analysis.png` / `penetration_analysis.txt`.

### Deposition mask

⚠️ **Not yet committed to git; no test coverage yet.** Restricts ion/radical
deposition (and the corresponding `bzone` region used for cluster-ejection
bookkeeping) to a sub-window of the box — `expose_zone` — leaving a masked
border around the edges untouched by incoming species, similar to a physical
etch mask. Not currently supported in `cycle-etch` mode (rejected by `validate()`).

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `mask_type` | `str\|None` | `None` | `"xymask"` \| `"xmask"` \| `"ymask"` \| `None` (no mask) |
| `mask_width` | `float\|Tuple[float,float]` | `0.1` | Fraction of box masked on each face. Single float for `xmask`/`ymask`; `(x_frac, y_frac)` tuple for `xymask` (a bare float is auto-normalized to `(w, w)`) |

**Mask types:**

| `mask_type` | Masked region |
|-------------|----------------|
| `xymask` | Both x- and y-faces masked; exposure window is a centered rectangle |
| `xmask` | Only x-faces masked; exposure window spans the full y extent |
| `ymask` | Only y-faces masked; exposure window spans the full x extent |

Each active fraction must satisfy `0.0 < frac < 0.5` (validated). E.g.
`mask_type="xymask", mask_width=0.3` leaves a centered window covering the
middle 40% of both x and y (30% masked on each side).

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
| `radical_energy` | `float` | `0.2` | `> 0` if `flux_ratio > 0` | eV per O• radical (used when `radical_temperature` is `None`) |
| `radical_temperature` | `float\|None` | `None` | | K; Maxwell-Boltzmann speed sampling for this phase's radicals |
| `radical_angle` | `float` | `0.0` | | Degrees from normal for radicals in fixed-angle mode |
| `radical_angle_distribution` | `bool` | `False` | | Lambert cosine angle distribution for this phase's radicals |
| `max_inter_neutral_time` | `float` | `5000.0` | | fs; per-radical halt time cap in stochastic mode |
| `radical_i_above` | `float` | `12.0` | | Å above surface to inject O• radical |

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

| Filename pattern | Mode | Description |
|-----------------|------|-------------|
| `event_dump_${c}.dump` | ion-etch, rie-etch, multi-ion | Ion impact trajectory |
| `event_dump_n${c}_${cn}.dump` | rie-etch | Radical (neutral) impact trajectory (`cn` = radical index within impact) |
| `event_dump_ion${c}.dump` | cycle-etch | Ion phase impact trajectory |

`c` = ion impact number; `cn` = O• radical index within that impact (1-indexed).

**Dump modes** (controlled by `dump_mode` on `SimSpec`):

| Mode | Behavior |
|------|----------|
| `"all"` | Every impact creates a dump file (default) |
| `"etch_only"` | Dump is created per impact but deleted at end unless a C-containing etch product or channeled atom was detected |
| `"none"` | No dump files; radical dumps also suppressed |

### `radical_log.txt`

Written in RIE-etch mode (one row per O• radical deposited). Space-separated, no header.

| Column | Name | Description |
|--------|------|-------------|
| 0 | `impact` | Ion impact number |
| 1 | `cn` | Radical index within this impact (1-indexed) |
| 2 | `energy_eV` | Sampled radical kinetic energy (eV) |
| 3 | `polar_deg` | Polar angle from surface normal (°) |
| 4 | `azimuthal_deg` | Azimuthal angle (°) |

Used by `plot_radical_distribution()` to verify Boltzmann and cosine sampling.

### `impact_stats.txt` (single-impact mode)

Written once per trial in single-impact mode. Space-separated, single-line header
(`#`-prefixed).

| Column | Name | Description |
|--------|------|-------------|
| 0 | `trial` | Trial number (1-indexed) |
| 1 | `surf_z_before_A` | Carbon surface z (Å) immediately before this trial's impact |
| 2 | `ion_z_after_A` | z (Å) of the embedded ion after impact; `9999.0` if the ion was not embedded (sputtered/reflected) |
| 3 | `ion_in_box` | `1` if the ion is still in the `insert` group after the impact, `0` otherwise |
| 4 | `pen_depth_A` | `surf_z_before_A - ion_z_after_A` (Å); the "final resting position" depth, as opposed to the trajectory-based maximum depth |

Parsed by `parse_impact_stats()` / `analyze_single_impact()` in
`diamond_etch_md.analysis.single_impact`.

### `thermalized.data` / `ntrials_done.txt` (single-impact mode)

- `thermalized.data` — the equilibrated surface state (positions + velocities,
  `nofix nocoeff`) that every trial resets to via `read_data ... add 0`. Written
  once, on the first run (crystal path) or by `thermalize_surface.lmp` before
  `head.lmp` ever runs (carbon path).
- `ntrials_done.txt` — plain integer, the number of completed trials. Written
  after every trial (`shell echo ${trial} > ntrials_done.txt`); read by the
  submit script as the restart counter (`n_complete`).

### `penetration_analysis.png` / `penetration_analysis.txt` (single-impact mode)

Written by `diamond_etch_md.analysis.single_impact` (`python -m
diamond_etch_md.analysis.single_impact [sim_dir]`). Combines `impact_stats.txt`
(final resting-position depth) with per-trial trajectory dumps in
`etch_event_trajs/` (maximum depth reached along the trajectory, the physically
meaningful quantity for comparison with SRIM/TRIM range predictions) into a
4-panel figure and a text summary.

### `ATOM_CHANNELED`

Presence-only flag file written when a channeled atom is detected. Checked by analysis tools to identify channeling events; does not persist between jobs (cleared at job start).

### `LAMMPS_FAILED`

Presence-only flag file written when LAMMPS exits with a non-zero return code. The submit script skips re-queuing when this file exists. Currently persists between job runs (see TODO: LAMMPS_FAILED auto-clear).

### `spec.json`

JSON snapshot of the `SimSpec` used to generate this simulation directory. Written by `make_sim()` for automatic spec recovery by `diamond-etch-md-plot`.

**Backward compatibility:** `SimSpec.from_dict()` maps old field names to their renamed equivalents:
- `temperature` → `surface_temperature`
- `angle` → `ion_angle`

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
--radical-energy 0.2       # eV per O• radical (fixed-energy mode)
--radical-temperature 500  # K; enables Maxwell-Boltzmann speed sampling
--radical-angle-distribution   # Lambert cosine polar angles for radicals
--max-inter-neutral-time 5000  # fs; cap on per-radical MD window (stochastic mode)
--box-x 9 --box-y 9        # lateral box size in lattice units
--surface-temperature 300  # K
--wall-hours 24            # SLURM wall time
--name my_sim              # SLURM job name
--email user@example.com   # SLURM mail address (omit for no mail)
--lammps-module lammps/kokkos/gpu_della9_2022
--seed-adjust 1            # increment for independent replicas
--remove-ar / --no-remove-ar
--nice 2

# Carbon-etch mode (arbitrary initial structure)
--initial-config-file /path/to/structure.data
--anchor-z-max 7.0         # Å; required when --initial-config-file is set
--initial-thermalization
--initial-thermalization-steps 300000

# Single-impact statistics mode
--single-impact
--n-trials 100
--randomize-velocities

# Deposition mask (experimental, uncommitted)
--mask-type xymask         # "xymask" | "xmask" | "ymask"
--mask-width 0.3           # fraction of box masked on each active face

# Burst radical injection
--radical-burst
--radical-burst-chunk 5
--radical-burst-attempt 200
--skip-radical-thermalization
```
