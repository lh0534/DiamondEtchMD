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
Not supported for `cycle-etch` (explicitly rejected by `validate()`).

---

## `SimSpec` fields

### Surface geometry

| Field | Type | Default | Valid values / notes |
|-------|------|---------|----------------------|
| `orientation` | `str` | `"100"` | `"100"`, `"110"`, `"111"`, `"113"`; hexagonal diamond `"hex0001"`, `"hex1120"`, `"hex1010"` |
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
| `hex0001`, `hex1120`, `hex1010` | `1x1` | Hexagonal diamond, bulk-terminated (one dangling bond per surface C) |

**ML factors** (atoms per monolayer = `ml_factor × box_x × box_y`):

| Orientation | `ml_factor` | Example (default box) |
|-------------|------------|----------------------|
| `100` | 1 | 9×9 → 81 |
| `110` | 4 | 4×6 → 96 |
| `111` | 2 | 5×9 → 90 |
| `113` | 4 | 9×3 → 108 |
| `hex0001` | 2 | 8×5 → 80 |
| `hex1120` | 4 | 5×5 → 100 |
| `hex1010` | 2 | 5×8 → 80 |

**Hexagonal diamond (lonsdaleite).** LAMMPS has no hexagonal-diamond lattice style, so `hexagonal.py` builds an orthohexagonal `lattice custom` cell (8 atoms, a × √3a × c) with its axes permuted so the surface normal is z: `hex0001` (basal, z unit c), `hex1120` (a-plane, z unit a), `hex1010` (m-plane, z unit √3a). The basis is shifted along z so the slab top (integer lattice z) and the addfix grid (−5/8 − k/2) both fall in the gap that cuts one bond per surface atom; the atomic planes repeat every half cell, so every replenished layer is identical. `config.lmp` carries the lattice as `hex_lattice` (custom-lattice arguments) and `hex_zu` (Å per lattice z unit); `make_surf_hex.lmp` serves all three. a, c and u (vertical bond / c) are the ReaxFF zero-pressure values in `hexagonal.HEX_LATTICE` (`tools/relax_lonsdaleite.py`). CNA counts hexagonal diamond as crystalline (second-neighbour shell 6 × 421 + 6 × 422, vs 12 × 421 for cubic).

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
| `flux_ratio` | `float` | `0` | O• radicals deposited before each ion impact; `0` = ion-etch; non-integer → stochastic floor/ceil per impact so the long-run average equals the specified value |
| `radical_energy` | `float` | `0.2` | eV per O• (used when `radical_temperature` is `None`) |
| `radical_temperature` | `float\|None` | `None` | K; enables Boltzmann energy sampling (overrides `radical_energy`); distribution set by `radical_energy_dist` |
| `radical_angle` | `float` | `0.0` | Degrees from surface normal for radicals in fixed-angle mode |
| `radical_angle_distribution` | `bool` | `False` | Lambert cosine (3-D) angle distribution for O• radicals |
| `radical_energy_dist` | `str` | `"flux"` | Boltzmann energy distribution: `"flux"` = E·e^(−E/kT) (striking flux, mean 2kT) \| `"gas"` = √E·e^(−E/kT) (gas-phase MB, mean 1.5kT). A `spec.json` without this key loads as `"gas"` |
| `radical_energy_threshold` | `float\|None` | `None` | eV; Boltzmann draws below this are logged but skipped (no MD, treated as reflected). Requires `radical_temperature`; with `radical_burst`, skipped draws are not inserted |
| `radical_threshold_on` | `str` | `"total"` | Threshold compares `"total"` energy E or `"normal"` energy E·cos²θ |
| `max_inter_neutral_time` | `float` | `5000.0` | fs; cap on per-radical halt time in stochastic mode |
| `radical_i_above` | `float` | `6.0` | Å above surface to inject O• radical |
| `skip_radical_thermalization` | `bool` | `False` | Omit `thermalize.lmp` between successive radicals in the one-at-a-time loop |
| `dump_mode` | `str` | `"all"` | Trajectory dump mode: `"all"` \| `"etch_only"` \| `"none"` |

**Stochastic radical sampling notes:**

- When `radical_temperature` is set, the energy is drawn per radical. `radical_energy_dist="flux"` (default for new specs) samples the striking-flux distribution E·e^(−E/kT) as E = −kT·ln(u₁u₂) (Gamma(2, kT)); `"gas"` uses three Box-Muller Gaussian deviates (σ = √(k_B T/m)), i.e. the gas-phase Maxwell-Boltzmann distribution. The flux form is the energy part of the v·cosθ flux weighting; the cosine angle distribution below is the angular part, so combining them is not double counting.
- `radical_energy_threshold`: every radical is drawn and logged, but draws with E (or E·cos²θ for `radical_threshold_on="normal"`) below the threshold are skipped — no deposit, MD or thermalization, only an unchanged `ncarbon.txt` row so `cn` stays contiguous and restarts resume correctly. Skipped radicals still count toward `flux_ratio`. Draws are seeded by (impact, cn), so two runs with the same `seed_adjust` see identical radicals (useful for threshold vs full-distribution validation pairs).
- When `radical_angle_distribution=True`, the polar angle θ is sampled from the Lambert cosine (flux-weighted) distribution: θ = arcsin(√U), φ = 2π·U₂. This is the correct distribution for thermalized species hitting a surface, where the flux is weighted by the normal velocity component cos θ.
- Both flags are independent and can be combined. Fixed-angle + Boltzmann speed is also valid (e.g. normal incidence with a thermal speed distribution).
- In stochastic mode, each radical's MD window is min(1.5·`radical_i_above`/|v_z|, `max_inter_neutral_time`).
- Sampled velocities are logged per-radical to `radical_log.txt` (columns: `impact cn energy_eV polar_deg azimuthal_deg [simulated]`) and visualised in `radical_distribution.png`.
- Cycling phases support all of the above per phase; the cycling neutral loop switches to runtime sampling only when some phase uses Boltzmann energies or cosine angles, otherwise the generated script is unchanged.

**Burst radical injection** (`radical_burst=True`): deposits all `flux_ratio` radicals at once instead of one at a time, each with its own draw (same sampling options as one-by-one radicals, including the threshold); requires `ml > 0`.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `radical_burst` | `bool` | `False` | Enable burst mode |
| `radical_burst_chunk` | `int` | `0` | Atoms deposited per chunk before running dynamics; `0` = auto (min(0.5 ML, what fits in one layer)); validation rejects a chunk that cannot fit in one layer |
| `radical_burst_attempt` | `int` | `5000` | Placement attempts per atom in burst deposit (cheap; 200 left ~2% of 10-atom chunks unplaced in a 6×6 C(100) cell) |
| `radical_burst_near` | `float` | `3.5` | Å; minimum distance of an inserted atom from any atom (`fix deposit ... near X units box`). A `spec.json` without it loads as 5.04 Å, the spacing old lattice-unit bursts used |

All atoms in a chunk are inserted into one ±0.1 Å layer (`z_surf_top + radical_i_above`, fixed once per chunk; `surface_top.lmp` sets `z_surf_top` to the top of the anchor-connected slab cluster, so detached gas cannot lift it) with no dynamics between insertions, so the chunk starts at one height and time and the box never grows during deposition. Each chunk then runs for `inter_neutral_time` with cluster sweeps, then thermalizes unless `skip_radical_thermalization` (RIE and cycling alike). After the last chunk, `sweep.lmp` runs with `sweep_any_vz = 1` and removes every detached atom/cluster above the surface regardless of its velocity, logging each to `etch_products.txt`. During the burst `cn = 1`. Draws go to `radical_log.txt` with `cn` = index within the burst; failed insertions are logged to `BURST_INCOMPLETE`. Burst deposits use `units box` (older generated scripts used lattice units, scaling velocities by the lattice spacings — the former "12.713× Kokkos bug" correction).

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

### CO₂* forced desorption

Surface C atoms bonded to exactly 2 O atoms (with each O exclusively bonded to that C, i.e. not bridging to another C) are CO₂ precursors. Setting `co2_desorb_fraction > 0` force-ejects a random fraction of these sites before each thermalization step, mimicking instantaneous CO₂ desorption.

| Field | Type | Default | Constraint | Notes |
|-------|------|---------|------------|-------|
| `co2_desorb_fraction` | `float` | `0.0` | `[0, 1]` | Fraction of eligible CO₂-precursor C atoms to force-eject per step; `0` = disabled |

When a C atom is ejected, its exclusively-bonded O atoms are removed with it. Each forced-desorption event is logged to `etch_products.txt` (standard 7-column format; `vcm_z = 0` flags it as a forced event rather than a dynamical ejection) and to a separate `co2_desorption.txt` (columns: `impact cn n_C n_O`).

`co2_desorb_fraction = 1.0` ejects all eligible sites deterministically every step; `0.5` ejects each independently at 50% probability.

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

Restricts ion/radical deposition (and the `bzone` region used for cluster-ejection
bookkeeping) to a sub-window of the box — `expose_zone` — leaving a masked border
untouched by incoming species, similar to a physical etch mask. Not supported in
`cycle-etch` mode (rejected by `validate()`).

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `mask_type` | `str\|None` | `None` | `"xymask"` \| `"xmask"` \| `"ymask"` \| `None` (no mask) |
| `mask_width` | `float\|Tuple[float,float]` | `0.1` | Fraction of box masked on each face. Single float for `xmask`/`ymask`; `(x_frac, y_frac)` tuple for `xymask` (a bare float is auto-normalized to `(w, w)`) |
| `invert_mask` | `bool` | `False` | Invert the mask: deposit at the edges/frame instead of the center window. Requires `mask_type`. |
| `freeze_mask` | `bool` | `False` | Freeze the topmost surface atoms in the masked region into the anchor group at startup. Requires `mask_type`. |
| `freeze_mask_depth` | `float` | `2.0` | Å depth from the initial surface to include in the frozen zone (≈1–2 diamond layers). Must be `> 0`. |

**Mask types:**

| `mask_type` | Masked region |
|-------------|----------------|
| `xymask` | Both x- and y-faces masked; exposure window is a centered rectangle |
| `xmask` | Only x-faces masked; exposure window spans the full y extent |
| `ymask` | Only y-faces masked; exposure window spans the full x extent |

Each active fraction must satisfy `0.0 < frac < 0.5` (validated). E.g.
`mask_type="xymask", mask_width=0.3` leaves a centered window covering the
middle 40% of both x and y (30% masked on each side).

**`freeze_mask` restart safety:** The z floor of the frozen zone is computed from
`bound(all,zmax)` on the first run (when `n_complete == 0`) and written to
`freeze_mask_z.txt`. On restarts it is read back unchanged — amorphous carbon that has
deposited on the masked surface during the simulation is therefore *not* frozen. Do not
delete `freeze_mask_z.txt` between restarts.

**State file:** `freeze_mask_z.txt` — single float; created automatically on first run.

---

### Step edge

Removes one monoatomic bilayer from one half of the simulation box, producing a
step-edge surface. The cut is applied after all reconstruction/termination steps and
before the final energy-minimization, so dangling bonds at the step edge are relaxed.
Compatible with every orientation and surface variant; `step_edge=True` may be combined
freely with mask or freeze-mask options.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `step_edge` | `bool` | `False` | Enable the step-edge cut. |
| `step_angle` | `float` | `0.0` | Angle (degrees) of the step edge measured from the x-axis. `0°` → edge runs parallel to y; lower terrace is the `−x` half. `90°` → edge runs parallel to x; lower terrace is the `−y` half. For C(111), `0°` gives a (0°) step and `90°` gives a (30°) step in crystal coordinates. |
| `step_position` | `float` | `0.5` | Fractional position of the step edge along the box (0–1, exclusive). `0.5` places the edge at the centre of the box. |
| `step_invert` | `bool` | `False` | Flip which half is removed. Default (`False`) removes the half the normal vector points toward; `True` removes the opposite half. |
| `step_depth` | `float\|None` | `None` | Thickness (Å) of the bilayer to remove. `None` uses the orientation-specific default: 2.0 Å for C(100) and C(113), 2.5 Å for C(110), 2.2 Å for C(111); hexagonal diamond 2.0 Å (hex0001), 1.2 Å (hex1120), 2.1 Å (hex1010). Must be `> 0` if set. |

**Implementation:** a LAMMPS `region plane` half-space is intersected with a `block`
z-slab covering the top `step_depth_ang` Å of the surface. All atoms inside the
intersection are deleted with `delete_atoms … compress no` (atom IDs are preserved for
restart-file compatibility). Region names `step_half`, `step_zslab`, and `step_zone` are
deleted immediately after use.

**`step_position` geometry:** the plane passes through the point
`(lx × step_position, ly × step_position, 0)` with normal vector
`(cos θ, sin θ, 0)` where `θ = step_angle`. At `step_angle = 0°` the x-coordinate
of the plane is `lx × step_position` and the y-coordinate is irrelevant; at
`step_angle = 90°` the y-coordinate is `ly × step_position`. At intermediate angles
the plane passes through both coordinates simultaneously.

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
| `flux_ratio` | `float` | `0` | `>= 0` | O• radicals deposited before each ion impact in this phase; non-integer → stochastic floor/ceil |
| `radical_energy` | `float` | `0.2` | `> 0` if `flux_ratio > 0` | eV per O• radical (used when `radical_temperature` is `None`) |
| `radical_temperature` | `float\|None` | `None` | `> 0` | K; Boltzmann energy sampling for this phase's radicals (distribution: `radical_energy_dist`) |
| `radical_angle` | `float` | `0.0` | | Degrees from normal for radicals in fixed-angle mode |
| `radical_angle_distribution` | `bool` | `False` | | Lambert cosine angle distribution for this phase's radicals |
| `radical_energy_dist` | `str` | `"flux"` | `"flux"`\|`"gas"` | As on `SimSpec`; missing key in a saved `spec.json` → `"gas"` |
| `radical_energy_threshold` | `float\|None` | `None` | `>= 0`; needs `radical_temperature` | eV; skip this phase's radical draws below it |
| `radical_threshold_on` | `str` | `"total"` | `"total"`\|`"normal"` | Energy compared to the threshold |
| `max_inter_neutral_time` | `float` | `5000.0` | | fs; per-radical halt time cap in stochastic mode |
| `radical_i_above` | `float` | `12.0` | | Å above surface to inject O• radical |
| `radical_burst` | `bool` | `False` | | Deposit this phase's radicals as a burst before each ion (any sampling options above) |
| `radical_burst_chunk` | `int` | `0` | must fit in one layer | Atoms per chunk; `0` = auto (min(0.5 ML, what fits)) |
| `radical_burst_attempt` | `int` | `5000` | | Placement attempts per atom |
| `radical_burst_near` | `float` | `3.5` | `> 0` | Å; minimum distance of an inserted burst atom from any atom |

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
| `impact` | 0 | Impact counter `c` when this event occurred |
| `cn` | 1 | Window it happened in: `0` = ion impact (and the thermalization before the next ion), `k` = the k-th one-by-one radical after ion `c`, `1` = the radical burst after ion `c` |
| `C` | 2 | Number of C atoms in the ejected cluster |
| `H` | 3 | Number of H atoms in the ejected cluster |
| `O` | 4 | Number of O atoms in the ejected cluster |
| `Ar` | 5 | Number of Ar atoms in the ejected cluster |
| `vcm_z` | 6 | Centre-of-mass z velocity (Å/fs); `0.0` marks forced CO₂ desorption |

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

Written in RIE-etch mode and in cycling runs with a Boltzmann/cosine phase (one row per O• radical draw). Space-separated, no header.

| Column | Name | Description |
|--------|------|-------------|
| 0 | `impact` | Ion impact number |
| 1 | `cn` | Radical index within this impact (1-indexed) |
| 2 | `energy_eV` | Sampled radical kinetic energy (eV) |
| 3 | `polar_deg` | Polar angle from surface normal (°) |
| 4 | `azimuthal_deg` | Azimuthal angle (°) |
| 5 | `simulated` | 1 = simulated, 0 = skipped below `radical_energy_threshold`. Present when a threshold is set (RIE) and always in cycling runs with a stochastic phase; absent → all simulated |

Used by `plot_radical_distribution()` to verify Boltzmann and cosine sampling (and the skip fraction), by `check_radical_reflecting()` to exclude skipped draws, and by `summary.txt`, whose radical sticking coefficient counts skipped draws as reflected.

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
--radical-temperature 500  # K; enables Boltzmann energy sampling
--radical-energy-dist flux # flux (E·exp(-E/kT), default) | gas (sqrt(E)·exp(-E/kT))
--radical-energy-threshold 0.1  # eV; skip draws below it (bare flag = 0.1)
--radical-threshold-on total    # total | normal (E·cos²θ)
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

# Deposition mask
--mask-type xymask         # "xymask" | "xmask" | "ymask"
--mask-width 0.3           # fraction of box masked on each active face
--invert-mask              # deposit at frame/edges instead of center
--freeze-mask              # freeze top-surface atoms in the masked zone
--freeze-mask-depth 2.0    # Å depth from initial surface to freeze

# Burst radical injection
--radical-burst
--radical-burst-chunk 5
--radical-burst-attempt 200
--skip-radical-thermalization
```
