# DiamondEtchMD

Python package for setting up and analysing LAMMPS ReaxFF molecular-dynamics
simulations of diamond surface etching. Handles three physically relevant
bombardment regimes — **theory-etch** (ion-only baseline), **RIE-etch**
(reactive-ion etching with simultaneous radical exposure), and **cycle-etch**
(multi-phase alternating-species cycling, including **ALE-etch**) — across four
crystal orientations and a range of surface reconstructions and terminations.
Produces simulation directories that are ready to submit to a SLURM cluster or
run locally. Requires a GPU.

---

## Etching modes

### theory-etch — ion bombardment baseline

Single-species ion bombardment with no radical co-exposure. Ions of one type
(O⁺, O₂⁺, or Ar⁺) are delivered one at a time at a specified energy and angle.

Use theory-etch to establish a baseline etch yield before adding radical
exposure. Comparing a theory-etch run to the equivalent RIE-etch run (same
species, energy, and fluence; `flux_ratio=0` vs `flux_ratio>0`) isolates the
**ion–radical synergy** — the excess etching that arises from the combined
action of ions and radicals beyond the sum of their individual contributions.

### RIE-etch — reactive-ion etching

A fixed number of O• radicals (`flux_ratio`) are deposited onto the surface
before each ion impact. The radical pre-exposure builds up surface oxygen, which
is then driven off as volatile COₓ by the subsequent ion. This is the standard
atomistic model for plasma-assisted reactive-ion etching of carbon.

`flux_ratio` is the number of O• radicals per ion impact. `radical_energy` is
the kinetic energy of each radical. The default of 0.2 eV is not thermal — it
is intentionally elevated to the ~95th percentile of the Maxwell–Boltzmann
distribution at 600 K, which compensates for the artificially low flux ratios
accessible in MD (real plasmas deliver orders of magnitude more radicals per
ion than can be simulated impact-by-impact).

To quantify ion–radical synergy, run a matched theory-etch at identical
parameters with `flux_ratio=0`.

### cycle-etch / ALE-etch — multi-phase cycling

Two or more ion species (or the same species at different conditions) alternate
in a repeating phase sequence. Each phase specifies its own species, energy,
fluence in ML per cycle, and an optional radical flux. The phase list repeats
for `cycles` repetitions.

**ALE-etch** (Atomic Layer Etching) is cycle-etch restricted to exactly two
phases — typically a surface modification phase followed by a layer-removal
phase. Use `make_ale()` in place of `make_sim()` to enforce the 2-phase
constraint. ALE-etch with `flux_ratio=0` in every phase reduces to alternating
theory-etch cycles.

`etch_mode(spec)` returns `"theory-etch"`, `"rie-etch"`, or `"cycle-etch"` for
any `SimSpec`.

---

## How simulations work

Each simulation proceeds as a sequence of independent impact events:

1. **Surface initialisation** — the surface slab is built (`make_surf.lmp`) and
   thermalized to the target temperature. The bottommost atomic layer is frozen
   throughout the run to prevent rigid-body translation.

2. **Radical pre-exposure** (RIE-etch and cycle-etch phases with `flux_ratio>0`)
   — before each ion impact, `flux_ratio` O• radicals are deposited one at a
   time from above the surface. Each radical is propagated in the NVE ensemble
   for `inter_neutral_time` fs.

3. **Ion injection** — a single ion is placed above the surface with the
   specified kinetic energy and incidence angle. The simulation runs in the NVE
   (microcanonical) ensemble for `impact_time` fs, allowing full energy
   transfer without thermostat bias.

4. **Etch product detection** — if a cluster of atoms separates from the surface,
   rises above all other clusters, and has a positive z-velocity component, it
   is identified as an etch product, removed from the cell, and recorded in
   `etch_products.txt`.

5. **Thermalisation** — after the impact, the mobile atoms are thermalised in the
   NVT ensemble for `thermalization_time` fs using a Nosé–Hoover chain, returning
   the substrate to the target temperature.

6. **Inert-ion removal** — Ar⁺ is chemically inert and is deleted from the cell
   after each impact by default (`remove_ar=True`). Set `remove_ar=False` to
   retain Ar for studies of implantation or trapping.

7. **Carbon replenishment** — if the number of C atoms in the cell falls below
   the initial lattice count (due to etching), a new layer of carbon is added to
   the bottom of the slab and thermalized. This maintains a constant slab
   thickness over long runs.

8. **Snapshots** — after every impact, a full atomic snapshot is written to
   `impact_snaps/[impact_number].data`. Every time one monolayer (ML) of ions
   has been delivered, a snapshot is also appended to `ML_impacts.dump` in the
   run directory. Detailed per-event atom trajectories are written to
   `etch_event_trajs/`. Simulation directories can exceed **1 GB** of disk
   space for long runs.

To consolidate all per-impact snapshots into a single trajectory file for
visualisation, use the bundled script (symlinked into each simulation directory):

```bash
python make_impact_dump.py [sim_dir]   # → sim_dir/all_impacts.dump
```

---

## Output data

| File / directory | Content |
|---|---|
| `ncarbon.txt` | Atom counts after every event — used to compute etch depth vs dose |
| `etch_products.txt` | Ejected cluster log: composition (n_C, n_H, n_O), velocity vector |
| `impact_snaps/` | Full atomic snapshots at every impact (positions, charges) |
| `ML_impacts.dump` | LAMMPS dump appended once per monolayer — compact long-range trajectory |
| `etch_event_trajs/` | Per-event atom trajectories through each impact |
| `all_impacts.dump` | Consolidated single-file trajectory (generated by `make_impact_dump.py`) |
| `summary.txt` | Block-averaged statistics (written by `diamond-etch-md-plot`) |

### Extractable quantities

**Etch yield** — C atoms removed per ion impact (atoms/ion or ML/ion), from
`etch_products.txt`.

**Surface species** — O uptake (surface O atoms vs dose), from `ncarbon.txt`.

**Etch product distribution** — counts of CO, CO₂, C₂, C₂O, … as a 2-D
heatmap and cumulative yield trajectories.

**Ion–radical synergy** — run a theory-etch and a RIE-etch at identical
conditions (`flux_ratio=0` and `flux_ratio>0`), then compare etch yields. The
synergy is the excess above the individual contributions.

**Etch per cycle** — for cycle-etch simulations, etch depth and O coverage are
decomposed per phase boundary.

**sp3 / amorphous layer** — post-hoc common-neighbour analysis from
`impact_snaps/*.data` gives sp3 fraction and amorphous layer thickness vs dose.

**Atom trajectories** — full 6-DOF trajectories in `etch_event_trajs/` and
`ML_impacts.dump`. Visualise with OVITO, VMD, or any LAMMPS-dump reader.

---

## LAMMPS and the force field

### Running LAMMPS

DiamondEtchMD uses LAMMPS with the **Kokkos/GPU** backend and requires at
least one GPU per job. On Princeton's Della cluster, LAMMPS is loaded via
environment module:

```bash
module load lammps/kokkos/gpu_della9_2022
```

For portability — other clusters, workstations, or containers — LAMMPS can be
run through an **Apptainer/Singularity** SIF image. The `container_image` field
on `SimSpec` (planned; see TODO) will wrap the `lmp` invocation automatically.
The `lmp_env.sh` template (symlinked into every simulation directory) sets
environment variables for the Kokkos GPU run.

### Force field

All simulations use the **ReaxFF** reactive force field for the C-H-O system,
with the **ZBL** (Ziegler-Biersack-Littmark) screened nuclear repulsion
correction applied to Ar-involved interactions. The parameterisation
(`ffield.reax`) is from:

> Draney J S, Vella J R, Panagiotopoulos A Z and Graves D B 2025
> "Atomic scale etching of diamond: insights from molecular dynamics simulations"
> *J. Phys. D: Appl. Phys.* **58** 025206.
> <https://doi.org/10.1088/1361-6463/ad78e6>

If you use this package, please also cite the paper in which these simulation
methods were originally applied:

> Draney J S et al. 2026
> "Plasma-assisted atomic layer etching of single-crystal diamond"
> *J. Vac. Sci. Technol. A* **44** 032603.
> <https://doi.org/10.1116/6.0005266>

---

## Supported surfaces and species

### Crystal orientations and surface states

| Orientation | Surface key | Description |
|---|---|---|
| `100` | `1x1` | Unreconstructed |
| `100` | `2x1` | 2×1 dimer-row reconstruction |
| `100` | `2x1_O` | 2×1 + ketone O atop surface C |
| `100` | `O_ether` | O bridging between adjacent surface C |
| `110` | `""` | Unterminated |
| `110` | `O` | O-terminated |
| `111` | `1x1` | Unreconstructed |
| `111` | `2x1_single` | 2×1 single-chain |
| `111` | `2x1_pandey` | 2×1 Pandey chain |
| `111` | `1x1_O` | 1×1 + O |
| `111` | `2x1_single_O` | 2×1 single-chain + O |
| `111` | `2x1_pandey_O` | 2×1 Pandey chain + O |
| `113` | `""` | Unterminated |
| `113` | `O` | O-terminated |

### Ion species

| Species | Modes | Force field | Notes |
|---|---|---|---|
| `O` | theory, RIE, cycle | ReaxFF (C-H-O) | Single atom; serves as O⁺ ion or O• radical |
| `O2` | theory, cycle | ReaxFF (C-H-O) | Injected as dimer; `energy` is total dimer KE (halved per atom) |
| `Ar` | theory, cycle | ReaxFF + ZBL | Inert; removed after each impact by default; cannot be used in RIE-etch |

---

## Examples

Simulation directories are created in Python using `make_sim()` or `make_ale()`.
The CLI (`diamond-etch-md`) exists for scripted use but is not recommended for
most research workflows — set parameters in Python where they are
version-controlled alongside your analysis code.

### theory-etch

[`examples/O_radical_100.py`](examples/O_radical_100.py) — O⁺ bombardment
of C(100) O-ether surface at 0.5 eV, no radical co-exposure.

```python
from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim

spec = SimSpec(
    orientation    = "100",
    surface        = "O_ether",
    temperature    = 300.0,
    species        = "O",
    energy         = 0.5,          # eV
    fluence        = 50,           # ML
    ml             = compute_ml("100", 9, 9),  # 81
    box_x=9, box_y=9, box_depth=3,
    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_Oether_O_0.5eV",
)
make_sim(spec, Path("theory_O_0.5eV"))
```

See also: [`Ar_sputtering_100.py`](examples/Ar_sputtering_100.py),
[`O2_bombardment_111.py`](examples/O2_bombardment_111.py),
[`O_etching_113.py`](examples/O_etching_113.py),
[`O_etching_110.py`](examples/O_etching_110.py).

### RIE-etch

[`examples/RIE_etching_100.py`](examples/RIE_etching_100.py) — O⁺ ions at
20 eV with 5 O• radicals (0.2 eV each) deposited before each ion impact.

```python
from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate

spec = SimSpec(
    orientation    = "100",
    surface        = "1x1",
    temperature    = 300.0,
    species        = "O",
    energy         = 20.0,         # eV
    fluence        = 50,
    ml             = compute_ml("100", 9, 9),
    box_x=9, box_y=9, box_depth=5,
    flux_ratio     = 5,            # 5 O• radicals before each O⁺ impact
    radical_energy = 0.2,          # eV per radical (≈ 95th percentile at 600 K)
    wall_hours     = 24,
    account        = "dgraves",
    name           = "100_1x1_O20eV_R5_RIE",
)
validate(spec)
make_sim(spec, Path("RIE_O20eV_R5"))

# Matched theory-etch baseline for ion-radical synergy analysis:
import dataclasses
spec_base = dataclasses.replace(spec, flux_ratio=0, name="100_1x1_O20eV_theory")
make_sim(spec_base, Path("theory_O20eV"))
```

### cycle-etch / ALE-etch

[`examples/cycling_Ar_O2_100.py`](examples/cycling_Ar_O2_100.py) — three
cycle-etch variants (2-phase Ar⁺→O₂⁺, 2-phase O⁺→O₂⁺, 3-phase Ar⁺→O⁺→O₂⁺).

```python
from pathlib import Path
from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, make_ale, validate

ml = compute_ml("100", 8, 8)  # 64 atoms/ML

# ALE-etch: Ar⁺ surface modification → O₂⁺ layer removal (2-phase)
spec = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    temperature = 300.0,
    ml=ml, box_x=8, box_y=8, box_depth=5,
    phases = [
        CyclePhase(species="Ar", energy=30.0, fluence_ml=5),
        CyclePhase(species="O2", energy=20.0, fluence_ml=5,
                   flux_ratio=10, radical_energy=0.2),
    ],
    cycles     = 10,
    wall_hours = 48,
    account    = "dgraves",
    name       = "100_Oether_Ar30eV_O2_20eV_R10_x10",
)
validate(spec)
make_ale(spec, Path("ALE_Ar_O2"))   # validates exactly 2 phases

# 3-phase cycle-etch: Ar⁺ → O⁺ → O₂⁺
spec3 = SimSpec(
    orientation = "100",
    surface     = "O_ether",
    temperature = 300.0,
    ml=ml, box_x=8, box_y=8, box_depth=6,
    phases = [
        CyclePhase(species="Ar", energy=50.0, fluence_ml=5),
        CyclePhase(species="O",  energy=1.0,  fluence_ml=3, flux_ratio=5),
        CyclePhase(species="O2", energy=20.0, fluence_ml=5, flux_ratio=10),
    ],
    cycles = 5,
    name   = "100_3phase_Ar_O_O2",
)
make_sim(spec3, Path("cycle_3phase"))
```

---

## Dispatching simulations

`make_sim()` writes `config.lmp`, `head.lmp`, `make_surf.lmp`, a `submit` SLURM
script, and symlinks to shared templates into the output directory.

### SLURM (Della or any SLURM cluster)

```bash
sbatch ALE_Ar_O2/submit
sbatch RIE_O20eV_R5/submit
sbatch theory_O_0.5eV/submit
```

The submit script:
- Builds the initial surface (`make_surf.lmp`) on first run, skips it on restart
- Detects resume state from `ncarbon.txt` and the latest `impact_snaps/` snapshot
- Re-queues itself automatically if the wall-time limit is reached before the
  target fluence is complete
- Runs a background `diamond-etch-md-plot` loop every `plot_interval_hours`
  hours during the job (default 12 h; set to 0 to disable)
- Runs a final CNA-strided analysis on normal completion

### Plain bash (no scheduler)

The submit script can be run directly as a bash script — the `#SBATCH` header
lines are silently ignored. Replace `srun lmp` with a plain `lmp` call if
`srun` is not available:

```bash
bash ALE_Ar_O2/submit
```

(A `use_slurm=False` option that emits a pure `run.sh` is planned; see TODO.)

### On-demand plot refresh

A symlinked `auto-plot.py` in each simulation directory regenerates all plots
and `summary.txt`:

```bash
python ALE_Ar_O2/auto-plot.py
python ALE_Ar_O2/auto-plot.py --no-cna        # skip CNA (faster during live runs)
python ALE_Ar_O2/auto-plot.py --cna-stride 10
```

### Consolidating impact snapshots

```bash
python ALE_Ar_O2/make_impact_dump.py          # → ALE_Ar_O2/all_impacts.dump
```

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9. Core package has no third-party dependencies.
Analysis and plotting require numpy and matplotlib:

```bash
pip install -e ".[analysis]"
```

---

## Package layout

```
diamond_etch_md/
  spec.py          SimSpec dataclass, compute_ml(), validate(), etch_mode()
  orientations.py  ORIENT registry — lattice commands, ML factors, surface definitions
  species.py       SPECIES registry — atom types, injection heights, ZBL/molecule flags
  builder.py       make_sim(), make_ale()
  cli.py           diamond-etch-md, diamond-etch-md-plot entry points
  lammps/
    config.py         get_config_lmp(), get_config_lmp_cycle_etch()
    head.py           get_head_lmp()            — theory-etch and RIE-etch driver
    head_cycling.py   get_head_lmp_cycle_etch() — cycle-etch multi-phase driver
    submit.py         get_submit_script(), get_submit_script_cycle_etch()
    templates/
      make_surf_100.lmp, make_surf_110.lmp, make_surf_111_*.lmp, make_surf_113.lmp
      O2.molecule, ffield.reax, lat_a.txt
      sweep.lmp, thermalize.lmp, addfix.lmp, lmp_env.sh
      auto-plot.py, make_impact_dump.py
  analysis/
    etch_products.py  parse_etch_products(), etch_yield()
    ncarbon.py        parse_ncarbon(), etch_depth(), is_cycling_format()
    cna.py            load_cna_series(), sp3_mask(), amorphous_thickness_angstrom()
    plot.py           make_plots()
    summary.py        analyze_run(), write_summary()
```

---

## SimSpec reference

| Field | Type | Default | Description |
|---|---|---|---|
| `orientation` | str | `"100"` | Crystal surface: `"100"`, `"110"`, `"111"`, or `"113"` |
| `surface` | str | `"1x1"` | Surface state — reconstruction + termination |
| `temperature` | float | `300.0` | Substrate temperature (K) |
| `species` | str | `"O"` | Incident ion: `"O"`, `"Ar"`, `"O2"` (single-species modes) |
| `energy` | float | `0.5` | Incident particle energy (eV); total dimer KE for O₂ |
| `angle` | float | `0.0` | Incidence angle from surface normal (degrees) |
| `fluence` | int | `50` | Total fluence in ML (single-species modes) |
| `flux_ratio` | int | `0` | O• radicals per ion impact — `0` = theory-etch, `>0` = RIE-etch |
| `radical_energy` | float | `0.2` | eV per O• radical (RIE-etch only) |
| `ml` | int | `0` | Atoms per monolayer; `0` triggers `compute_ml()` |
| `box_x` | int | `9` | Lateral box size, x (lattice units) |
| `box_y` | int | `9` | Lateral box size, y (lattice units) |
| `box_depth` | int | `3` | Surface slab depth `lat_top` (lattice units) |
| `impact_time` | float | `2000.0` | NVE time per ion impact (fs) |
| `thermalization_time` | float | `500.0` | NVT thermalisation after each impact (fs) |
| `inter_neutral_time` | float | `1000.0` | NVE time per O• radical impact (fs) |
| `remove_ar` | bool | `True` | Delete Ar atoms after each impact; set `False` to retain |
| `wall_hours` | int | `24` | SLURM wall-clock limit (hours) |
| `name` | str | `""` | SLURM job name (auto-generated if empty) |
| `account` | str | `"dgraves"` | SLURM account to charge |
| `email` | str | `""` | Email for END/FAIL notifications; empty = no mail |
| `lammps_module` | str | `"lammps/kokkos/gpu_della9_2022"` | LAMMPS module loaded in submit script |
| `plot_interval_hours` | int | `12` | Hours between auto-plot runs during job (0 = disabled) |
| `phases` | list\|None | `None` | `CyclePhase` list for cycle-etch; `None` = single-species |
| `cycles` | int | `1` | Number of phase-list repetitions (cycle-etch only) |

### Atoms per monolayer

`compute_ml(orientation, box_x, box_y)` = `ml_factor × box_x × box_y`:

| Orientation | `ml_factor` | Example (default box) |
|---|---|---|
| `100` | 1 | 9×9 → 81 |
| `110` | 4 | 4×6 → 96 |
| `111` | 2 | 5×9 → 90 |
| `113` | 4 | 9×3 → 108 |

### Box depth guidance

| Ion energy | Recommended `box_depth` |
|---|---|
| ≤ 20 eV | 5 |
| 50 eV | 6 |
| 100 eV | 10 |
| 200 eV | 12 |

For low-energy radicals (< 1 eV) the default of 3 is sufficient.

---

## CyclePhase fields

| Field | Type | Default | Description |
|---|---|---|---|
| `species` | str | — | Ion species: `"O"`, `"O2"`, `"Ar"` |
| `energy` | float | — | Ion energy in eV (total dimer energy for O₂) |
| `fluence_ml` | int | — | ML of this species per cycle repetition |
| `flux_ratio` | int | `0` | O• radicals deposited before each ion impact (0 = none) |
| `radical_energy` | float | `0.2` | eV per O• radical |

When no phase uses Ar, plain ReaxFF (3 atom types) is used — faster than the
4-type ZBL hybrid.

---

## Analysis

```python
from diamond_etch_md.analysis.etch_products import parse_etch_products, etch_yield
from diamond_etch_md.analysis.ncarbon import parse_ncarbon, etch_depth

# Etch yield (C atoms ejected per ion impact)
records = parse_etch_products("my_sim/etch_products.txt")
y = etch_yield(records, ml=81)

# Etch depth vs impact number (in monolayers)
nc = parse_ncarbon("my_sim/ncarbon.txt")
depths = etch_depth(nc, ml=81, box_x=9, box_y=9, orientation="100")
```

`diamond-etch-md-plot <sim_dir>` generates all plots and `summary.txt`:

| Plot | Content |
|---|---|
| `etch.png` | Etch depth (ML) vs dose; phase-boundary lines for cycle-etch |
| `o_uptake.png` | Surface O (ML) vs dose |
| `product_grid.png` | 2-D count heatmap: n_C vs n_O per ejected cluster |
| `product_trajectory.png` | Cumulative yield per product species vs dose |
| `amorphous.png` | Amorphous C (ML) + sp3 fraction vs dose |
| `amorphous_thickness.png` | Disorder depth (10%–90% density criterion) vs dose |
| `etch_per_cycle.png` | Etch depth per cycle (cycle-etch only) |
| `per_phase_yield.png` | Per-phase etch yield breakdown (cycle-etch only) |
| `o_per_cycle.png` | O loading/unloading per cycle (cycle-etch only) |

### Output file formats

**`etch_products.txt`** — one line per ejected cluster:
```
impact#  atom_type  n_C  n_H  n_O  vx  vy  vz
```

**`ncarbon.txt`** — theory-etch (4 columns):
```
impact#  n_carbon  n_hydrogen  n_oxygen
```

**`ncarbon.txt`** — RIE-etch and cycle-etch (5 columns):
```
impact#  radical#  n_carbon  n_hydrogen  n_oxygen
```
`radical# > 0` after each O• radical (1-indexed); `radical# = 0` after each ion
impact. This format enables mid-radical-loop restarts after wall-time preemption.

---

## All examples

[`example_all_options.py`](example_all_options.py) — all three ion species with
annotations on every field.

| File | Mode | What it demonstrates |
|---|---|---|
| [`O_radical_100.py`](examples/O_radical_100.py) | theory-etch | O⁺ on C(100) O-ether surface |
| [`Ar_sputtering_100.py`](examples/Ar_sputtering_100.py) | theory-etch | Ar⁺ physical sputtering of C(100) 1×1 at 100 eV |
| [`O2_bombardment_111.py`](examples/O2_bombardment_111.py) | theory-etch | O₂⁺ dimer on C(111) Pandey chain |
| [`O_terminated_111_pandey.py`](examples/O_terminated_111_pandey.py) | theory-etch | C(111) Pandey chain + O-terminated |
| [`O_etching_113.py`](examples/O_etching_113.py) | theory-etch | C(113) O-terminated surface |
| [`O_etching_110.py`](examples/O_etching_110.py) | theory-etch | C(110) bare and O-terminated |
| [`angled_Ar_100.py`](examples/angled_Ar_100.py) | theory-etch | 45° off-normal Ar⁺ incidence |
| [`high_energy_O_100.py`](examples/high_energy_O_100.py) | theory-etch | 200 eV O⁺ with deep slab |
| [`RIE_etching_100.py`](examples/RIE_etching_100.py) | RIE-etch | O⁺ at 20 eV + O• pre-exposure (flux_ratio=5) |
| [`cycling_Ar_O2_100.py`](examples/cycling_Ar_O2_100.py) | cycle/ALE | Ar⁺→O₂⁺, O⁺→O₂⁺, 3-phase Ar⁺→O⁺→O₂⁺ |

---

## Running tests

```bash
pytest           # unit + integration tests (no cluster required)
pytest -m slurm  # also submit a real SLURM job (requires Della)
```

---

## Roadmap

Planned features are tracked in [`TODO.md`](TODO.md).

---

## Credits

**DiamondEtchMD** was developed by
[Louis E.S. Hoffenberg](mailto:lhoff@princeton.edu) and
[Jack S. Draney](mailto:jackdraney@princeton.edu)
in the [Graves Lab](https://graveslab.princeton.edu) at Princeton University.

If you use this package in published work, please cite the force-field paper and
the methods paper:

> Draney J S, Vella J R, Panagiotopoulos A Z and Graves D B 2025
> "Atomic scale etching of diamond: insights from molecular dynamics simulations"
> *J. Phys. D: Appl. Phys.* **58** 025206.
> <https://doi.org/10.1088/1361-6463/ad78e6>

> Draney J S et al. 2026
> "Plasma-assisted atomic layer etching of single-crystal diamond"
> *J. Vac. Sci. Technol. A* **44** 032603.
> <https://doi.org/10.1116/6.0005266>

A `CITATION.cff` file is included for automated citation generation.
