# DiamondEtchMD — TODO

Planned features and known gaps, roughly ordered by expected impact.

---

## Submit script

- [x] **Auto re-submit** — after LAMMPS exits cleanly, the submit script reads
  `n_complete` from `ncarbon.txt` and compares to `end_fluence * ML`. If incomplete,
  re-queues with `sbatch "$0"`. Skips re-submit on LAMMPS failure. Also exits early
  (before running LAMMPS) if already complete, making the script idempotent.
  `--signal=B:USR1@120` + backgrounded `srun` + `wait` ensures the wall-time trap
  fires before SLURM kills the job.

- [x] **Channeled atom safeguard** — ion inner loop detects atoms that passed through
  the slab bottom (`z < bottom`) via a temporary region and deletes them each iteration,
  preventing simulation artifacts from channeled ions.

- [x] **Selectable LAMMPS module** — `lammps_module` field on `SimSpec` (default
  `"lammps/kokkos/gpu_della9_2022"`), emitted in submit script. CLI: `--lammps-module`.

- [x] **`SPEC.md`** — reference document covering all `SimSpec` / `IonComponent` /
  `CyclePhase` fields, valid values, defaults, and output file formats (`ncarbon.txt`,
  `etch_products.txt`, dump naming).

- [x] **Email notifications** — `email` field on `SimSpec`; when non-empty emits
  `#SBATCH --mail-type=END,FAIL` and `#SBATCH --mail-user=<email>` in the submit script.

- [x] **Channeling increments event counter** — when a channeled atom is detected and
  deleted, `event_count` is now incremented so downstream dump files are consistently
  numbered (`head.py` and `head_cycling.py`).

- [x] **Dump filename by impact number** — dump filenames use only the impact counter,
  not the cumulative event counter: `event_dump_${c}.dump` (ion), `event_dump_n${c}_${cn}.dump`
  (radical, `cn` = radical index), `event_dump_ion${c}.dump` (cycling ion).

- [x] **Dump mode control** — `dump_mode` field on `SimSpec` (`"all"` | `"etch_only"` | `"none"`).
  `"etch_only"` creates each dump then deletes it at end of impact unless a C-containing
  etch product or channeling event was detected; `"none"` skips all ion and radical dumps.

- [x] **Boltzmann radical energy distribution** — `radical_temperature` field on `SimSpec`
  and `CyclePhase`. When set, O• radical speed is sampled on-the-fly via a 3-component
  Box-Muller transform (σ = √(k_BT/m)) giving a Maxwell-Boltzmann speed distribution.
  Restartable: seed = f(c, cn) without file state.

- [x] **Lambert cosine radical angle distribution** — `radical_angle_distribution: bool` on `SimSpec`
  and `CyclePhase`. When `True`, samples θ = arcsin(√U), φ = 2πU₂ (full 3-D cosine law).

- [x] **Per-radical halt time** — in stochastic mode (Boltzmann or cosine),
  each radical's halt time = min(2 × `radical_i_above` / |v_z|, `max_inter_neutral_time`)
  so slow radicals still reach the 12 Å injection height surface.

- [x] **Radical velocity log + auto-plot** — every radical writes a row to `radical_log.txt`
  (impact, cn, energy_eV, polar_deg, azimuthal_deg). `make_plots()` produces
  `radical_distribution.png`: energy histogram vs MB PDF, angle histogram vs cosine PDF.

- [x] **Ion/radical angle split** — LAMMPS variable `angl` split into `ion_angl` and
  `rad_angl`. SimSpec fields: `ion_angle` (was `angle`), `radical_angle`. Backward compat
  via `SimSpec.from_dict()`.

- [x] **Field renames** — `temperature` → `surface_temperature`, `angle` → `ion_angle`.
  Old field names still accepted by `SimSpec.from_dict()`.

- [x] **Default timing update** — `impact_time` 2000 → 1000 fs; `inter_neutral_time`
  1000 → 1500 fs. Per-radical halt time replaces `inter_neutral_time` in stochastic mode.

- [ ] **Ion angle distribution** — analogous to `radical_angle_distribution` for radicals;
  add `ion_radical_angle_distribution: bool` and sample θ from a cosine or Gaussian distribution
  to model divergent ion beams or tilted-beam rastering.

- [ ] **Non-integer flux ratio** — change `flux_ratio` from `int` to `float` and use a
  Bresenham accumulator in `head.lmp` / `head_cycling.lmp` so the delivered radical count
  converges exactly to the target ratio over many impacts. The accumulator state must be
  persisted across wall-time restarts (e.g. an extra column in `ncarbon.txt`).

- [ ] **Flexible resource selection** — add fields (and CLI flags) for:
  - `ntasks` (default 1) — MPI ranks
  - `cpus_per_task` (default 1) — OpenMP threads
  - `use_gpu` (default `True`) — when `False`, omit `--gres=gpu:1` and switch the
    `srun lmp` invocation from `-k on g 1 -sf kk` to a CPU-only call
  - `mem_gb` (default 16) — memory per node

- [x] **Auto-analysis** — `diamond-etch-md-plot <sim_dir>` CLI:
  - CNA / sp3 analysis from `data_files/*.data` (post-hoc, pure numpy)
  - `amorphous.png` — amorphous C (ML) + sp3 fraction vs dose
  - `amorphous_thickness.png` — disorder depth (10%–90% density criterion) vs dose
  - `etch.png` — etch depth (ML) vs dose; phase-boundary lines for cycling
  - `o_uptake.png` — surface O (ML) vs dose
  - `product_grid.png` — 2-D count heatmap: n_C vs n_O
  - `product_trajectory.png` — cumulative yield per product species vs dose
  - `etch_per_cycle.png`, `per_phase_yield.png`, `o_per_cycle.png` (cycling only)
  - `summary.txt` — block-averaged stats: etch yield, O uptake, amorphous C/thickness,
    product composition, throughput (ML/day); per-phase breakdown for cycling
  - `spec.json` saved by `make_sim()` for automatic spec recovery by the plot tool

- [x] **In-situ auto-plot** — `plot_interval_hours` field on `SimSpec` (default 12 h).
  Submit script starts a background loop that runs `diamond-etch-md-plot . --no-cna`
  every N hours while LAMMPS is running; kills the loop on wall-time resubmit or
  normal exit.  On successful completion runs one final `--cna-stride $ML` analysis.
  Set `--plot-interval-hours 0` on the CLI to disable.

  **Remaining plot gaps (can be added to plot.py):**
  - [ ] Rolling-mean etch yield overlay on etch.png
  - [ ] Sputtering yield fit once steady-state reached (single-species)
  - [ ] ALE: saturation curves per half-cycle; surface O at end of each half-cycle
  - [ ] Radical-only: O uptake saturation curve; O/C surface ratio vs dose
  - [ ] Multi-ion/multi-RIE: stochastic composition verification — bar chart or
    running-fraction plot comparing the actual delivered species (and energy) distribution
    from `ion_impacts.txt` against the specified `ion_mix` fractions. Shows how quickly
    the realized composition converges to the target and catches any sampling bugs.

- [ ] **Optional SLURM / plain bash** — add a `use_slurm` flag (default `True`).
  When `False`, emit a plain `run.sh` instead of a SLURM `submit` script: no `#SBATCH`
  headers, no `srun`, just `lmp -k on g 1 -sf kk ...` directly. Useful for running on
  a workstation or inside a container without a scheduler.

- [x] **Overwrite protection in make scripts** — `make_sim()` now calls `squeue` before
  writing any files; if a job with the same `spec.name` is in state `R`, it aborts with
  an informative error and `scancel` hint. Skipped gracefully when `squeue` is unavailable.

- [x] **`LAMMPS_FAILED` auto-clear on resubmit** — the submit script now renames
  `LAMMPS_FAILED` to `LAMMPS_FAILED.<timestamp>` before each LAMMPS launch, so the flag
  only reflects the current run's outcome. Old failure logs are preserved for debugging.

- [x] **Status CLI** — `diamond-etch-md-status [dir ...]` prints an impact-count /
  progress / queue-status table. Auto-detects sim dirs (looks for `spec.json` or
  `ncarbon.txt`); searches one level deep if given a parent directory. Calls `squeue`
  once to get queue state for all jobs.

---

## Portability / containerization

- [ ] **Container image** — provide a `Dockerfile` that compiles LAMMPS with ReaxFF,
  Kokkos (CPU target), and ZBL support, then installs DiamondEtchMD. Publish to
  Docker Hub so users can pull it locally or convert it for HPC use.
  - **Local / workstation**: run directly with Docker + the `use_slurm=False` bash
    run script (`lmp` → `docker run ... lmp`).
  - **Cluster (Della)**: convert to Apptainer/Singularity SIF with
    `apptainer pull docker://...`; the SLURM submit script calls
    `srun apptainer exec image.sif lmp -k on g 1 -sf kk ...`.
  - Add an optional `container_image` field on `SimSpec` (default `""`); when set,
    the generated run/submit script wraps `lmp` with `apptainer exec ${container_image}`.

---

## Batch submission

- [ ] **Batch sweep** — add a `diamond-etch-md-batch` CLI (or a `make_batch()` builder)
  that accepts vectors of parameters (energy, species, reconstruction, termination,
  angle, …) and generates a directory tree + one `submit_all.sh` that loops over them.
  Parameter vectors could be specified via a TOML/YAML config file or as repeated
  `--energy 0.5 1.0 2.0` flags.  Analysis of batch results is a separate task.

---

## Ion bombardment species

- [x] **O₂ ion** — `"O2"` added to SPECIES.  Molecule file `O2.molecule` bundled in
  templates; `head.lmp` uses `fix deposit ... mol O2`; config.lmp halves energy
  per atom automatically.

- [x] **Ar ion** — `"Ar"` added to SPECIES.  All templates now use 4 atom types
  (`create_box 4`).  `head.lmp` emits hybrid ReaxFF+ZBL pair style, QEQ on
  non-Ar atoms, and post-impact Ar removal.

---

## Cycling modes

- [x] **Ion-species cycling** — `SimSpec.phases` accepts a list of `CyclePhase`
  objects (species, energy, fluence_ml, flux_ratio, radical_energy).  `builder.py`
  routes to `head_cycling.py` / `get_config_lmp_cycling()` / `get_submit_script_cycling()`.
  N-phase combinations (Ar+O2, O+O2, Ar+O+O2, …) work in any order.
  - Plain ReaxFF (3 types) used when no Ar phase present — faster.
  - Per-phase O• radical flux with per-phase radical energy.
  - Mid-radical-loop restarts via `neut_complete` variable in `ncarbon.txt` col 2.
  - Channeled atom safeguard in ion inner loop.

- [x] **Flux-ratio cycling** — implemented as `CyclePhase.flux_ratio` (O• radicals
  per ion impact) and `CyclePhase.radical_energy` per phase.  No separate option needed.

---

## Surface orientations

- [x] **C(110) orientation** — added `"110"` to `ORIENT`.  Lattice: z=[110],
  x=[-110], y=[001], spacing `2/sqrt(2), 1, 3/sqrt(2)`.  ml_factor=4
  (verified: 4×6 → 96).  Default box (4,6,5).  Bare + O terminations.
  - [ ] **1×2 missing-row reconstruction** — would require deleting every other
    surface row, which is higher-touch than displacement-based reconstructions.
