# DiamondEtchMD — TODO

Planned features and known gaps, roughly ordered by expected impact.

---

## Submit script

- [ ] **Auto re-submit** — after the job finishes, detect whether the target fluence has
  been reached (compare `ncarbon.txt` last line to `end_fluence`) and re-queue with
  `sbatch --dependency=afterok:$SLURM_JOB_ID $0` if not done.  Should be idempotent
  and respect the wall-hour limit so a re-queued job doesn't start with 30 s left.

- [ ] **Selectable LAMMPS module** — add a `lammps_module` field to `SimSpec` (default
  `"lammps/kokkos/gpu_della9_2022"`) and emit `module load ${lammps_module}` in the
  submit script.  Expose as `--lammps-module` in the CLI.

- [ ] **Flexible resource selection** — add fields (and CLI flags) for:
  - `ntasks` (default 1) — MPI ranks
  - `cpus_per_task` (default 1) — OpenMP threads
  - `use_gpu` (default `True`) — when `False`, omit `--gres=gpu:1` and switch the
    `srun lmp` invocation from `-k on g 1 -sf kk` to a CPU-only call
  - `mem_gb` (default 16) — memory per node

- [ ] **In-situ auto-plot** — add a `plot_interval_hours` field (default 0 = disabled).
  When set, emit a background loop in the submit script (or a companion SLURM array
  step) that calls `diamond-etch-md-plot` every N hours to regenerate etch-yield and
  O-uptake curves from the live `etch_products.txt` / `ncarbon.txt`.  The plot command
  should live in `diamond_etch_md/analysis/plot.py` and produce PNGs in the sim dir.

---

## Batch submission

- [ ] **Batch sweep** — add a `diamond-etch-md-batch` CLI (or a `make_batch()` builder)
  that accepts vectors of parameters (energy, species, reconstruction, termination,
  angle, …) and generates a directory tree + one `submit_all.sh` that loops over them.
  Parameter vectors could be specified via a TOML/YAML config file or as repeated
  `--energy 0.5 1.0 2.0` flags.  Analysis of batch results is a separate task.

---

## Ion bombardment species

- [ ] **O₂ ion** — add `"O2"` to the `SPECIES` registry.  Requires a LAMMPS molecule
  file (like `O2.molecule` in `dfiles/RIE/`) and an updated injection routine in
  `sweep.lmp` / `addfix.lmp` to insert the dimer rather than a single atom.  The
  C-H-O ReaxFF potential already supports O-O bonds; no force-field change needed.

- [ ] **Ar ion** — add `"Ar"` to `SPECIES`.  Requires extending the force field to
  4 atom types (C-H-O-Ar with ZBL) and updating `make_surf.lmp` `create_box` calls to
  `create_box 4 bbox`.  The `ffield.reax` in `dfiles/RIE/` already includes Ar/ZBL
  parameters and can be used as the template.

---

## Cycling modes

- [ ] **Ion-species cycling** — add a `cycle_species` option: a list such as
  `["Ar", "O", "O2"]` with per-species fluence counts.  The submit script (or a new
  LAMMPS driver script) iterates through the list, restarting from the latest snapshot
  between phases.  Useful for simulating alternating Ar sputtering + O passivation.

- [ ] **Flux-ratio cycling** — add a `cycle_flux_ratios` option: a list of
  `(n_radicals, n_ions)` pairs applied in sequence between ion impacts.  This enables
  simulation of dose-modulated etching without spawning separate jobs.

---

## Surface orientations

- [ ] **C(110) orientation** — add `"110"` to `ORIENT` in `orientations.py`.
  This is higher-touch than 100/111/113:
  - Derive the correct LAMMPS `lattice` orient/spacing for the (110) plane.
  - Write a `make_surf_110.lmp` template (bare surface + H/O/O_ether terminations).
  - Determine valid reconstructions (e.g. bare, 1×2 missing-row); the 1×2 requires
    removing every other surface row, which is not a simple atom displacement.
  - Compute `ml_factor` analytically and verify empirically.
  - Add integration tests analogous to the existing 100/111/113 tests.
