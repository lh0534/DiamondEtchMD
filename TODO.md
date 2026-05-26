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
