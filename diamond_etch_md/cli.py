"""
cli.py — command-line interface for DiamondEtchMD.

Generates a simulation directory (config.lmp, head.lmp, make_surf.lmp, submit,
and symlinks to shared scripts) from surface and bombardment parameters.

Usage:
    diamond-etch-md [options] <outdir>

Examples:
    diamond-etch-md --orientation 100 --surface 1x1 --energy 0.5 my_sim
    diamond-etch-md --orientation 100 --surface 2x1_O --energy 0.5 my_sim
    diamond-etch-md --orientation 100 --surface O_ether --energy 0.5 my_sim
    diamond-etch-md --orientation 111 --surface 2x1_pandey --energy 1.0 my_sim
    diamond-etch-md --orientation 111 --surface 2x1_pandey_O --energy 1.0 my_sim
    diamond-etch-md --species Ar --energy 100 --box-depth 10 my_sim
    diamond-etch-md --species O2 --energy 50 my_sim
"""

import argparse
from pathlib import Path

from .orientations import ORIENT
from .species import SPECIES
from .spec import SimSpec, compute_ml, validate
from .builder import make_sim, make_ale

# Default surface per orientation (used when --surface is omitted)
_DEFAULT_SURFACE = {
    "100": "1x1",
    "110": "",
    "111": "1x1",
    "113": "",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("outdir", help="Output directory for the simulation")

    surf = p.add_argument_group("surface")
    surf.add_argument(
        "--orientation", default="100", choices=list(ORIENT),
        help="Crystal surface orientation (default: 100)",
    )
    surf.add_argument(
        "--surface", default=None,
        help=(
            "Surface state (reconstruction + termination). Default: 1x1 for 100/111.\n"
            "  100: 1x1, 2x1, 2x1_O, O_ether\n"
            "  110: O  (or omit for unterminated)\n"
            "  111: 1x1, 2x1_single, 2x1_pandey, 1x1_O, 2x1_single_O, 2x1_pandey_O\n"
            "  113: O  (or omit for unterminated)"
        ),
    )
    surf.add_argument(
        "--temperature", type=float, default=300.0, metavar="K",
        help="Substrate temperature in K (default: 300)",
        dest="surface_temperature",
    )

    bomb = p.add_argument_group("bombardment")
    bomb.add_argument(
        "--species", default="O", choices=list(SPECIES),
        help="Incident species (default: O)",
    )
    bomb.add_argument(
        "--energy", type=float, default=0.5, metavar="eV",
        help="Incident particle energy in eV (default: 0.5)",
    )
    bomb.add_argument(
        "--angle", type=float, default=0.0, metavar="deg", dest="ion_angle",
        help="Ion incidence angle in degrees from surface normal (default: 0)",
    )
    bomb.add_argument(
        "--flux-ratio", type=int, default=0, dest="flux_ratio", metavar="N",
        help=(
            "RIE-etch: number of O• radicals deposited before each ion impact "
            "(0 = ion-etch, no radicals; default: 0)"
        ),
    )
    bomb.add_argument(
        "--radical-energy", type=float, default=0.2, dest="radical_energy", metavar="eV",
        help="Kinetic energy per O• radical for RIE-etch (default: 0.2 eV)",
    )
    bomb.add_argument(
        "--radical-temperature", type=float, default=None, dest="radical_temperature",
        metavar="K",
        help=(
            "Enable Maxwell-Boltzmann radical speed sampling at this temperature "
            "(overrides --radical-energy; default: None = fixed energy)"
        ),
    )
    bomb.add_argument(
        "--radical-angle", type=float, default=0.0, dest="radical_angle", metavar="deg",
        help="Radical incidence angle from surface normal in fixed-angle mode (default: 0)",
    )
    bomb.add_argument(
        "--radical-angle-distribution", action="store_true", dest="radical_angle_distribution",
        help="Enable Lambert cosine angle distribution for O• radicals",
    )
    bomb.add_argument(
        "--dump-mode", default="all", dest="dump_mode",
        choices=["all", "etch_only", "none"],
        help=(
            "Trajectory dump mode: all = every impact, "
            "etch_only = only impacts with C-containing etch or channeling, "
            "none = no dumps (default: all)"
        ),
    )

    sim = p.add_argument_group("simulation size")
    sim.add_argument(
        "--fluence", type=int, default=50, metavar="ML",
        help="Total fluence in monolayers (default: 50)",
    )
    sim.add_argument(
        "--ml", type=int, default=0,
        help="Atoms per monolayer (default: auto-computed: 100→81, 111→90, 113→108)",
    )
    sim.add_argument(
        "--box-x", type=int, default=0, dest="box_x", metavar="N",
        help="Lateral box size in x lattice units (default: orientation-specific)",
    )
    sim.add_argument(
        "--box-y", type=int, default=0, dest="box_y", metavar="N",
        help="Lateral box size in y lattice units (default: orientation-specific)",
    )
    sim.add_argument(
        "--box-depth", type=int, default=0, dest="box_depth", metavar="N",
        help="Surface depth lat_top in lattice units (default: 3)",
    )
    sim.add_argument(
        "--impact-time", type=float, default=1000.0, dest="impact_time", metavar="fs",
        help="Simulation time per ion impact in fs (default: 1000)",
    )
    sim.add_argument(
        "--thermalization-time", type=float, default=500.0,
        dest="thermalization_time", metavar="fs",
        help="Thermalization time after each impact in fs (default: 500)",
    )

    job = p.add_argument_group("SLURM")
    job.add_argument(
        "--wall-hours", type=int, default=24, dest="wall_hours",
        help="Job wall-clock time limit in hours (default: 24)",
    )
    job.add_argument(
        "--name", default="",
        help="SLURM job name (default: auto-generated from orientation/species/energy/T)",
    )
    job.add_argument(
        "--account", default="dgraves",
        help="SLURM account to charge (default: dgraves)",
    )
    job.add_argument(
        "--email", default="",
        help="Email address for SLURM END/FAIL notifications (default: none)",
    )
    job.add_argument(
        "--lammps-module", default="lammps/kokkos/gpu_della9_2022",
        dest="lammps_module",
        help="LAMMPS module to load in submit script (default: lammps/kokkos/gpu_della9_2022)",
    )
    job.add_argument(
        "--plot-interval-hours", type=int, default=12, dest="plot_interval_hours",
        metavar="H",
        help="Hours between auto-plot runs during the job (0 = disabled, default: 12)",
    )
    job.add_argument(
        "--nice", type=int, default=2,
        help="SLURM --nice priority offset (must be >= 1, default: 2)",
    )

    return p


def main():
    args = build_parser().parse_args()

    orient_cfg = ORIENT[args.orientation]
    dx, dy, dz = orient_cfg["default_box"]

    box_x = args.box_x or dx
    box_y = args.box_y or dy

    surface = args.surface if args.surface is not None else _DEFAULT_SURFACE[args.orientation]

    computed_ml = compute_ml(args.orientation, box_x, box_y)
    ml = args.ml if args.ml else computed_ml

    spec = SimSpec(
        orientation          = args.orientation,
        surface              = surface,
        surface_temperature  = args.surface_temperature,
        species              = args.species,
        energy               = args.energy,
        ion_angle            = args.ion_angle,
        fluence              = args.fluence,
        ml                   = ml,
        box_x                = box_x,
        box_y                = box_y,
        box_depth            = args.box_depth or dz,
        impact_time          = args.impact_time,
        thermalization_time  = args.thermalization_time,
        wall_hours           = args.wall_hours,
        account              = args.account,
        email                = args.email,
        lammps_module        = args.lammps_module,
        plot_interval_hours  = args.plot_interval_hours,
        nice                 = args.nice,
        flux_ratio           = args.flux_ratio,
        radical_energy       = args.radical_energy,
        radical_temperature  = args.radical_temperature,
        radical_angle        = args.radical_angle,
        radical_angle_distribution   = args.radical_angle_distribution,
        dump_mode            = args.dump_mode,
        name                 = args.name or (
            f"{args.orientation}_{args.species}_{args.energy}eV"
            f"_{int(args.surface_temperature)}K"
        ),
    )

    validate(spec)
    make_sim(spec, Path(args.outdir))


def plot_main():
    """Entry point for diamond-etch-md-plot.

    Usage:
        diamond-etch-md-plot <sim_dir> [options]

    Generates analysis plots and summary.txt in the simulation directory.
    Reads spec.json (written by make_sim) to recover SimSpec automatically.
    """
    import argparse
    from .analysis.cna import load_cna_series, update_cna_cache, save_cna_cache, _CACHE_NAME
    from .analysis.plot import make_plots
    from .analysis.summary import analyze_run, write_summary

    pp = argparse.ArgumentParser(
        description="Generate analysis plots and summary.txt for a DiamondEtchMD simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pp.add_argument("sim_dir", help="Simulation directory (must contain ncarbon.txt)")
    pp.add_argument(
        "--ml", type=int, default=0,
        help="Atoms per monolayer (auto-loaded from spec.json if 0)",
    )
    pp.add_argument(
        "--no-cna", action="store_true",
        help="Skip CNA analysis (faster; omits amorphous C plots and stats)",
    )
    pp.add_argument(
        "--cna", action="store_true",
        help="Plot using existing cna_cache.json without running any new CNA computation",
    )
    pp.add_argument(
        "--cna-run", action="store_true", dest="cna_run",
        help="Update CNA cache (compute new impacts) using cna_stride from spec.json, then plot",
    )
    pp.add_argument(
        "--cna-stride", type=int, default=1, dest="cna_stride", metavar="N",
        help="Analyze every N-th impact_snaps/*.data for CNA (default: 1 = every impact)",
    )
    pp.add_argument(
        "--n-blocks", type=int, default=10, dest="n_blocks",
        help="Number of blocks for block-averaging uncertainties (default: 10)",
    )
    args = pp.parse_args()

    sim_dir = Path(args.sim_dir)
    spec = None
    spec_path = sim_dir / 'spec.json'
    if spec_path.exists():
        from .analysis.plot import _load_spec
        spec = _load_spec(spec_path)

    ml = args.ml or (spec.ml if spec else 0)
    if ml <= 0:
        import sys
        sys.exit(
            "Error: cannot determine ml. Either pass --ml or ensure spec.json "
            "exists in the simulation directory."
        )

    cna_records = None
    data_dir = sim_dir / 'impact_snaps'
    if not data_dir.exists():
        data_dir = sim_dir / 'data_files'
    if args.cna:
        import json
        cache_path = sim_dir / _CACHE_NAME
        if cache_path.exists():
            with open(cache_path) as f:
                cna_records = json.load(f)
            print(f"Loaded CNA from cache ({len(cna_records)} snapshots, no update).")
        else:
            print("No cna_cache.json found — skipping CNA plots.")
    elif not args.no_cna and data_dir.exists():
        if args.cna_run:
            stride = (spec.cna_stride if spec and spec.cna_stride > 0 else ml)
            cache_path = sim_dir / _CACHE_NAME
            cna_records = update_cna_cache(
                cache_path, data_dir, stride=stride, verbose=True,
            )
            print(f"  CNA ready — {len(cna_records)} snapshots total.")
        else:
            stride = args.cna_stride if args.cna_stride > 0 else ml
            print(f"Computing CNA from {data_dir} (stride={stride}) ...")
            cna_records = load_cna_series(data_dir, stride=stride, verbose=True)
            print(f"  Done — {len(cna_records)} snapshots analyzed.")
    elif not args.no_cna:
        print("impact_snaps/ (or data_files/) not found — skipping CNA analysis.")

    print("Generating plots ...")
    try:
        make_plots(sim_dir, spec=spec, ml=ml, cna_records=cna_records)
    except ImportError as e:
        print(f"Warning: skipping plots ({e})")

    print("Computing summary statistics ...")
    stats = analyze_run(sim_dir, spec=spec, ml=ml,
                        n_blocks=args.n_blocks, cna_records=cna_records)
    summary_path = sim_dir / 'summary.txt'
    write_summary(stats, summary_path)

    print(f"\nOutputs written to {sim_dir}:")
    for png in sorted(sim_dir.glob('*.png')):
        print(f"  {png.name}")
    print(f"  summary.txt")


def ale_main():
    """Entry point for diamond-etch-md-ale.

    ALE-etch (Atomic Layer Etching) requires exactly 2 phases defined in Python
    code, because CLI arguments cannot express multi-phase configurations.

    To use ALE-etch, use the Python API instead:

        from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_ale
        from pathlib import Path

        spec = SimSpec(
            orientation="100",
            surface="O_ether",
            ml=compute_ml("100", 8, 8),
            box_x=8, box_y=8, box_depth=5,
            phases=[
                CyclePhase(species="Ar", energy=30.0, fluence_ml=5),
                CyclePhase(species="O2", energy=20.0, fluence_ml=5,
                           flux_ratio=10, radical_energy=0.2),
            ],
            cycles=10,
            name="my_ale_sim",
        )
        make_ale(spec, Path("my_ale_sim"))
    """
    print(
        "diamond-etch-md-ale: ALE-etch requires specifying 2 phases in Python.\n"
        "Use the Python API: make_ale(spec, outdir) where spec.phases has exactly 2 entries.\n"
        "\n"
        "Example:\n"
        "    from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_ale\n"
        "    from pathlib import Path\n"
        "\n"
        "    spec = SimSpec(\n"
        "        orientation='100', surface='O_ether',\n"
        "        ml=compute_ml('100', 8, 8), box_x=8, box_y=8, box_depth=5,\n"
        "        phases=[\n"
        "            CyclePhase(species='Ar', energy=30.0, fluence_ml=5),\n"
        "            CyclePhase(species='O2', energy=20.0, fluence_ml=5,\n"
        "                       flux_ratio=10, radical_energy=0.2),\n"
        "        ],\n"
        "        cycles=10, name='my_ale_sim',\n"
        "    )\n"
        "    make_ale(spec, Path('my_ale_sim'))\n"
    )


def status_main():
    """Entry point for diamond-etch-md-status.

    Prints a status table for one or more simulation directories showing
    impact progress, queue state, and failure flags.

    Usage:
        diamond-etch-md-status [dir ...]

    With no arguments, searches the current directory and one level of
    subdirectories for simulation directories (those containing spec.json
    or ncarbon.txt).
    """
    import argparse
    import json
    import os
    import subprocess

    pp = argparse.ArgumentParser(
        description="Print progress and queue status for DiamondEtchMD simulations.",
    )
    pp.add_argument(
        "dirs", nargs="*", default=["."],
        help="Simulation directory or parent directory to search (default: .)",
    )
    args = pp.parse_args()

    def _is_sim_dir(d: Path) -> bool:
        return d.is_dir() and ((d / "ncarbon.txt").exists() or (d / "spec.json").exists())

    sim_dirs = []
    for raw in args.dirs:
        base = Path(raw)
        if _is_sim_dir(base):
            sim_dirs.append(base)
        else:
            for sub in sorted(base.iterdir()):
                if _is_sim_dir(sub):
                    sim_dirs.append(sub)

    if not sim_dirs:
        print("No simulation directories found.")
        return

    # Query squeue once for all jobs
    queue: dict = {}  # name -> list of (state, reason)
    try:
        user = os.environ.get("USER") or subprocess.check_output(["whoami"], text=True).strip()
        sq = subprocess.run(
            ["squeue", "-u", user, "-o", "%.100j %.2t %R", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        for line in sq.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2:
                jname, state = parts[0], parts[1]
                reason = parts[2].strip() if len(parts) > 2 else ""
                queue.setdefault(jname, []).append((state, reason))
    except Exception:
        pass

    def _queue_status(name: str) -> str:
        entries = queue.get(name, [])
        if not entries:
            return "not queued"
        states = {s for s, _ in entries}
        if "R" in states:
            return "running"
        non_dep = [r for _, r in entries if r != "Dependency"]
        return f"queued ({non_dep[0]})" if non_dep else "queued (after current)"

    rows = []
    for sd in sim_dirs:
        spec = {}
        spec_path = sd / "spec.json"
        if spec_path.exists():
            try:
                spec = json.loads(spec_path.read_text())
            except Exception:
                pass

        # Determine mode label and key fields from spec
        if spec.get("phases"):
            mode = "cycling"
            species_label = "/".join(p["species"] for p in spec["phases"])
            energy_label = "/".join(str(p["energy"]) for p in spec["phases"])
        elif spec.get("ion_mix"):
            mode = "multi-ion"
            species_label = "+".join(c["species"] for c in spec["ion_mix"])
            energy_label = "mix"
        else:
            mode = "rie" if (spec.get("flux_ratio", 0) or 0) > 0 else "ion"
            species_label = str(spec.get("species", "?"))
            energy_label = f"{spec.get('energy', '?')}eV"

        name = spec.get("name", "")
        ml = spec.get("ml", 0)
        end_fluence = spec.get("fluence", 0)

        # Fall back to config.lmp for ml/end_fluence
        config_path = sd / "config.lmp"
        if config_path.exists():
            try:
                for line in config_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] == "ML" and parts[2] == "equal" and not ml:
                        ml = int(parts[3])
                    if len(parts) >= 4 and parts[1] == "end_fluence" and parts[2] == "equal" and not end_fluence:
                        end_fluence = int(parts[3])
            except Exception:
                pass

        total = ml * end_fluence if ml and end_fluence else 0

        # Read progress from ncarbon.txt
        n_done = 0
        nc_path = sd / "ncarbon.txt"
        if nc_path.exists():
            try:
                last = nc_path.read_text().strip().splitlines()[-1]
                n_done = int(last.split()[0])
            except Exception:
                pass

        failed = (sd / "LAMMPS_FAILED").exists()
        q_stat = _queue_status(name) if name else "?"

        if total > 0 and n_done >= total:
            status = "complete"
        elif failed and q_stat == "not queued":
            status = "FAILED"
        elif failed:
            status = f"FAILED + {q_stat}"
        else:
            status = q_stat

        pct = f"{n_done / total * 100:.1f}%" if total > 0 else "?"

        rows.append({
            "dir":     sd.name,
            "name":    name or "-",
            "mode":    mode,
            "species": species_label,
            "energy":  energy_label,
            "n_done":  str(n_done),
            "total":   str(total) if total else "?",
            "pct":     pct,
            "status":  status,
        })

    # Dynamic column widths
    cols = ["dir", "name", "mode", "species", "energy", "n_done", "total", "pct", "status"]
    headers = ["Directory", "Job name", "Mode", "Species", "Energy", "Done", "Total", "Progress", "Status"]
    widths = [max(len(h), max(len(r[c]) for r in rows)) for h, c in zip(headers, cols)]

    def fmt_row(vals):
        return "  ".join(v.ljust(w) if i < len(cols) - 1 else v for i, (v, w) in enumerate(zip(vals, widths)))

    print(fmt_row(headers))
    print("-" * (sum(widths) + 2 * (len(cols) - 1)))
    for r in rows:
        print(fmt_row([r[c] for c in cols]))


def view_main():
    """Entry point for diamond-etch-md-view.

    Creates (or refreshes) a viewing/ directory inside each simulation run folder,
    populating it with symlinks to the most-recent ion-impact and radical dumps
    from etch_event_trajs/.  Re-running clears old links before creating new ones.

    Dump patterns recognised:
        ion      — event_dump_<N>.dump   (digit immediately after event_dump_)
        radical  — event_dump_n*.dump    (non-burst)
                   event_dump_burst_*.dump (burst mode)

    Usage:
        diamond-etch-md-view [run_dir ...] [-n N]
    """
    import argparse
    import re

    vp = argparse.ArgumentParser(
        description="Symlink the latest ion and radical dump files into <run_dir>/viewing/.",
    )
    vp.add_argument(
        "run_dirs", nargs="*", default=["."],
        help="Simulation run directory/directories (default: current directory)",
    )
    vp.add_argument(
        "-n", "--num", type=int, default=10,
        help="Number of most-recent dumps to link for each type (default: 10)",
    )
    args = vp.parse_args()

    _ion_pat = re.compile(r"^event_dump_\d+\.dump$")

    for raw in args.run_dirs:
        run_dir = Path(raw).resolve()
        trajs   = run_dir / "etch_event_trajs"
        viewing = run_dir / "viewing"

        if not trajs.is_dir():
            print(f"{run_dir.name}: no etch_event_trajs/ found, skipping")
            continue

        viewing.mkdir(exist_ok=True)

        for link in viewing.iterdir():
            if link.is_symlink():
                link.unlink()

        all_dumps = sorted(trajs.glob("event_dump_*.dump"), key=lambda p: p.stat().st_mtime)

        ion_dumps     = [p for p in all_dumps if _ion_pat.match(p.name)]
        radical_dumps = [p for p in all_dumps if not _ion_pat.match(p.name)]

        selected = ion_dumps[-args.num:] + radical_dumps[-args.num:]

        for src in selected:
            (viewing / src.name).symlink_to(src)

        print(
            f"{run_dir.name}: viewing/ refreshed — "
            f"{min(len(ion_dumps), args.num)} ion, "
            f"{min(len(radical_dumps), args.num)} radical dumps"
        )


if __name__ == "__main__":
    main()
