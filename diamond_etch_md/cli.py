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
from .builder import make_sim

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
        "--angle", type=float, default=0.0, metavar="deg",
        help="Incidence angle in degrees from surface normal (default: 0)",
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
        "--impact-time", type=float, default=2000.0, dest="impact_time", metavar="fs",
        help="Simulation time per impact event in fs (default: 2000)",
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
        orientation         = args.orientation,
        surface             = surface,
        temperature         = args.temperature,
        species             = args.species,
        energy              = args.energy,
        angle               = args.angle,
        fluence             = args.fluence,
        ml                  = ml,
        box_x               = box_x,
        box_y               = box_y,
        box_depth           = args.box_depth or dz,
        impact_time         = args.impact_time,
        thermalization_time = args.thermalization_time,
        wall_hours          = args.wall_hours,
        account             = args.account,
        email               = args.email,
        lammps_module       = args.lammps_module,
        plot_interval_hours = args.plot_interval_hours,
        name                = args.name or (
            f"{args.orientation}_{args.species}_{args.energy}eV_{int(args.temperature)}K"
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
    from .analysis.cna import load_cna_series
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
        "--cna-stride", type=int, default=1, dest="cna_stride", metavar="N",
        help="Analyze every N-th data_files/*.data for CNA (default: 1 = every impact)",
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
    data_dir = sim_dir / 'data_files'
    if not args.no_cna and data_dir.exists():
        print(f"Computing CNA from {data_dir} (stride={args.cna_stride}) ...")
        cna_records = load_cna_series(data_dir, stride=args.cna_stride, verbose=True)
        print(f"  Done — {len(cna_records)} snapshots analyzed.")
    elif not args.no_cna:
        print("data_files/ not found — skipping CNA analysis.")

    print("Generating plots ...")
    make_plots(sim_dir, spec=spec, ml=ml, cna_records=cna_records)

    print("Computing summary statistics ...")
    stats = analyze_run(sim_dir, spec=spec, ml=ml,
                        n_blocks=args.n_blocks, cna_records=cna_records)
    summary_path = sim_dir / 'summary.txt'
    write_summary(stats, summary_path)

    print(f"\nOutputs written to {sim_dir}:")
    for png in sorted(sim_dir.glob('*.png')):
        print(f"  {png.name}")
    print(f"  summary.txt")


if __name__ == "__main__":
    main()
