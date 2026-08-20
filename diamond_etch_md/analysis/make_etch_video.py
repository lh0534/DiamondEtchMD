"""make_etch_video.py — animate etch_trajectory plot frame-by-frame.

Generates one PNG per sampled ion impact (every `step` impacts), then
stitches them into an mp4 with ffmpeg at the given fps.

Usage:
    python -m diamond_etch_md.analysis.make_etch_video [sim_dir] [options]
    python make_etch_video.py [sim_dir] [options]

Options:
    --step N        Sample every N ion impacts (default: 2)
    --fps N         Output video frame rate (default: 15)
    --out PATH      Output mp4 path (default: <sim_dir>/etch_trajectory.mp4)
    --cna-stride N  Run CNA on impact_snaps/ every N impacts and use the
                    result as the amorphous C layer (replaces density proxy).
    --keep-frames   Keep per-frame PNGs after stitching (default: delete)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _load_deps(sim_dir: Path):
    """Load nc_records, ep_records, spec, ml from sim_dir."""
    # Try importing from the installed package first, then fall back to
    # finding the package relative to this file's location.
    try:
        from diamond_etch_md.analysis.ncarbon import parse_ncarbon
        from diamond_etch_md.analysis.etch_products import parse_etch_products
        from diamond_etch_md.spec import SimSpec
        from diamond_etch_md.analysis.plot import (
            _apply_style, _parse_ml_dump, _spec_summary_str,
            _ml_density_label, plot_etch,
        )
    except ImportError:
        # Running as standalone script — add package root to path
        pkg_root = Path(__file__).resolve().parents[2]
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))
        from diamond_etch_md.analysis.ncarbon import parse_ncarbon
        from diamond_etch_md.analysis.etch_products import parse_etch_products
        from diamond_etch_md.spec import SimSpec
        from diamond_etch_md.analysis.plot import (
            _apply_style, _parse_ml_dump, _spec_summary_str,
            _ml_density_label, plot_etch,
        )

    nc_path  = sim_dir / "ncarbon.txt"
    ep_path  = sim_dir / "etch_products.txt"
    spec_path = sim_dir / "spec.json"

    import json
    spec = None
    if spec_path.exists():
        spec = SimSpec.from_dict(json.loads(spec_path.read_text()))

    ml = spec.ml if spec else None
    if ml is None or ml <= 0:
        # Fall back: count atoms in the initial data file header
        raise ValueError("ml must be set in spec.json")

    nc_records = parse_ncarbon(nc_path)
    ep_records = parse_etch_products(ep_path) if ep_path.exists() else []

    amorphous_rho = None
    dump_path = sim_dir / "ML_impacts.dump"
    if dump_path.exists():
        try:
            amorphous_rho = _parse_ml_dump(dump_path) or None
        except Exception:
            pass

    lat_a = None
    lat_a_path = sim_dir / "lat_a.txt"
    if lat_a_path.exists():
        try:
            lat_a = float(lat_a_path.read_text().strip())
        except ValueError:
            pass

    summary      = _spec_summary_str(spec)
    ml_dens      = _ml_density_label(sim_dir, ml)

    return dict(
        nc_records=nc_records,
        ep_records=ep_records,
        spec=spec,
        ml=ml,
        amorphous_rho=amorphous_rho,
        lat_a=lat_a,
        summary=summary,
        ml_dens=ml_dens,
        _apply_style=_apply_style,
        plot_etch=plot_etch,
    )


def _fixed_axes_limits(nc_records, ep_records, ml):
    """Compute fixed x/y limits from the full dataset."""
    ion_recs = [r for r in nc_records if r['cn'] == 0 and r['impact'] > 0]
    if not ion_recs:
        return (0, 1), (0, 1)

    max_impact = max(r['impact'] for r in ion_recs)
    x_max = max_impact / ml

    # Full etched C trajectory
    carbon_ep = [r for r in ep_records if r['n_C'] > 0]
    if carbon_ep:
        max_ep = max(r['impact'] for r in carbon_ep)
        cumC = np.zeros(max(max_impact, max_ep) + 1)
        for r in carbon_ep:
            cumC[r['impact']] += r['n_C']
        depth_y_full = np.cumsum(cumC) / ml
        y_max_etch = float(depth_y_full[-1])
    else:
        y_max_etch = 0.0

    o_ml_full = np.array([r['n_oxygen'] / ml for r in ion_recs])
    y_max_o   = float(o_ml_full.max()) if len(o_ml_full) else 0.0

    y_max = max(y_max_etch, y_max_o) * 1.1 or 1.0

    return (0, x_max * 1.02), (0, y_max)


def render_frames(sim_dir: Path, step: int, frames_dir: Path,
                  cna_records=None) -> list[Path]:
    """Render one PNG per sampled impact into frames_dir. Returns sorted list."""
    deps = _load_deps(sim_dir)
    deps["_apply_style"]()

    nc_records    = deps["nc_records"]
    ep_records    = deps["ep_records"]
    spec          = deps["spec"]
    ml            = deps["ml"]
    # CNA records take priority over the density proxy when provided
    amorphous_rho = None if cna_records else deps["amorphous_rho"]
    lat_a         = deps["lat_a"]
    summary       = deps["summary"]
    ml_dens       = deps["ml_dens"]
    plot_etch     = deps["plot_etch"]

    # Sorted unique ion impact indices (cn==0, impact>0)
    ion_impacts = sorted({r['impact'] for r in nc_records
                          if r['cn'] == 0 and r['impact'] > 0})

    xlim, ylim = _fixed_axes_limits(nc_records, ep_records, ml)
    is_cyc     = bool(spec and spec.phases)
    plot_spec  = spec if is_cyc else None

    sampled = ion_impacts[::step]
    paths   = []

    for frame_idx, impact in enumerate(sampled):
        nc_sub  = [r for r in nc_records if r['impact'] <= impact]
        ep_sub  = [r for r in ep_records if r['impact'] <= impact and r['n_C'] > 0]

        # amorphous C layer — CNA records take priority over density proxy
        cna_sub = None
        rho_sub = None
        if cna_records:
            cna_sub = [r for r in cna_records if r['impact'] <= impact] or None
        elif amorphous_rho:
            dose_now = impact / ml
            rho_sub  = [(d, t) for d, t in amorphous_rho if d <= dose_now] or None

        fig = plot_etch(
            nc_sub, ml,
            spec=plot_spec,
            ep_records=ep_sub,
            cna_records=cna_sub,
            amorphous_rho=rho_sub,
            lat_a=lat_a,
            spec_summary=summary,
            ml_density_str=ml_dens,
        )
        if fig is None:
            # plot_etch returns None when nc_records is empty — shouldn't happen
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")

        # Fix axes to full-simulation limits so they don't rescale per frame
        for ax in fig.axes:
            if ax.get_xlabel() or ax.lines:
                ax.set_xlim(*xlim)
                # Only fix left (primary) y-axis; twin right axis scales with it
                if ax.get_ylabel() != "":
                    ax.set_ylim(*ylim)
                break

        out_png = frames_dir / f"frame_{frame_idx:05d}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_png)

        if (frame_idx + 1) % 50 == 0:
            print(f"  {frame_idx + 1}/{len(sampled)} frames rendered "
                  f"(impact {impact})")

    return paths


def stitch_video(frames_dir: Path, fps: int, out_path: Path):
    """Call ffmpeg to stitch PNGs into mp4."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # libx264 requires even dimensions
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sim_dir", nargs="?", default=".",
                    help="Simulation directory (default: .)")
    ap.add_argument("--step", type=int, default=2,
                    help="Sample every N ion impacts (default: 2)")
    ap.add_argument("--fps",  type=int, default=15,
                    help="Output frame rate (default: 15)")
    ap.add_argument("--out",  default=None,
                    help="Output mp4 path (default: <sim_dir>/etch_trajectory.mp4)")
    ap.add_argument("--cna-stride", type=int, default=None, metavar="N",
                    help="Compute CNA on impact_snaps/ every N impacts and use "
                         "as amorphous C layer (replaces density proxy)")
    ap.add_argument("--keep-frames", action="store_true",
                    help="Keep per-frame PNGs after stitching")
    args = ap.parse_args()

    sim_dir  = Path(args.sim_dir).resolve()
    out_path = Path(args.out) if args.out else sim_dir / "etch_trajectory.mp4"

    frames_dir = sim_dir / "_etch_video_frames"
    frames_dir.mkdir(exist_ok=True)

    cna_records = None
    if args.cna_stride:
        from diamond_etch_md.analysis.cna import load_cna_series
        data_dir = sim_dir / "impact_snaps"
        print(f"Computing CNA (stride={args.cna_stride}) on {data_dir} …")
        cna_records = load_cna_series(data_dir, stride=args.cna_stride, verbose=True)
        print(f"CNA done: {len(cna_records)} records.")

    print(f"Rendering frames (step={args.step}) into {frames_dir} …")
    paths = render_frames(sim_dir, args.step, frames_dir, cna_records=cna_records)
    print(f"Rendered {len(paths)} frames.")

    print(f"Stitching video at {args.fps} fps → {out_path} …")
    stitch_video(frames_dir, args.fps, out_path)
    print(f"Video saved: {out_path}")

    if not args.keep_frames:
        shutil.rmtree(frames_dir)
        print("Frame directory cleaned up.")


if __name__ == "__main__":
    main()
