"""
render.py — headless video rendering from LAMMPS dump files via OVITO.

Supports local parallel rendering (for vis nodes) and SLURM batch submission.
Uses xvfb + OpenGL renderer so output matches the OVITO GUI appearance.

The .ovito session file must have been saved from OVITO Basic; this module
patches the header on-the-fly so the OVITO Python package can load it.
"""

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Ovito session file patching
# ---------------------------------------------------------------------------

def patch_ovito_scene(src: Path, dst: Path) -> None:
    """Patch an OVITO Basic session file so the Python API can load it.

    OVITO Basic writes 'OVITO Basic' (22 bytes UTF-16BE) in the file header.
    The Python package only accepts 'OVITO Pro'. We replace the product name
    and compensate the lost 4 bytes by padding the version string with spaces,
    keeping the header exactly 70 bytes so absolute offsets in chunk data are
    preserved.
    """
    with open(src, "rb") as f:
        orig = f.read()
    if orig[20:38] == "OVITO Pro".encode("utf-16-be"):
        with open(dst, "wb") as f:     # saved by the Python module: already loadable
            f.write(orig)
        return

    product     = "OVITO Pro".encode("utf-16-be")   # 18 bytes (was 22)
    version_str = "3.15.0  ".encode("utf-16-be")    # 16 bytes (was 12) → net 0

    new_header = (
        orig[:8]
        + struct.pack(">II", 30016, 8)
        + struct.pack(">I", len(product)) + product
        + struct.pack(">III", 3, 15, 0)
        + struct.pack(">I", len(version_str)) + version_str
    )
    assert len(new_header) == 70, f"Header size mismatch: {len(new_header)}"

    with open(dst, "wb") as f:
        f.write(new_header + orig[70:])


def _run_with_xvfb(cmd, env=None, screen="1280x800x24"):
    """Run cmd against a private Xvfb server; returns the CompletedProcess.

    Xvfb -displayfd claims a free display number itself and reports it on a
    pipe, so any number of concurrent renders (parallel auto-plots, pool
    workers) get distinct displays.  xvfb-run -a and fixed display numbers
    race: two servers pick the same number, one dies, and xvfb-run's cleanup
    fails with "kill: ... No such process".
    """
    r, w = os.pipe()
    xvfb = subprocess.Popen(
        ["Xvfb", "-displayfd", str(w), "-screen", "0", screen, "-nolisten", "tcp"],
        pass_fds=(w,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.close(w)
    try:
        with os.fdopen(r) as f:
            display = f.readline().strip()
        if not display:
            raise RuntimeError("Xvfb failed to start")
        run_env = dict(os.environ if env is None else env)
        run_env["DISPLAY"] = f":{display}"
        return subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    finally:
        xvfb.terminate()
        try:
            xvfb.wait(timeout=5)
        except subprocess.TimeoutExpired:
            xvfb.kill()


# ---------------------------------------------------------------------------
# Scene / camera helpers (run in the main process, ovito already imported)
# ---------------------------------------------------------------------------

def _load_scene(scene_path, dump_path):
    """Load patched scene + dump; return (pipeline, render_settings, viewport)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import ovito
        ovito.scene.load(str(scene_path))
        p = ovito.scene.pipelines[0]
        p.source.load(str(dump_path))
        return p, ovito.scene.render_settings, ovito.scene.viewports.active_vp


def _apply_camera_adjustments(vp, zoom=1.0, z_shift=0.0):
    """Scale FOV and translate camera vertically (world z-axis)."""
    import math
    if zoom != 1.0:
        vp.fov = vp.fov / zoom
    if z_shift != 0.0:
        pos = list(vp.camera_pos)
        pos[2] += z_shift
        vp.camera_pos = tuple(pos)


def _render_single_frame(vp, rs, frame_idx, out_path, cam_overrides=None):
    """Render one frame with optional camera overrides; return path."""
    from ovito.vis import OpenGLRenderer
    if cam_overrides:
        _apply_camera_adjustments(vp, **cam_overrides)
    vp.render_image(
        size=tuple(rs.size),
        background=tuple(float(x) for x in rs.background_color),
        renderer=OpenGLRenderer(),
        frame=frame_idx,
        filename=str(out_path),
    )
    return out_path


# ---------------------------------------------------------------------------
# Single-frame preview via xvfb subprocess (avoids needing a display in main)
# ---------------------------------------------------------------------------

_PREVIEW_SCRIPT = """\
import warnings, ovito
from ovito.vis import OpenGLRenderer
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ovito.scene.load({scene!r})
    p = ovito.scene.pipelines[0]
    p.source.load({dump!r})
    rs = ovito.scene.render_settings
    vp = ovito.scene.viewports.active_vp

zoom    = {zoom}
z_shift = {z_shift}
vp.fov = vp.fov / zoom
pos = list(vp.camera_pos)
pos[2] += z_shift
vp.camera_pos = tuple(pos)

vp.render_image(
    size=tuple(rs.size),
    background=tuple(float(x) for x in rs.background_color),
    renderer=OpenGLRenderer(),
    frame={frame},
    filename={outfile!r},
)
"""


def _hstack_images(left: Path, right: Path, out: Path) -> None:
    """Combine two images horizontally (side by side) using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(left), "-i", str(right),
         "-filter_complex", "hstack", str(out)],
        check=True, capture_output=True,
    )


def _xvfb_preview_frame(scene_path, dump_path, frame_idx, out_path,
                         zoom=1.0, z_shift=0.0):
    """Render a single preview frame via xvfb + OpenGL in a subprocess."""
    script = _PREVIEW_SCRIPT.format(
        scene=str(scene_path),
        dump=str(dump_path),
        frame=frame_idx,
        outfile=str(out_path),
        zoom=zoom,
        z_shift=z_shift,
    )
    result = _run_with_xvfb([sys.executable, "-W", "ignore", "-c", script])
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Preview render failed (exit {result.returncode})")


# ---------------------------------------------------------------------------
# Interactive camera adjustment
# ---------------------------------------------------------------------------

def interactive_adjust(scene_path, dump_path, preview_dir: Path,
                       f_first: int = 0, f_last: int = 0,
                       zoom_start_default: float = 1.0, z_start_default: float = 0.0,
                       zoom_end_default: float = 1.0, z_end_default: float = 0.0):
    """Two-pass interactive camera adjustment.

    Pass 1 adjusts the camera for the first frame; pass 2 adjusts independently
    for the last frame.  Workers linearly interpolate between the two settings
    across all frames, producing a smooth pan/zoom.

    Returns (zoom_start, z_start, zoom_end, z_end).
    """
    import warnings
    preview_dir.mkdir(parents=True, exist_ok=True)

    first_file = preview_dir / "preview_first.png"
    last_file  = preview_dir / "preview_last.png"
    comp_file  = preview_dir / "preview_comparison.png"

    # Load scene once just to show camera info (no rendering, no display needed).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, vp0 = _load_scene(scene_path, dump_path)

    print("\n── Interactive camera adjustment ──────────────────────────")
    print(f"  Camera pos : {tuple(round(x,2) for x in vp0.camera_pos)}")
    print(f"  Camera dir : {tuple(round(x,3) for x in vp0.camera_dir)}")
    print(f"  FOV        : {round(vp0.fov, 4)} rad  "
          f"({round(vp0.fov * 180 / 3.14159, 1)}°)")
    print("  Adjustments are relative to the scene camera above.")
    print("  Comparison image: LEFT = first frame, RIGHT = last frame.\n")

    def _prompt_and_update(zoom, z_shift):
        """Prompt for new zoom/z_shift; return (new_zoom, new_z, done)."""
        print(f"  Current : zoom={zoom:.3f}  z_shift={z_shift:.2f} Å")
        print("  Enter new values (blank = keep, 'ok' or empty = accept):")
        try:
            raw_zoom = input("    Zoom factor [>1 in, <1 out]: ").strip()
            raw_z    = input("    Z shift (Å) [+up, -down]  : ").strip()
        except EOFError:
            return zoom, z_shift, True
        if raw_zoom.lower() == "ok" or raw_z.lower() == "ok":
            return zoom, z_shift, True
        if raw_zoom == "" and raw_z == "":
            try:
                ans = input("  Accept? [y/n]: ").strip().lower()
            except EOFError:
                return zoom, z_shift, True
            return zoom, z_shift, ans in ("y", "yes", "")
        try:
            if raw_zoom:
                zoom = float(raw_zoom)
            if raw_z:
                z_shift = float(raw_z)
        except ValueError:
            print("  Invalid value — try again.")
        return zoom, z_shift, False

    # ── Pass 1: adjust START camera ───────────────────────────────────────
    print("  [STEP 1/2] Adjust START camera (both frames rendered with same settings)")
    zoom_start, z_start = zoom_start_default, z_start_default
    while True:
        print(f"  Rendering frames {f_first} and {f_last}...", flush=True)
        _xvfb_preview_frame(scene_path, dump_path, f_first, first_file,
                             zoom=zoom_start, z_shift=z_start)
        _xvfb_preview_frame(scene_path, dump_path, f_last,  last_file,
                             zoom=zoom_start, z_shift=z_start)
        _hstack_images(first_file, last_file, comp_file)
        print(f"  Comparison : {comp_file}")
        zoom_start, z_start, done = _prompt_and_update(zoom_start, z_start)
        if done:
            break

    # ── Pass 2: adjust END camera (first frame locked) ────────────────────
    print("\n  [STEP 2/2] Adjust END camera (left=first locked, right=last adjusting)")
    print("             Camera will interpolate smoothly between the two.")
    # Re-render first frame once with locked settings so it stays current in comp.
    _xvfb_preview_frame(scene_path, dump_path, f_first, first_file,
                         zoom=zoom_start, z_shift=z_start)
    zoom_end, z_end = zoom_end_default, z_end_default
    while True:
        print(f"  Rendering frame {f_last}...", flush=True)
        _xvfb_preview_frame(scene_path, dump_path, f_last, last_file,
                             zoom=zoom_end, z_shift=z_end)
        _hstack_images(first_file, last_file, comp_file)
        print(f"  Comparison : {comp_file}")
        zoom_end, z_end, done = _prompt_and_update(zoom_end, z_end)
        if done:
            break

    print(f"\n  Start : zoom={zoom_start:.3f}  z={z_start:.2f} Å")
    print(f"  End   : zoom={zoom_end:.3f}  z={z_end:.2f} Å")
    if zoom_start == zoom_end and z_start == z_end:
        print("  (identical settings — no interpolation)")
    else:
        print("  (camera will pan/zoom smoothly across the video)")
    print("──────────────────────────────────────────────────────────\n")
    return zoom_start, z_start, zoom_end, z_end


# ---------------------------------------------------------------------------
# Per-worker render script (run in a subprocess under _run_with_xvfb)
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = """\
import ovito
from ovito.vis import OpenGLRenderer

ovito.scene.load({scene!r})
p = ovito.scene.pipelines[0]
p.source.load({dump!r})
rs = ovito.scene.render_settings
vp = ovito.scene.viewports.active_vp

zoom_start = {zoom_start}
zoom_end   = {zoom_end}
z_start    = {z_start}
z_end      = {z_end}
f_first    = {f_first}
f_last     = {f_last}
span       = max(f_last - f_first, 1)

# Save base camera state; restore before each frame so lerp is absolute.
base_fov = vp.fov
base_pos = list(vp.camera_pos)

frames = {frames!r}
outdir = {outdir!r}
size   = {size!r}
bg     = {bg!r}

for i, f in enumerate(frames):
    t       = (f - f_first) / span
    zoom    = zoom_start + t * (zoom_end - zoom_start)
    z_shift = z_start   + t * (z_end   - z_start)
    vp.fov = base_fov / zoom
    pos = list(base_pos)
    pos[2] += z_shift
    vp.camera_pos = tuple(pos)
    vp.render_image(
        size=size,
        background=bg,
        renderer=OpenGLRenderer(),
        frame=f,
        filename=f'{{outdir}}/frame_{{f:06d}}.png',
    )
    if i % 20 == 0:
        print(f'worker {{i+1}}/{{len(frames)}}', flush=True)
"""


def _xvfb_render_chunk(scene_path, dump_path, frames, outdir, size, bg,
                        zoom_start=1.0, zoom_end=1.0, z_start=0.0, z_end=0.0,
                        f_first=0, f_last=0):
    """Render frames on a private Xvfb display (see _run_with_xvfb)."""
    script = _WORKER_SCRIPT.format(
        scene=str(scene_path),
        dump=str(dump_path),
        frames=frames,
        outdir=str(outdir),
        size=size,
        bg=bg,
        zoom_start=zoom_start,
        zoom_end=zoom_end,
        z_start=z_start,
        z_end=z_end,
        f_first=f_first,
        f_last=f_last,
    )
    result = _run_with_xvfb([sys.executable, "-W", "ignore", "-c", script])
    if result.returncode != 0:
        print("── worker stdout ──", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print("── worker stderr ──", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Worker failed (exit {result.returncode})")
    return result.stdout


# ---------------------------------------------------------------------------
# Local parallel render
# ---------------------------------------------------------------------------

def render_local(dump_path, scene_path, output_path, fps=15, cores=4,
                 frames_to_render=None,
                 zoom_start=1.0, zoom_end=1.0, z_start=0.0, z_end=0.0,
                 size=None, verbose=True):
    """Render dump_path to output_path using N parallel xvfb workers.

    frames_to_render: explicit list of dump frame indices to render.
                      If None, renders all frames.
    size: (width, height) override in pixels. None = use the .ovito file value.
    Camera is linearly interpolated from (zoom_start, z_start) at the first
    frame to (zoom_end, z_end) at the last frame.
    """
    import multiprocessing as mp, time, warnings

    patched = Path(tempfile.mktemp(suffix=".ovito"))
    patch_ovito_scene(scene_path, patched)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import ovito
        ovito.scene.load(str(patched))
        p = ovito.scene.pipelines[0]
        p.source.load(str(dump_path))
        rs   = ovito.scene.render_settings
        size = size if size is not None else tuple(rs.size)
        bg   = tuple(float(x) for x in rs.background_color)
        if frames_to_render is None:
            frames_to_render = list(range(p.source.num_frames))

    f_first = frames_to_render[0]
    f_last  = frames_to_render[-1]

    if verbose:
        print(f"Dump:    {dump_path}")
        print(f"Frames:  {len(frames_to_render)} "
              f"(indices {f_first}–{f_last})")
        print(f"Size:    {size[0]}×{size[1]}, {fps} fps")
        print(f"Workers: {cores}")
        if zoom_start == zoom_end and z_start == z_end:
            print(f"Camera:  zoom={zoom_start:.3f}  z_shift={z_start:.2f} Å (fixed)")
        else:
            print(f"Camera:  zoom {zoom_start:.3f}→{zoom_end:.3f}  "
                  f"z {z_start:.2f}→{z_end:.2f} Å (interpolated)")

    chunks = [frames_to_render[i::cores] for i in range(cores)]

    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()

        # Each worker starts its own Xvfb (_run_with_xvfb picks a free display).
        args_list = [
            (patched, dump_path, chunk, str(tmpdir), size, bg,
             zoom_start, zoom_end, z_start, z_end, f_first, f_last)
            for chunk in chunks
        ]
        with mp.Pool(cores) as pool:
            pool.starmap(_xvfb_render_chunk, args_list)

        if verbose:
            print(f"Frames rendered in {time.time()-t0:.1f}s")

        _stitch(frames_to_render, tmpdir, output_path, fps, verbose)

    patched.unlink(missing_ok=True)
    if verbose:
        print(f"Video: {output_path}")


# ---------------------------------------------------------------------------
# SLURM batch render
# ---------------------------------------------------------------------------

_SLURM_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={logfile}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --time={wall}
#SBATCH --account={account}
#SBATCH --mem={mem}G

export LD_LIBRARY_PATH=/usr/licensed/anaconda3/2024.10/lib:$LD_LIBRARY_PATH

python -W ignore {render_script}
"""

_SLURM_RENDER_SCRIPT = """\
import os, sys, tempfile, multiprocessing as mp
from pathlib import Path

sys.path.insert(0, {pkg_root!r})
from diamond_etch_md.render import patch_ovito_scene, _xvfb_render_chunk, _stitch

dump_path     = Path({dump!r})
scene_path    = Path({scene!r})
output        = Path({output!r})
fps           = {fps}
cores         = {cores}
frames        = {frames!r}
zoom_start    = {zoom_start}
zoom_end      = {zoom_end}
z_start       = {z_start}
z_end         = {z_end}
size_override = {size_override!r}
f_first       = frames[0]
f_last        = frames[-1]

patched = Path(tempfile.mktemp(suffix='.ovito'))
patch_ovito_scene(scene_path, patched)

import warnings, ovito
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ovito.scene.load(str(patched))
    p = ovito.scene.pipelines[0]
    p.source.load(str(dump_path))
    rs   = ovito.scene.render_settings
    size = size_override if size_override is not None else tuple(rs.size)
    bg   = tuple(float(x) for x in rs.background_color)

print(f'Rendering {{len(frames)}} frames ({{cores}} workers)', flush=True)
chunks = [frames[i::cores] for i in range(cores)]

import time; t0 = time.time()
with tempfile.TemporaryDirectory() as tmpdir:
    with mp.Pool(cores) as pool:
        pool.map(
            lambda c: _xvfb_render_chunk(
                patched, dump_path, c, tmpdir, size, bg,
                zoom_start=zoom_start, zoom_end=zoom_end,
                z_start=z_start, z_end=z_end,
                f_first=f_first, f_last=f_last,
            ),
            chunks,
        )
    print(f'Frames done in {{time.time()-t0:.1f}}s', flush=True)
    _stitch(frames, tmpdir, output, fps, verbose=True)

patched.unlink(missing_ok=True)
print(f'Done: {{output}}')
"""


def render_slurm(dump_path, scene_path, output_path, fps=15, cores=16,
                 frames_to_render=None,
                 zoom_start=1.0, zoom_end=1.0, z_start=0.0, z_end=0.0,
                 size=None, account="dgraves", wall_hours=1, mem_gb=32, verbose=True):
    """Submit a SLURM job to render the video on a CPU cluster node.

    size: (width, height) override in pixels. None = use the .ovito file value.
    """
    pkg_root = str(Path(__file__).parent.parent.resolve())
    job_name = f"ovito_{Path(dump_path).stem[:30]}"
    log_file = Path(output_path).with_suffix(".slurm.log")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     prefix="ovito_render_", delete=False) as f:
        f.write(_SLURM_RENDER_SCRIPT.format(
            pkg_root=pkg_root,
            dump=str(dump_path),
            scene=str(scene_path),
            output=str(output_path),
            fps=fps, cores=cores,
            frames=frames_to_render,
            zoom_start=zoom_start, zoom_end=zoom_end,
            z_start=z_start, z_end=z_end,
            size_override=size,
        ))
        render_script = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                     prefix="ovito_slurm_", delete=False) as f:
        f.write(_SLURM_SCRIPT.format(
            jobname=job_name,
            logfile=str(log_file),
            cores=cores,
            wall=f"{wall_hours:02d}:00:00",
            account=account,
            mem=mem_gb,
            render_script=render_script,
        ))
        slurm_script = f.name

    result = subprocess.run(["sbatch", slurm_script], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("sbatch failed")

    job_id = result.stdout.strip().split()[-1]
    if verbose:
        print(f"Submitted job {job_id}")
        print(f"Output: {output_path}")
        print(f"Log:    {log_file}")
    return job_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stitch(frames_to_render, frame_dir, output_path, fps, verbose=True):
    """Stitch PNG frames (named by dump frame index) into an mp4."""
    frame_dir = Path(frame_dir)
    # Write a file list in order so ffmpeg doesn't need sequential numbering
    list_file = frame_dir / "frames.txt"
    with open(list_file, "w") as f:
        for idx in frames_to_render:
            png = frame_dir / f"frame_{idx:06d}.png"
            f.write(f"file '{png}'\n")
            f.write(f"duration {1/fps:.6f}\n")
    if verbose:
        print(f"Stitching {len(frames_to_render)} frames → {output_path}")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output_path),
    ], check=True, capture_output=not verbose)


# ---------------------------------------------------------------------------
# Phase-end snapshot renders (cycling)
# ---------------------------------------------------------------------------

_ANACONDA_LIB = "/usr/licensed/anaconda3/2024.10/lib"
# OVITO + the LD_LIBRARY_PATH fix are tested with this interpreter; jobs may run
# auto-plot under a different python3 (e.g. 3.10 from the LAMMPS module).
_OVITO_PYTHON = "/usr/licensed/anaconda3/2024.10/bin/python3"


def _run_render_jobs(out_dir, jobs, preset="standard", size=None, views=(0,),
                     verbose=True):
    """Render snapshot jobs [{data, png, meta, group}] in one xvfb/OVITO subprocess.

    Each group shares one height window (Å above the lowest C atom, from just
    below the lattice bottom to the group's tallest surface); the .json
    sidecars give each image's extent on that axis.  `views` are camera
    rotations (deg) about the vertical axis; view 0 goes to <png>, others to
    <png stem>_<angle>.png, all listed in the sidecar under "views".
    """
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = out_dir / "_scene.ovito"
    patch_ovito_scene(Path(__file__).parent / "ovito_configs" / f"{preset}.ovito", scene)
    job_file = out_dir / "_jobs.json"
    job_file.write_text(json.dumps({"scene": str(scene), "jobs": jobs,
                                    "size": list(size) if size else None,
                                    "views": [int(v) for v in views]}))
    pkg_root = str(Path(__file__).resolve().parents[1])
    code = (f"import sys; sys.path.insert(0, {pkg_root!r}); "
            f"from diamond_etch_md.render import _render_worker; "
            f"_render_worker({str(job_file)!r})")
    python = _OVITO_PYTHON if Path(_OVITO_PYTHON).exists() else sys.executable
    env = {**os.environ,
           "LD_LIBRARY_PATH": _ANACONDA_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    try:
        result = _run_with_xvfb([python, "-W", "ignore", "-c", code], env=env)
    finally:
        scene.unlink(missing_ok=True)
        job_file.unlink(missing_ok=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"snapshot render failed (exit {result.returncode})")
    if verbose:
        print(result.stdout.strip())
    return [Path(j["png"]) for j in jobs]


def render_phase_ends(sim_dir, spec, ml, preset="standard", views=(0,), verbose=True):
    """Render every completed phase-end snapshot; one height window per cycle.
    Writes renders/phase_ends/<stem>.png + .json.  Returns the PNG paths."""
    from .analysis.density import phase_end_snapshots, render_stem, RENDER_DIR
    sim_dir = Path(sim_dir).resolve()
    out_dir = sim_dir / RENDER_DIR
    jobs = [{"data": str(path), "group": c,
             "png": str(out_dir / f"{render_stem(c, pi, d)}.png"),
             "meta": str(out_dir / f"{render_stem(c, pi, d)}.json")}
            for c, pi, d, path in phase_end_snapshots(sim_dir, spec, ml)]
    return _run_render_jobs(out_dir, jobs, preset, views=views, verbose=verbose) if jobs else []


def render_ml_frames(sim_dir, ml, preset="standard", size=(400, 800), views=(0,),
                     verbose=True):
    """Render the movie frames (analysis.density.movie_impacts: one per ML, or
    finer for short runs), all on one height window.
    Writes renders/ml_frames/<impact>.png + .json.  Returns the PNG paths."""
    from .analysis.density import _snap_path, movie_impacts, _frame_stem, ML_FRAME_DIR
    sim_dir = Path(sim_dir).resolve()
    out_dir = sim_dir / ML_FRAME_DIR
    if out_dir.is_dir():                        # window/cadence may change between runs
        for f in list(out_dir.glob("*.png")) + list(out_dir.glob("*.json")):
            f.unlink()
    jobs = [{"data": str(_snap_path(sim_dir, imp)), "group": 0,
             "png": str(out_dir / f"{_frame_stem(imp)}.png"),
             "meta": str(out_dir / f"{_frame_stem(imp)}.json")}
            for imp in movie_impacts(sim_dir, ml)]
    return (_run_render_jobs(out_dir, jobs, preset, size=size, views=views, verbose=verbose)
            if jobs else [])


def _render_worker(job_file):
    """Runs under xvfb with OVITO importable; see _run_render_jobs."""
    import json
    import warnings
    import numpy as np
    from PIL import Image
    warnings.filterwarnings("ignore")
    import ovito
    from ovito.vis import OpenGLRenderer

    cfg = json.loads(Path(job_file).read_text())
    ovito.scene.load(cfg["scene"])
    p = ovito.scene.pipelines[0]
    vp = ovito.scene.viewports.active_vp
    rs = ovito.scene.render_settings
    W, H = (int(x) for x in (cfg["size"] or rs.size))
    bg = tuple(float(x) for x in rs.background_color)
    pos0 = np.array(vp.camera_pos, dtype=float)
    dir0 = np.array(vp.camera_dir, dtype=float)
    orbit = np.array(vp.orbit_center, dtype=float)
    base_pos = list(pos0)

    def set_view(angle):
        """Rotate the preset camera by `angle` deg about the vertical axis
        through the orbit center (same height, same zoom)."""
        nonlocal base_pos
        c, s_ = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        vp.camera_dir = tuple(rot @ dir0)
        base_pos = list(orbit + rot @ (pos0 - orbit))

    views = cfg.get("views") or [0]

    def view_png(job, angle):
        return job["png"] if angle == 0 else str(
            Path(job["png"]).with_name(f"{Path(job['png']).stem}_{angle}.png"))

    p.source.load(cfg["jobs"][0]["data"], atom_style="charge")
    radius = {t.id: float(t.radius or 0.5) for t in p.compute().particles.particle_types.types}

    info = []
    for job in cfg["jobs"]:
        p.source.load(job["data"], atom_style="charge")
        src = p.source.data.particles          # raw positions: no modifiers needed
        z = np.asarray(src.positions[:, 2])
        types = np.asarray(src["Particle Type"].array)
        z_bot = float(z[types == 1].min())
        top_i = int(np.argmax(z))
        r = np.array([radius.get(int(t), 0.5) for t in types])
        info.append(dict(z_bot=z_bot, h_top=float(z[top_i]) - z_bot,
                         r_top=radius.get(int(types[top_i]), 0.5),
                         z_low=float((z - r).min())))

    def render(cam_z, fn):
        pos = list(base_pos)
        pos[2] = cam_z
        vp.camera_pos = tuple(pos)
        vp.render_image(size=(W, H), background=bg, renderer=OpenGLRenderer(),
                        filename=str(fn))

    def bottom_edge(fn):
        """Pixel row just below the lowest drawn pixel (None if frame is empty)."""
        a = np.asarray(Image.open(fn).convert("RGB")).astype(int)
        rows = np.where((a < 235).any(axis=2).any(axis=1))[0]
        return int(rows.max()) + 1 if len(rows) else None

    def calibrate(job, i):
        """Measure px/Å and the view-center offset (view center != camera z,
        and the offset does not scale with zoom) from the lattice bottom of
        one snapshot: find a camera height with the bottom edge inside the
        frame, then render again slightly higher.  The bottom layer spans the
        whole width, so it cannot be cropped out the way a lone top atom can."""
        p.source.load(job["data"], atom_style="charge")
        cell_vis = p.source.data.cell.vis
        cell_vis.enabled = False
        z_low = i["z_low"]
        tmp = Path(job["png"]).with_name("_cal.png")
        vis_guess = vp.fov * H / W
        cam1, r1 = z_low, None
        for _ in range(40):
            render(cam1, tmp)
            r1 = bottom_edge(tmp)
            if r1 is None:                  # everything above the frame
                cam1 += vis_guess / 3
            elif r1 >= H:                   # lattice bottom below the frame
                cam1 -= vis_guess / 3
            elif r1 > 0.7 * H:              # too close to the bottom for the 2nd shot
                cam1 -= vis_guess / 6
            else:
                break
        dz = 0.2 * vis_guess
        render(cam1 + dz, tmp)
        r2 = bottom_edge(tmp)
        tmp.unlink(missing_ok=True)
        cell_vis.enabled = True
        if r1 is None or r2 is None or not 0 < r1 < r2 < H:
            raise RuntimeError(f"camera calibration failed (lattice-bottom rows {r1}, {r2})")
        scale = (r2 - r1) / dz
        return scale, z_low - cam1 - (H / 2 - r1) / scale

    # Visible height is fov * H / W for this portrait ortho preset.
    summary = []
    for g in sorted({j["group"] for j in cfg["jobs"]}):
        group = [(j, i) for j, i in zip(cfg["jobs"], info) if j["group"] == g]
        h_lo = -1.0
        h_hi = max(i["h_top"] + i["r_top"] for _, i in group) + 3.0
        h_center = (h_lo + h_hi) / 2
        vp.fov = (h_hi - h_lo) * W / H
        metas = {id(j): {"z_bottom": i["z_bot"], "snapshot": j["data"], "views": {}}
                 for j, i in group}
        # The camera is horizontal, so spinning it about the vertical axis
        # leaves the height->row mapping unchanged: calibrate once and share
        # it so every view sits on the same height axis.
        set_view(views[0])
        scale, offset = calibrate(*max(group, key=lambda ji: ji[1]["h_top"]))
        vis_h = H / scale
        for angle in views:
            set_view(angle)
            for job, i in group:
                p.source.load(job["data"], atom_style="charge")
                render(i["z_bot"] + h_center - offset, view_png(job, angle))
                metas[id(job)]["views"][str(angle)] = {
                    "png": Path(view_png(job, angle)).name,
                    "h_lo": h_center - vis_h / 2, "h_hi": h_center + vis_h / 2,
                    "width_A": W / scale}
            summary.append(f"group {g} view {angle}°: {h_lo:.0f}-{h_hi:.0f} Å, "
                           f"{scale:.1f} px/Å")
        for job, _ in group:
            m = metas[id(job)]
            m.update({k: m["views"][str(views[0])][k] for k in ("h_lo", "h_hi", "width_A")})
            Path(job["meta"]).write_text(json.dumps(m))
    print(f"Rendered {len(cfg['jobs'])} snapshots ({'; '.join(summary)})")
