"""
builder.py — assemble a complete simulation directory from a SimSpec.

`make_sim` writes config.lmp, head.lmp, make_surf.lmp, and the SLURM submit
script into `outdir`, then creates symlinks to shared LAMMPS scripts from the
package's lammps/templates/ directory and to data files from the `dfiles/` tree.
"""

import shutil
from pathlib import Path

from .orientations import ORIENT
from .spec import SimSpec
from .lammps.config import get_config_lmp
from .lammps.head import get_head_lmp
from .lammps.submit import get_submit_script


def get_make_surf_source(spec: SimSpec, dfiles_root: Path) -> Path:
    """Return the absolute Path to the make_surf.lmp source for this spec.

    Parameters
    ----------
    spec:
        The simulation specification; orientation, reconstruction, and
        termination are used to select the correct source file.
    dfiles_root:
        Absolute path to the dfiles/ directory that anchors relative paths
        stored in ORIENT[...]["make_surf"].
    """
    ms = ORIENT[spec.orientation]["make_surf"]
    if spec.orientation == "113" and spec.termination == "O" and spec.reconstruction == "bare":
        # O-terminated 113: prefer the O_terminated make_surf if available
        rel = ms.get("O", ms["bare"])
    else:
        rel = ms.get(spec.reconstruction, ms[next(iter(ms))])
    if rel.startswith("package:"):
        return Path(__file__).parent / rel[len("package:"):]
    return dfiles_root / rel


def make_sim(spec: SimSpec, outdir: Path, dfiles_root: Path) -> None:
    """Create a complete simulation directory for the given spec.

    Parameters
    ----------
    spec:
        Fully-populated and validated SimSpec.
    outdir:
        Destination directory (created if it does not exist).
    dfiles_root:
        Absolute path to the dfiles/ directory used for source file resolution
        and symlink targets.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # make_surf.lmp — copy from orientation/reconstruction source
    src = get_make_surf_source(spec, dfiles_root)
    shutil.copy(src, outdir / "make_surf.lmp")

    # head.lmp — generated with orientation-specific lattice and bottom
    (outdir / "head.lmp").write_text(get_head_lmp(spec))

    # config.lmp — generated from spec
    (outdir / "config.lmp").write_text(get_config_lmp(spec))

    # submit script
    submit = outdir / "submit"
    submit.write_text(get_submit_script(spec))
    submit.chmod(0o755)

    # symlink shared LAMMPS scripts from package templates
    templates = Path(__file__).parent / "lammps" / "templates"
    for fname in ("sweep.lmp", "thermalize.lmp", "addfix.lmp"):
        dst = outdir / fname
        if not dst.exists():
            dst.symlink_to(templates / fname)

    # symlink shared data files from package templates
    for fname in ("ffield.reax", "lat_a.txt", "lmp_env.sh"):
        dst = outdir / fname
        if not dst.exists():
            dst.symlink_to(templates / fname)

    print(f"Simulation created at: {outdir}")
    print(f"  surface:    {spec.orientation}  reconstruction={spec.reconstruction}  termination={spec.termination}")
    print(f"  bombardment: {spec.species} at {spec.energy} eV, angle={spec.angle}\u00b0, T={spec.temperature} K")
    print(f"  box:        {spec.box_x}\u00d7{spec.box_y}\u00d7{spec.box_depth} lattice units,  ML={spec.ml}")
    print(f"  fluence:    {spec.fluence} ML  ({spec.fluence * spec.ml} total impacts)")
    print(f"  wall time:  {spec.wall_hours} h")
    print(f"\nTo submit: sbatch {outdir}/submit")
    print(f"\nNote: verify ML={spec.ml} matches actual surface atom count from data_files/0.data")
