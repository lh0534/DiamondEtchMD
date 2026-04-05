"""
builder.py — assemble a complete simulation directory from a SimSpec.

`make_sim` writes config.lmp, head.lmp, make_surf.lmp, and the SLURM submit
script into `outdir`, then creates symlinks to shared LAMMPS scripts bundled
with the package (ffield.reax, lat_a.txt, sweep.lmp, thermalize.lmp, addfix.lmp).
All surface templates are package-bundled; no external dfiles directory is needed.
"""

import shutil
from pathlib import Path

from .orientations import ORIENT
from .species import SPECIES
from .spec import SimSpec
from .lammps.config import get_config_lmp
from .lammps.head import get_head_lmp
from .lammps.submit import get_submit_script

_TEMPLATES = Path(__file__).parent / "lammps" / "templates"


def get_make_surf_source(spec: SimSpec) -> Path:
    """Return the absolute Path to the make_surf.lmp template for this spec."""
    rel = ORIENT[spec.orientation]["make_surf"][spec.reconstruction]
    assert rel.startswith("package:"), f"unexpected make_surf path: {rel}"
    return Path(__file__).parent / rel[len("package:"):]


def make_sim(spec: SimSpec, outdir: Path) -> None:
    """Create a complete simulation directory for the given spec.

    Parameters
    ----------
    spec:
        Fully-populated and validated SimSpec.
    outdir:
        Destination directory (created if it does not exist).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # make_surf.lmp — copied from the package template for this orientation/reconstruction
    shutil.copy(get_make_surf_source(spec), outdir / "make_surf.lmp")

    # head.lmp — generated with orientation-specific lattice and bottom
    (outdir / "head.lmp").write_text(get_head_lmp(spec))

    # config.lmp — generated from spec
    (outdir / "config.lmp").write_text(get_config_lmp(spec))

    # submit script
    submit = outdir / "submit"
    submit.write_text(get_submit_script(spec))
    submit.chmod(0o755)

    # symlink shared LAMMPS scripts and data files from package templates
    for fname in ("sweep.lmp", "thermalize.lmp", "addfix.lmp",
                  "ffield.reax", "lat_a.txt", "lmp_env.sh"):
        dst = outdir / fname
        if not dst.exists():
            dst.symlink_to(_TEMPLATES / fname)

    # symlink molecule file if the species requires one (e.g. O2.molecule)
    species_cfg = SPECIES[spec.species]
    if species_cfg["molecule_file"]:
        mol_dst = outdir / species_cfg["molecule_file"]
        if not mol_dst.exists():
            mol_dst.symlink_to(_TEMPLATES / species_cfg["molecule_file"])

    print(f"Simulation created at: {outdir}")
    print(f"  surface:     {spec.orientation}  reconstruction={spec.reconstruction}  termination={spec.termination}")
    print(f"  bombardment: {spec.species} at {spec.energy} eV, angle={spec.angle}\u00b0, T={spec.temperature} K")
    print(f"  box:         {spec.box_x}\u00d7{spec.box_y}\u00d7{spec.box_depth} lattice units,  ML={spec.ml}")
    print(f"  fluence:     {spec.fluence} ML  ({spec.fluence * spec.ml} total impacts)")
    print(f"  wall time:   {spec.wall_hours} h  account={spec.account}")
    print(f"\nTo submit: sbatch {outdir}/submit")
    print(f"\nNote: verify ML={spec.ml} matches actual surface atom count from data_files/0.data")
