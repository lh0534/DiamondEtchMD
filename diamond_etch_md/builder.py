"""
builder.py — assemble a complete simulation directory from a SimSpec.

`make_sim` writes config.lmp, head.lmp, make_surf.lmp, and the SLURM submit
script into `outdir`, then creates symlinks to shared LAMMPS scripts bundled
with the package (ffield.reax, lat_a.txt, sweep.lmp, thermalize.lmp, addfix.lmp).
All surface templates are package-bundled; no external dfiles directory is needed.

Cycling mode (spec.phases is not None) routes to the cycling generators and
symlinks O2.molecule if any phase uses it.
"""

import shutil
from pathlib import Path

from .orientations import ORIENT
from .species import SPECIES
from .spec import SimSpec
from .lammps.config import get_config_lmp, get_config_lmp_cycling
from .lammps.head import get_head_lmp
from .lammps.head_cycling import get_head_lmp_cycling
from .lammps.submit import get_submit_script, get_submit_script_cycling

_TEMPLATES = Path(__file__).parent / "lammps" / "templates"


def get_make_surf_source(spec: SimSpec) -> Path:
    """Return the absolute Path to the make_surf.lmp template for this spec."""
    rel = ORIENT[spec.orientation]["surfaces"][spec.surface]["template"]
    assert rel.startswith("package:"), f"unexpected make_surf path: {rel}"
    return Path(__file__).parent / rel[len("package:"):]


def make_sim(spec: SimSpec, outdir: Path) -> None:
    """Create a complete simulation directory for the given spec."""
    outdir.mkdir(parents=True, exist_ok=True)

    # make_surf.lmp — copied from the package template for this surface
    shutil.copy(get_make_surf_source(spec), outdir / "make_surf.lmp")

    # symlink shared LAMMPS scripts and data files from package templates
    for fname in ("sweep.lmp", "thermalize.lmp", "addfix.lmp",
                  "ffield.reax", "lat_a.txt", "lmp_env.sh"):
        dst = outdir / fname
        if not dst.exists():
            dst.symlink_to(_TEMPLATES / fname)

    surface_label = spec.surface if spec.surface else "(unterminated)"

    if spec.phases is not None:
        # ── Cycling mode ──────────────────────────────────────────────────────
        (outdir / "head.lmp").write_text(get_head_lmp_cycling(spec))
        (outdir / "config.lmp").write_text(get_config_lmp_cycling(spec))
        submit = outdir / "submit"
        submit.write_text(get_submit_script_cycling(spec))
        submit.chmod(0o755)

        # Symlink O2.molecule if any phase uses it
        for p in spec.phases:
            mol = SPECIES[p.species]["molecule_file"]
            if mol:
                mol_dst = outdir / mol
                if not mol_dst.exists():
                    mol_dst.symlink_to(_TEMPLATES / mol)

        total_ml = spec.cycles * sum(p.fluence_ml for p in spec.phases)
        phase_str = " → ".join(
            f"{p.species}@{p.energy}eV×{p.fluence_ml}ML"
            + (f"+O•R{p.flux_ratio}" if p.flux_ratio > 0 else "")
            for p in spec.phases
        )
        print(f"Simulation created at: {outdir}")
        print(f"  surface:     {spec.orientation}  {surface_label}")
        print(f"  phases:      {phase_str}")
        print(f"  cycles:      {spec.cycles}  ({total_ml} ML total, {total_ml * spec.ml} impacts)")
        print(f"  box:         {spec.box_x}×{spec.box_y}×{spec.box_depth} lattice units,  ML={spec.ml}")
        print(f"  T={spec.temperature} K  angle={spec.angle}°")
        print(f"  wall time:   {spec.wall_hours} h  account={spec.account}")
    else:
        # ── Single-species mode ───────────────────────────────────────────────
        (outdir / "head.lmp").write_text(get_head_lmp(spec))
        (outdir / "config.lmp").write_text(get_config_lmp(spec))
        submit = outdir / "submit"
        submit.write_text(get_submit_script(spec))
        submit.chmod(0o755)

        species_cfg = SPECIES[spec.species]
        if species_cfg["molecule_file"]:
            mol_dst = outdir / species_cfg["molecule_file"]
            if not mol_dst.exists():
                mol_dst.symlink_to(_TEMPLATES / species_cfg["molecule_file"])

        print(f"Simulation created at: {outdir}")
        print(f"  surface:     {spec.orientation}  {surface_label}")
        print(f"  bombardment: {spec.species} at {spec.energy} eV, angle={spec.angle}°, T={spec.temperature} K")
        print(f"  box:         {spec.box_x}×{spec.box_y}×{spec.box_depth} lattice units,  ML={spec.ml}")
        print(f"  fluence:     {spec.fluence} ML  ({spec.fluence * spec.ml} total impacts)")
        print(f"  wall time:   {spec.wall_hours} h  account={spec.account}")

    print(f"\nTo submit: sbatch {outdir}/submit")
    print(f"\nNote: verify ML={spec.ml} matches actual surface atom count from data_files/0.data")
