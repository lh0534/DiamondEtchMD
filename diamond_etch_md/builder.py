"""
builder.py — assemble a complete simulation directory from a SimSpec.

`make_sim` writes config.lmp, head.lmp, make_surf.lmp, and the SLURM submit
script into `outdir`, then creates symlinks to shared LAMMPS scripts bundled
with the package (ffield.reax, lat_a.txt, sweep.lmp, thermalize.lmp, addfix.lmp).
All surface templates are package-bundled; no external dfiles directory is needed.

Routing by etch mode (from spec.py etch_mode()):
  ion-etch  — spec.phases is None, flux_ratio == 0
  rie-etch     — spec.phases is None, flux_ratio > 0
  cycle-etch   — spec.phases is not None
  ALE-etch     — cycle-etch with exactly 2 phases; use make_ale() factory

cycle-etch mode routes to the cycle-etch generators and symlinks O2.molecule
if any phase uses it.
"""

import dataclasses
import json
import shutil
import sys
from pathlib import Path

from .orientations import ORIENT
from .species import SPECIES
from .spec import SimSpec, etch_mode
from .lammps.config import get_config_lmp, get_config_lmp_cycle_etch
from .lammps.head import get_head_lmp
from .lammps.head_cycling import get_head_lmp_cycle_etch
from .lammps.submit import get_submit_script, get_submit_script_cycle_etch

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
                  "ffield.reax", "lat_a.txt", "lmp_env.sh", "auto-plot.py",
                  "make_impact_dump.py"):
        dst = outdir / fname
        if not dst.exists():
            dst.symlink_to(_TEMPLATES / fname)

    surface_label = spec.surface if spec.surface else "(unterminated)"

    mode = etch_mode(spec)

    if spec.phases is not None:
        # ── Cycle-etch mode ───────────────────────────────────────────────────
        (outdir / "head.lmp").write_text(get_head_lmp_cycle_etch(spec))
        (outdir / "config.lmp").write_text(get_config_lmp_cycle_etch(spec))
        submit = outdir / "submit"
        submit.write_text(get_submit_script_cycle_etch(spec))
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
        print(f"Simulation created at: {outdir}  [{mode}]")
        print(f"  surface:     {spec.orientation}  {surface_label}")
        print(f"  phases:      {phase_str}")
        print(f"  cycles:      {spec.cycles}  ({total_ml} ML total, {total_ml * spec.ml} impacts)")
        print(f"  box:         {spec.box_x}×{spec.box_y}×{spec.box_depth} lattice units,  ML={spec.ml}")
        print(f"  T={spec.temperature} K  angle={spec.angle}°")
        print(f"  wall time:   {spec.wall_hours} h  account={spec.account}")
    else:
        # ── Theory-etch or RIE-etch mode ──────────────────────────────────────
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

        rie_str = (
            f"  flux_ratio:  {spec.flux_ratio} O• radicals/impact  "
            f"(radical_energy={spec.radical_energy} eV)\n"
        ) if mode == "rie-etch" else ""

        print(f"Simulation created at: {outdir}  [{mode}]")
        print(f"  surface:     {spec.orientation}  {surface_label}")
        print(f"  bombardment: {spec.species} at {spec.energy} eV, angle={spec.angle}°, T={spec.temperature} K")
        if rie_str:
            print(rie_str, end="")
        print(f"  box:         {spec.box_x}×{spec.box_y}×{spec.box_depth} lattice units,  ML={spec.ml}")
        print(f"  fluence:     {spec.fluence} ML  ({spec.fluence * spec.ml} total impacts)")
        print(f"  wall time:   {spec.wall_hours} h  account={spec.account}")

    # Save spec as JSON for analysis tools
    spec_dict = dataclasses.asdict(spec)
    (outdir / 'spec.json').write_text(json.dumps(spec_dict, indent=2))

    print(f"\nTo submit: sbatch {outdir}/submit")
    print(f"\nNote: verify ML={spec.ml} matches actual surface atom count from impact_snaps/0.data")


def make_ale(spec: SimSpec, outdir: Path) -> None:
    """Create a simulation directory for ALE-etch (Atomic Layer Etching).

    ALE-etch is a cycle-etch with exactly 2 phases.  This function validates
    the 2-phase constraint and then delegates to make_sim().

    Args:
        spec:   A SimSpec with exactly 2 CyclePhase entries in spec.phases.
        outdir: Output directory path.

    Raises:
        SystemExit if spec.phases is None or len(spec.phases) != 2.
    """
    if spec.phases is None:
        sys.exit(
            "make_ale() requires spec.phases to be set with exactly 2 CyclePhase entries. "
            "Use make_sim() for single-species (ion-etch or RIE-etch) simulations."
        )
    if len(spec.phases) != 2:
        sys.exit(
            f"ALE-etch requires exactly 2 phases, got {len(spec.phases)}. "
            "For more phases, use make_sim() directly (cycle-etch)."
        )
    print("ALE-etch (cycle-etch with 2 phases)")
    make_sim(spec, outdir)
