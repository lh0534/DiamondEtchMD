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
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .orientations import ORIENT
from .species import SPECIES
from .spec import SimSpec, etch_mode
from .lammps.config import get_config_lmp, get_config_lmp_cycle_etch, get_config_lmp_multi_ion
from .lammps.head import get_head_lmp, get_head_lmp_multi_ion
from .lammps.head_cycling import get_head_lmp_cycle_etch
from .lammps.submit import get_submit_script, get_submit_script_cycle_etch

_TEMPLATES = Path(__file__).parent / "lammps" / "templates"


def _check_no_running_job(spec: SimSpec, outdir: Path) -> None:
    """Abort if a SLURM job with spec.name is running and would be overwritten.

    Overwiting head.lmp while LAMMPS is mid-run corrupts the byte offsets used
    by 'jump SELF <label>' — every subsequent label seek lands in the wrong place.
    """
    if not spec.name or not (outdir / "head.lmp").exists():
        return
    try:
        user = os.environ.get("USER") or subprocess.check_output(["whoami"], text=True).strip()
        result = subprocess.run(
            ["squeue", "-u", user, "-o", "%.100j %.2t", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return  # squeue unavailable (local workstation) — skip check
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == spec.name and parts[1] == "R":
            sys.exit(
                f"ABORT: job '{spec.name}' is currently running (SLURM state R).\n"
                f"Overwriting {outdir / 'head.lmp'} while LAMMPS is mid-run would\n"
                f"corrupt jump-SELF byte offsets in the active job.\n"
                f"Wait for the job to finish, or cancel it first:\n"
                f"  scancel --name {spec.name}"
            )


def multi_ion_dir_name(spec: SimSpec) -> str:
    """Return a canonical directory name for a multi-ion run.

    Format (with radicals, flux_ratio > 0): RIE_{ion0}_{pct0}p_{e0}eV_..._R{FR}
    Format (no radicals, flux_ratio == 0):  ION_{ion0}_{pct0}p_{e0}eV_...

    Example (50% O@50eV + 50% O2@100eV, flux_ratio=5): RIE_O_50p_50eV_O2_50p_100eV_R5
    Example (same mix, no radicals):                    ION_O_50p_50eV_O2_50p_100eV

    Fractions are normalized before computing percentages so un-normalized mixes
    produce the same name as their normalized equivalents.
    """
    if spec.ion_mix is None:
        raise ValueError("multi_ion_dir_name requires spec.ion_mix to be set")
    total = sum(c.fraction for c in spec.ion_mix)
    parts = []
    for comp in spec.ion_mix:
        pct = round(comp.fraction / total * 100)
        parts.append(f"{comp.species}_{pct}p_{int(comp.energy)}eV")
    body = "_".join(parts)
    if spec.flux_ratio > 0:
        return f"RIE_{body}_R{spec.flux_ratio}"
    return f"ION_{body}"


def get_make_surf_source(spec: SimSpec) -> Path:
    """Return the absolute Path to the make_surf.lmp template for this spec."""
    rel = ORIENT[spec.orientation]["surfaces"][spec.surface]["template"]
    assert rel.startswith("package:"), f"unexpected make_surf path: {rel}"
    return Path(__file__).parent / rel[len("package:"):]


def make_sim(spec: SimSpec, outdir: Path) -> None:
    """Create a complete simulation directory for the given spec."""
    _check_no_running_job(spec, outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # make_surf.lmp — copied from the package template for this surface
    shutil.copy(get_make_surf_source(spec), outdir / "make_surf.lmp")

    # symlink shared LAMMPS scripts and data files from package templates
    for fname in ("sweep.lmp", "thermalize.lmp", "addfix.lmp",
                  "notify_channeled.lmp",
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
    elif spec.ion_mix is not None:
        # ── Multi-ion-etch or multi-RIE-etch mode ─────────────────────────────
        (outdir / "head.lmp").write_text(get_head_lmp_multi_ion(spec))
        (outdir / "config.lmp").write_text(get_config_lmp_multi_ion(spec))
        submit = outdir / "submit"
        submit.write_text(get_submit_script(spec))
        submit.chmod(0o755)

        # Symlink molecule files for all molecule species in the mix
        for comp in spec.ion_mix:
            mol = SPECIES[comp.species]["molecule_file"]
            if mol:
                mol_dst = outdir / mol
                if not mol_dst.exists():
                    mol_dst.symlink_to(_TEMPLATES / mol)

        total = sum(c.fraction for c in spec.ion_mix)
        mix_str = " + ".join(
            f"{c.species}@{c.energy}eV({c.fraction/total:.0%})"
            for c in spec.ion_mix
        )
        rie_str = (
            f"  flux_ratio:  {spec.flux_ratio} O• radicals/impact  "
            f"(radical_energy={spec.radical_energy} eV)\n"
        ) if mode == "multi-rie-etch" else ""

        print(f"Simulation created at: {outdir}  [{mode}]")
        print(f"  surface:     {spec.orientation}  {surface_label}")
        print(f"  ion mix:     {mix_str}  angle={spec.angle}°  T={spec.temperature} K")
        if rie_str:
            print(rie_str, end="")
        print(f"  box:         {spec.box_x}×{spec.box_y}×{spec.box_depth} lattice units,  ML={spec.ml}")
        print(f"  fluence:     {spec.fluence} ML  ({spec.fluence * spec.ml} total impacts)")
        print(f"  wall time:   {spec.wall_hours} h  account={spec.account}")
    else:
        # ── Ion-etch or RIE-etch mode ──────────────────────────────────────────
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
