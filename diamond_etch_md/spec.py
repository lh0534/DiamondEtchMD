"""
spec.py — SimSpec dataclass, ML formula, and validation.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Optional

from .orientations import ORIENT
from .species import SPECIES


@dataclass
class CyclePhase:
    """One phase of a cycling simulation.

    Each phase defines an ion species, its energy, how many monolayers to run
    per cycle repetition, the number of O• radicals deposited before each ion
    impact (flux_ratio), and the kinetic energy of those radicals.
    """
    species:        str
    energy:         float          # eV  (total dimer energy for O2)
    fluence_ml:     int            # ML of this ion species per cycle
    flux_ratio:     int   = 0      # O• radicals deposited per ion impact (0 = none)
    radical_energy: float = 0.2    # eV for each O• radical


@dataclass
class SimSpec:
    orientation:         str   = "100"
    surface:             str   = "1x1"       # surface state (reconstruction + termination)
                                              # 100: 1x1, 2x1, 2x1_O, O_ether
                                              # 110: "", O
                                              # 111: 1x1, 2x1_single, 2x1_pandey,
                                              #      1x1_O, 2x1_single_O, 2x1_pandey_O
                                              # 113: "", O
    temperature:         float = 300.0    # K
    species:             str   = "O"      # single-species mode only
    energy:              float = 0.5      # eV  (single-species mode only)
    angle:               float = 0.0      # degrees from surface normal
    fluence:             int   = 50       # monolayers (single-species mode only)
    ml:                  int   = 0        # atoms per monolayer; 0 = compute from ml_factor*x*y
    box_x:               int   = 9        # lattice units
    box_y:               int   = 9        # lattice units
    box_depth:           int   = 3        # lat_top in lattice units
    impact_time:         float = 2000.0   # fs — ion impact window
    thermalization_time: float = 500.0    # fs — post-impact thermalisation
    inter_neutral_time:  float = 1000.0   # fs — O• radical impact window (cycling only)
    wall_hours:          int   = 24
    name:                str   = ""
    account:             str   = "dgraves"
    email:               str   = ""    # empty = no mail directives
    lammps_module:       str   = "lammps/kokkos/gpu_della9_2022"
    plot_interval_hours: int   = 12     # hours between auto-plot runs (0 = disabled)
    # ── Cycling mode ──────────────────────────────────────────────────────────
    phases:              Optional[List[CyclePhase]] = None  # None = single-species mode
    cycles:              int   = 1      # how many times the phase list repeats


def compute_ml(orientation: str, box_x: int, box_y: int) -> int:
    """Return the atoms-per-monolayer count for a given orientation and box size.

    Formula: ML = ml_factor * box_x * box_y
    """
    return ORIENT[orientation]["ml_factor"] * box_x * box_y


def validate(spec: SimSpec) -> None:
    """Validate a SimSpec; exit with an informative message on any error."""
    if spec.orientation not in ORIENT:
        sys.exit(f"Unknown orientation '{spec.orientation}'. Choose from: {list(ORIENT)}")

    orient_cfg = ORIENT[spec.orientation]
    valid_surfaces = list(orient_cfg["surfaces"])
    if spec.surface not in orient_cfg["surfaces"]:
        sys.exit(
            f"Surface '{spec.surface}' not valid for {spec.orientation}. "
            f"Choose from: {valid_surfaces}"
        )

    if spec.ml <= 0:
        sys.exit("ML (atoms per monolayer) must be > 0.")

    if spec.orientation == "100" and spec.surface in ("2x1", "2x1_O"):
        if spec.box_x % 2 != 0 or spec.box_y % 2 != 0:
            sys.exit(
                f"C(100) 2×1 reconstruction requires even box dimensions; "
                f"got box_x={spec.box_x}, box_y={spec.box_y}."
            )

    if spec.phases is not None:
        # ── Cycling-mode validation ────────────────────────────────────────
        if len(spec.phases) < 2:
            sys.exit("Cycling requires at least 2 phases.")
        if spec.cycles <= 0:
            sys.exit("cycles must be > 0.")
        for i, p in enumerate(spec.phases):
            if p.species not in SPECIES:
                sys.exit(
                    f"Phase {i} ({p.species!r}): unknown species. "
                    f"Choose from: {list(SPECIES)}"
                )
            if p.fluence_ml <= 0:
                sys.exit(f"Phase {i}: fluence_ml must be > 0, got {p.fluence_ml}.")
            if p.flux_ratio < 0:
                sys.exit(f"Phase {i}: flux_ratio must be >= 0, got {p.flux_ratio}.")
    else:
        # ── Single-species-mode validation ────────────────────────────────
        if spec.species not in SPECIES:
            sys.exit(f"Unknown species '{spec.species}'. Choose from: {list(SPECIES)}")
