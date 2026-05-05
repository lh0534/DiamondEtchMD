"""
spec.py — SimSpec dataclass, ML formula, and validation.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from .orientations import ORIENT
from .species import SPECIES


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
    species:             str   = "O"
    energy:              float = 0.5      # eV
    angle:               float = 0.0      # degrees from surface normal
    fluence:             int   = 50       # monolayers
    ml:                  int   = 0        # atoms per monolayer; 0 = compute from ml_factor*x*y
    box_x:               int   = 9        # lattice units
    box_y:               int   = 9        # lattice units
    box_depth:           int   = 3        # lat_top in lattice units
    impact_time:         float = 2000.0   # fs
    thermalization_time: float = 500.0    # fs
    wall_hours:          int   = 24
    name:                str   = ""
    account:             str   = "dgraves"
    email:               str   = ""    # empty = no mail directives
    lammps_module:       str   = "lammps/kokkos/gpu_della9_2022"


def compute_ml(orientation: str, box_x: int, box_y: int) -> int:
    """Return the atoms-per-monolayer count for a given orientation and box size.

    Formula: ML = ml_factor * box_x * box_y
    """
    return ORIENT[orientation]["ml_factor"] * box_x * box_y


def validate(spec: SimSpec) -> None:
    """Validate a SimSpec; exit with an informative message on any error."""
    if spec.orientation not in ORIENT:
        sys.exit(f"Unknown orientation '{spec.orientation}'. Choose from: {list(ORIENT)}")
    if spec.species not in SPECIES:
        sys.exit(f"Unknown species '{spec.species}'. Choose from: {list(SPECIES)}")

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
