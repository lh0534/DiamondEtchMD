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
    reconstruction:      str   = "bare_1x1"  # bare_1x1, bare_2x1 (100)
                                              # bare_1x1, bare_2x1_single, bare_2x1_pandey (111)
                                              # bare (113)
    termination:         str   = "bare"       # bare, O, O_ether (100)
                                              # bare, O_1x1, O_2x1_single, O_2x1_pandey (111)
                                              # bare, O (113)
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


def compute_ml(orientation: str, box_x: int, box_y: int) -> int:
    """Return the atoms-per-monolayer count for a given orientation and box size.

    Formula: ML = ml_factor * box_x * box_y

    ML factors (analytically derived, empirically verified):
      100 → 1  (verified: 9×9 → 81)
      111 → 2  (verified: 5×9 → 90)
      113 → 4  (verified: 9×3 → 108)
    """
    return ORIENT[orientation]["ml_factor"] * box_x * box_y


def validate(spec: SimSpec) -> None:
    """Validate a SimSpec; exit with an informative message on any error."""
    if spec.orientation not in ORIENT:
        sys.exit(f"Unknown orientation '{spec.orientation}'. Choose from: {list(ORIENT)}")
    if spec.species not in SPECIES:
        sys.exit(f"Unknown species '{spec.species}'. Choose from: {list(SPECIES)}")

    orient_cfg = ORIENT[spec.orientation]
    valid_reconstructions = list(orient_cfg["make_surf"])
    if spec.reconstruction not in orient_cfg["make_surf"]:
        sys.exit(
            f"Reconstruction '{spec.reconstruction}' not valid for {spec.orientation}. "
            f"Choose from: {valid_reconstructions}"
        )

    valid_terms = orient_cfg["valid_terminations"][spec.reconstruction]
    if spec.termination not in valid_terms:
        sys.exit(
            f"Termination '{spec.termination}' not valid for "
            f"{spec.orientation} / {spec.reconstruction}. "
            f"Choose from: {sorted(valid_terms)}"
        )

    if spec.ml <= 0:
        sys.exit("ML (atoms per monolayer) must be > 0.")
