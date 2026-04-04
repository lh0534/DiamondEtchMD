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
    orientation:         str   = "001"
    reconstruction:      str   = "bare"   # bare, 2x1 (001); bare/1x1/2x1_single/2x1_pandey (111); bare/O (113)
    termination:         str   = "bare"   # bare, H, O, O_ether
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


def compute_ml(orientation: str, box_x: int, box_y: int) -> int:
    """Return the atoms-per-monolayer count for a given orientation and box size.

    Formula: ML = ml_factor * box_x * box_y

    ML factors (analytically derived, empirically verified):
      001 → 1  (verified: 9×9 → 81)
      111 → 2  (verified: 5×9 → 90)
      113 → 4  (verified: 9×3 → 108)
    """
    return ORIENT[orientation]["ml_factor"] * box_x * box_y


def validate(spec: SimSpec, dfiles_root: Path = None) -> None:
    """Validate a SimSpec; exit with an informative message on any error.

    Parameters
    ----------
    spec:
        The simulation specification to validate.
    dfiles_root:
        Optional path to the dfiles/ directory.  When provided, the function
        also checks that the make_surf source file exists on disk.
    """
    if spec.orientation not in ORIENT:
        sys.exit(f"Unknown orientation '{spec.orientation}'. Choose from: {list(ORIENT)}")
    if spec.species not in SPECIES:
        sys.exit(f"Unknown species '{spec.species}'. Choose from: {list(SPECIES)}")

    ms = ORIENT[spec.orientation]["make_surf"]
    if "*" not in ms and spec.reconstruction not in ms:
        sys.exit(
            f"Reconstruction '{spec.reconstruction}' not available for {spec.orientation}. "
            f"Choose from: {[k for k in ms]}"
        )

    valid_term = {"bare", "H", "O", "O_ether"}
    if spec.termination not in valid_term:
        sys.exit(f"Unknown termination '{spec.termination}'. Choose from: {valid_term}")

    # Warn about combinations that aren't verified
    if (
        spec.orientation == "113"
        and spec.reconstruction not in ("bare", "O")
        and spec.termination not in ("bare", "O")
    ):
        print(
            f"Warning: termination '{spec.termination}' on 113 uses the bare make_surf.lmp "
            f"with config vars; verify output.",
            file=sys.stderr,
        )

    if spec.ml <= 0:
        sys.exit("ML (atoms per monolayer) must be > 0.")

    if dfiles_root is not None:
        from .builder import get_make_surf_source
        src = get_make_surf_source(spec, dfiles_root)
        if not src.exists():
            sys.exit(f"make_surf source not found: {src}")
