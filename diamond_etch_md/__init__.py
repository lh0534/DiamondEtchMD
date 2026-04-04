"""
DiamondEtchMD — LAMMPS simulation setup and analysis for diamond surface etching.

Supports radical (O, H) and ion bombardment of diamond C(001), C(111), and C(113)
surfaces using the ReaxFF potential within the LAMMPS radicals framework.
"""

from .spec import SimSpec, compute_ml, validate
from .builder import make_sim
from .orientations import ORIENT
from .species import SPECIES

__all__ = ["SimSpec", "compute_ml", "validate", "make_sim", "ORIENT", "SPECIES"]
