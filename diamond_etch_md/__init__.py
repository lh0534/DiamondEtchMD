"""
DiamondEtchMD — LAMMPS simulation setup and analysis for diamond surface etching.

Supports radical (O, H) and ion bombardment of diamond C(100), C(111), and C(113)
surfaces using the ReaxFF potential within the LAMMPS radicals framework.

Simulation modes
----------------
theory-etch  — single species, ions only, no radicals (flux_ratio == 0).
RIE-etch     — single species, ions with O• radical pre-exposure (flux_ratio > 0).
cycle-etch   — multi-phase cycling (spec.phases is not None).
ALE-etch     — cycle-etch with exactly 2 phases; use make_ale() factory.
"""

from .spec import SimSpec, CyclePhase, compute_ml, validate, etch_mode
from .builder import make_sim, make_ale
from .orientations import ORIENT
from .species import SPECIES

__all__ = [
    "SimSpec", "CyclePhase", "compute_ml", "validate", "etch_mode",
    "make_sim", "make_ale",
    "ORIENT", "SPECIES",
]
