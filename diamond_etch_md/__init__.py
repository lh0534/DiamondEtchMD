"""
DiamondEtchMD — LAMMPS simulation setup and analysis for diamond surface etching.

Supports radical (O, H) and ion bombardment of diamond C(100), C(111), and C(113)
surfaces using the ReaxFF potential within the LAMMPS radicals framework.

Simulation modes
----------------
ion-etch       — single species, ions only, no radicals (flux_ratio == 0).
rie-etch       — single species, ions with O• radical pre-exposure (flux_ratio > 0).
multi-ion-etch — stochastic multi-ion mix, no radicals (ion_mix is not None, flux_ratio == 0).
multi-rie-etch — stochastic multi-ion mix with O• radicals (ion_mix is not None, flux_ratio > 0).
cycle-etch     — multi-phase cycling (spec.phases is not None).
ALE-etch       — cycle-etch with exactly 2 phases; use make_ale() factory.
"""

from .spec import (SimSpec, CyclePhase, IonComponent, compute_ml, validate,
                   etch_mode, normalize_ion_mix,
                   parse_data_file_box, compute_ml_langmuir)
from .builder import make_sim, make_ale, multi_ion_dir_name
from .orientations import ORIENT
from .species import SPECIES

__all__ = [
    "SimSpec", "CyclePhase", "IonComponent",
    "compute_ml", "validate", "etch_mode", "normalize_ion_mix",
    "parse_data_file_box", "compute_ml_langmuir",
    "make_sim", "make_ale", "multi_ion_dir_name",
    "ORIENT", "SPECIES",
]
