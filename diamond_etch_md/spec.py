"""
spec.py — SimSpec dataclass, ML formula, and validation.

Simulation modes
----------------
ion-etch       — single species, ions only, no radicals (flux_ratio == 0, phases is None,
                 ion_mix is None).
rie-etch       — single species, ions with O• radicals before each impact (flux_ratio > 0,
                 phases is None, ion_mix is None). Requires a non-Ar species.
multi-ion-etch — multiple ion species sampled stochastically, no radicals
                 (ion_mix is not None, flux_ratio == 0).
multi-rie-etch — multiple ion species sampled stochastically, with O• radical pre-exposure
                 (ion_mix is not None, flux_ratio > 0).
cycle-etch     — multi-phase cycling (phases is not None).
ALE-etch       — cycle-etch with exactly 2 phases; validated via make_ale().
"""

import dataclasses
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .orientations import ORIENT
from .species import SPECIES


@dataclass
class IonComponent:
    """One ion species in a multi-ion stochastic mix.

    `fraction` is the probability of selecting this ion per impact; all fractions
    must sum to 1.0.  `energy` is the total kinetic energy in eV (for O2, this is
    the full dimer energy, split per-atom at generation time).
    """
    species:  str
    fraction: float  # 0 < fraction <= 1; must sum to 1.0 across all components
    energy:   float  # eV (total dimer energy for O2)


@dataclass
class CyclePhase:
    """One phase of a cycling simulation.

    Each phase defines an ion species, its energy, how many monolayers to run
    per cycle repetition, the number of O• radicals deposited before each ion
    impact (flux_ratio), and the kinetic energy or temperature of those radicals.
    """
    species:                str
    energy:                 float          # eV  (total dimer energy for O2)
    fluence_ml:             int            # ML of this ion species per cycle
    flux_ratio:             int   = 0      # O• radicals deposited per ion impact (0 = none)
    radical_energy:         float = 0.2    # eV per O• radical (ignored when radical_temperature set)
    radical_temperature:    Optional[float] = None  # K; enables Maxwell-Boltzmann radical sampling
    radical_angle:          float = 0.0    # deg from normal for fixed-angle radicals
    radical_angle_distribution:     bool  = False  # cosine (Lambert) angle distribution for O• radicals
    max_inter_neutral_time: float = 5000.0 # fs; cap on per-radical halt time in stochastic mode
    radical_i_above:       float = 12.0   # Å above surface to inject O• radical


@dataclass
class SimSpec:
    orientation:            str   = "100"
    surface:                str   = "1x1"       # surface state (reconstruction + termination)
                                                 # 100: 1x1, 2x1, 2x1_O, O_ether
                                                 # 110: "", O
                                                 # 111: 1x1, 2x1_single, 2x1_pandey,
                                                 #      1x1_O, 2x1_single_O, 2x1_pandey_O
                                                 # 113: "", O
    surface_temperature:    float = 300.0    # K  (substrate thermostat temperature)
    species:                str   = "O"      # single-species mode only
    energy:                 float = 0.5      # eV  (single-species mode only)
    ion_angle:              float = 0.0      # degrees from surface normal (ion beam)
    fluence:                int   = 50       # monolayers (single-species mode only)
    ml:                     int   = 0        # atoms per monolayer; 0 = compute from ml_factor*x*y
    box_x:                  int   = 9        # lattice units
    box_y:                  int   = 9        # lattice units
    box_depth:              int   = 3        # lat_top in lattice units
    impact_time:            float = 1000.0   # fs — ion impact window
    thermalization_time:    float = 500.0    # fs — post-impact thermalisation
    inter_neutral_time:     float = 1500.0   # fs — O• radical impact window (fixed-angle monoenergetic mode)
    wall_hours:             int   = 24
    name:                   str   = ""
    account:                str   = "dgraves"
    email:                  str   = ""    # empty = no mail directives
    lammps_module:          str   = "lammps/kokkos/gpu_della9_2022"
    plot_interval_hours:    int   = 12     # hours between auto-plot runs (0 = disabled)
    cna_stride:             int   = 0      # CNA stride for --cna mode (0 = 1 per ML)
    nice:                   int   = 2      # SLURM --nice priority offset (≥ 1)
    remove_ar:              bool  = True   # delete Ar atoms after each impact; set False to retain
    seed_adjust:            int   = 0      # random seed offset; increment for independent replicas
    # ── RIE-etch mode (single-species with radical pre-exposure) ──────────────
    flux_ratio:             int   = 0      # O• radicals per ion impact (0 = ion-etch; >0 = RIE-etch)
    radical_energy:         float = 0.2    # eV per O• (used when radical_temperature is None)
    radical_temperature:    Optional[float] = None  # K; enables Maxwell-Boltzmann speed sampling
    radical_angle:          float = 0.0    # deg from normal for radicals in fixed-angle mode
    radical_angle_distribution:     bool  = False  # cosine (Lambert) angle distribution for O• radicals
    max_inter_neutral_time: float = 5000.0 # fs; cap on per-radical halt time in stochastic mode
    radical_i_above:       float = 12.0   # Å above surface to inject O• radical
    dump_mode:              str   = "all"  # "all" | "etch_only" | "none"
    # ── Cycling mode ──────────────────────────────────────────────────────────
    phases:                 Optional[List[CyclePhase]]    = None  # None = single-species mode
    cycles:                 int   = 1      # how many times the phase list repeats
    # ── Multi-ion mode ────────────────────────────────────────────────────────
    ion_mix:                Optional[List[IonComponent]] = None   # None = single-species mode

    @classmethod
    def from_dict(cls, data: dict) -> "SimSpec":
        """Deserialize from a dict, handling renamed fields for backward compat."""
        d = dict(data)
        # field renames (old spec.json → new field names)
        if "temperature" in d and "surface_temperature" not in d:
            d["surface_temperature"] = d.pop("temperature")
        if "angle" in d and "ion_angle" not in d:
            d["ion_angle"] = d.pop("angle")
        if "angle_distribution" in d and "radical_angle_distribution" not in d:
            d["radical_angle_distribution"] = d.pop("angle_distribution")
        if "chemical_i_above" in d and "radical_i_above" not in d:
            d["radical_i_above"] = d.pop("chemical_i_above")
        valid = {f.name for f in dataclasses.fields(cls)}
        # nested dataclasses
        if "phases" in d and d["phases"] is not None:
            d["phases"] = [CyclePhase(**p) for p in d["phases"]]
        if "ion_mix" in d and d["ion_mix"] is not None:
            d["ion_mix"] = [IonComponent(**c) for c in d["ion_mix"]]
        return cls(**{k: v for k, v in d.items() if k in valid})


def compute_ml(orientation: str, box_x: int, box_y: int) -> int:
    """Return the atoms-per-monolayer count for a given orientation and box size.

    Formula: ML = ml_factor * box_x * box_y
    """
    return ORIENT[orientation]["ml_factor"] * box_x * box_y


def etch_mode(spec: "SimSpec") -> str:
    """Return the etch mode string for a SimSpec.

    Returns:
        "ion-etch"       — single species, no radicals
        "rie-etch"       — single species with O• radicals
        "multi-ion-etch" — stochastic multi-ion mix, no radicals
        "multi-rie-etch" — stochastic multi-ion mix with O• radicals
        "cycle-etch"     — multi-phase cycling
    """
    if spec.phases is not None:
        return "cycle-etch"
    if spec.ion_mix is not None:
        return "multi-rie-etch" if spec.flux_ratio > 0 else "multi-ion-etch"
    if spec.flux_ratio > 0:
        return "rie-etch"
    return "ion-etch"


def validate(spec: "SimSpec") -> None:
    """Validate a SimSpec; exit with an informative message on any error."""
    if spec.orientation not in ORIENT:
        sys.exit(f"Unknown orientation '{spec.orientation}'. Choose from: {list(ORIENT)}")

    if spec.dump_mode not in ("all", "etch_only", "none"):
        sys.exit(f"dump_mode must be 'all', 'etch_only', or 'none'; got '{spec.dump_mode}'.")

    if spec.radical_temperature is not None and spec.radical_temperature <= 0:
        sys.exit(f"radical_temperature must be > 0 K; got {spec.radical_temperature}.")

    orient_cfg = ORIENT[spec.orientation]
    valid_surfaces = list(orient_cfg["surfaces"])
    if spec.surface not in orient_cfg["surfaces"]:
        sys.exit(
            f"Surface '{spec.surface}' not valid for {spec.orientation}. "
            f"Choose from: {valid_surfaces}"
        )

    if spec.ml <= 0:
        sys.exit("ML (atoms per monolayer) must be > 0.")

    if spec.nice < 1:
        sys.exit(f"nice must be >= 1, got {spec.nice}.")

    if spec.orientation == "100" and spec.surface in ("2x1", "2x1_O"):
        if spec.box_x % 2 != 0 or spec.box_y % 2 != 0:
            sys.exit(
                f"C(100) 2×1 reconstruction requires even box dimensions; "
                f"got box_x={spec.box_x}, box_y={spec.box_y}."
            )

    if spec.phases is not None:
        # ── Cycling-mode validation ────────────────────────────────────────
        # (flux_ratio and radical_energy on SimSpec are ignored in cycle-etch)
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
    elif spec.ion_mix is not None:
        # ── Multi-ion-mode validation ──────────────────────────────────────
        if len(spec.ion_mix) < 2:
            sys.exit("ion_mix must have at least 2 components.")
        for i, comp in enumerate(spec.ion_mix):
            if comp.species not in SPECIES:
                sys.exit(
                    f"ion_mix[{i}] ({comp.species!r}): unknown species. "
                    f"Choose from: {list(SPECIES)}"
                )
            if comp.fraction <= 0:
                sys.exit(f"ion_mix[{i}]: fraction must be > 0, got {comp.fraction}.")
            if comp.energy <= 0:
                sys.exit(f"ion_mix[{i}]: energy must be > 0, got {comp.energy}.")
        total = sum(c.fraction for c in spec.ion_mix)
        if abs(total - 1.0) > 0.01:
            sys.exit(
                f"ion_mix fractions must sum to 1.0, got {total:.4f}. "
                f"Tip: pass fractions that sum to 1 (e.g. 0.5, 0.5) or use "
                f"normalize_ion_mix() to auto-normalize."
            )
        if spec.flux_ratio < 0:
            sys.exit(f"flux_ratio must be >= 0, got {spec.flux_ratio}.")
        if spec.flux_ratio > 0 and spec.radical_energy <= 0:
            sys.exit(
                f"radical_energy must be > 0 when flux_ratio > 0, "
                f"got {spec.radical_energy}."
            )
    else:
        # ── Single-species-mode validation (ion-etch or RIE-etch) ──────
        if spec.species not in SPECIES:
            sys.exit(f"Unknown species '{spec.species}'. Choose from: {list(SPECIES)}")
        if spec.flux_ratio < 0:
            sys.exit(f"flux_ratio must be >= 0, got {spec.flux_ratio}.")
        if spec.flux_ratio > 0:
            if spec.radical_energy <= 0:
                sys.exit(
                    f"radical_energy must be > 0 when flux_ratio > 0, "
                    f"got {spec.radical_energy}."
                )


def normalize_ion_mix(mix: List[IonComponent]) -> List[IonComponent]:
    """Return a copy of mix with fractions normalized to sum exactly to 1.0."""
    total = sum(c.fraction for c in mix)
    return [IonComponent(c.species, c.fraction / total, c.energy) for c in mix]
