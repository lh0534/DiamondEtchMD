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

carbon-etch prefix — any of the above modes run on an arbitrary LAMMPS data file instead
                 of a generated diamond surface.  Triggered by setting initial_config_file.
                 Uses anchor_z_max (Å) to define the frozen region, Langmuir ML from the
                 box XY area, and no carbon replenishment.
"""

import dataclasses
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

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


def _norm_angle(v) -> Tuple[float, float]:
    """Normalize an angle spec to a (polar_deg, azimuth_deg) tuple.

    Accepts float → (v, 90.0), tuple/list of 2 → (theta, phi).
    Azimuth default 90° maintains backward-compat YZ-plane tilt (velx=0).
    """
    if isinstance(v, (int, float)):
        return (float(v), 90.0)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return (float(v[0]), float(v[1]))
    raise ValueError(
        f"angle must be a float (polar) or [polar, azimuth] pair; got {v!r}"
    )


@dataclass
class CyclePhase:
    """One phase of a cycling simulation.

    Each phase defines an ion species (or ion mix), its energy, how many
    monolayers to run per cycle repetition, the number of O• radicals deposited
    before each ion impact (flux_ratio), and the kinetic energy or temperature
    of those radicals.

    For a stochastic mixture of ion species within a phase set ion_mix to a
    list of IonComponent entries (fractions must sum to 1.0) and leave species
    as an empty string.  Only one of species / ion_mix may be set per phase.
    """
    species:                str   = ""
    energy:                 float = 0.0    # eV  (total dimer energy for O2; ignored when ion_mix set)
    fluence_ml:             int   = 0      # ML of this phase per cycle
    flux_ratio:             float = 0      # O• radicals deposited per ion impact (0 = none; non-integer → stochastic floor/ceil)
    radical_energy:         float = 0.2    # eV per O• radical (ignored when radical_temperature set)
    radical_temperature:    Optional[float] = None  # K; enables Maxwell-Boltzmann radical sampling
    radical_energy_threshold: Optional[float] = None  # eV; Boltzmann draws below this are skipped (no MD, counted as reflected)
    radical_threshold_on:   str   = "total"  # "total" = E, "normal" = E*cos^2(theta) compared to the threshold
    radical_energy_dist:    str   = "flux"   # Boltzmann draw: "flux" = E*exp(-E/kT) (striking flux) | "gas" = sqrt(E)*exp(-E/kT)
    radical_angle:          Union[float, Tuple[float, float]] = 0.0  # deg from normal; or (polar, azimuth)
    radical_angle_distribution:     bool  = False  # cosine (Lambert) angle distribution for O• radicals
    max_inter_neutral_time: float = 5000.0 # fs; cap on per-radical halt time in stochastic mode
    radical_i_above:        float = 6.0    # Å above surface to inject O• radical
    radical_burst:          bool  = False  # inject flux_ratio O• as a burst before the ion (same sampling options as one-by-one)
    radical_burst_chunk:    int   = 0      # atoms per burst chunk; 0 = auto (min(0.5 ML, what fits in one layer))
    radical_burst_attempt:  int   = 5000   # placement attempts per atom (cheap; 200 dropped ~2% of 10-atom chunks in a 6x6 cell)
    radical_burst_near:     float = 3.5    # Å; minimum distance of a burst atom from any atom at insertion
    ion_mix:                Optional[List["IonComponent"]] = None  # stochastic multi-ion mix for this phase

    def __post_init__(self):
        self.radical_angle = _norm_angle(self.radical_angle)
        if self.ion_mix is not None:
            self.ion_mix = [IonComponent(**c) if isinstance(c, dict) else c for c in self.ion_mix]


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
    ion_angle:              Union[float, Tuple[float, float]] = 0.0  # deg from normal; or (polar, azimuth)
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
    remove_ar:              bool  = False  # delete Ar/ZBL ion after each impact; False keeps ion in impact_snaps
    seed_adjust:            int   = 0      # random seed offset; increment for independent replicas
    # ── RIE-etch mode (single-species with radical pre-exposure) ──────────────
    flux_ratio:             float = 0      # O• radicals per ion impact (0 = ion-etch; >0 = RIE-etch; non-integer → stochastic floor/ceil)
    radical_energy:         float = 0.2    # eV per O• (used when radical_temperature is None)
    radical_temperature:    Optional[float] = None  # K; enables Maxwell-Boltzmann speed sampling
    radical_energy_threshold: Optional[float] = None  # eV; Boltzmann draws below this are skipped (no MD, counted as reflected)
    radical_threshold_on:   str   = "total"  # "total" = E, "normal" = E*cos^2(theta) compared to the threshold
    radical_energy_dist:    str   = "flux"   # Boltzmann draw: "flux" = E*exp(-E/kT) (striking flux) | "gas" = sqrt(E)*exp(-E/kT)
    radical_angle:          Union[float, Tuple[float, float]] = 0.0  # deg from normal; or (polar, azimuth)
    radical_angle_distribution:     bool  = False  # cosine (Lambert) angle distribution for O• radicals
    max_inter_neutral_time: float = 5000.0 # fs; cap on per-radical halt time in stochastic mode
    radical_i_above:        float = 6.0    # Å above surface to inject O• radical
    skip_radical_thermalization: bool = False  # omit thermalize.lmp between successive radicals
    radical_burst:          bool  = False  # inject flux_ratio O• as a burst instead of one-by-one (same sampling options)
    radical_burst_chunk:    int   = 0      # max atoms deposited before running dynamics; 0 = auto (min(0.5 ML, what fits in one layer))
    radical_burst_attempt:  int   = 5000   # placement attempts per atom (cheap; 200 dropped ~2% of 10-atom chunks in a 6x6 cell)
    radical_burst_near:     float = 3.5    # Å; minimum distance of a burst atom from any atom at insertion
    dump_mode:              str   = "all"  # "all" | "etch_only" | "none"
    above_surf_eject:       bool  = True   # require ejected cluster CoM above surface before counting
    dump_first_impact:      bool  = True   # in etch_only mode, always keep dump for 1st ion + 1st radical
    # ── Cycling mode ──────────────────────────────────────────────────────────
    phases:                 Optional[List[CyclePhase]]    = None  # None = single-species mode
    cycles:                 int   = 1      # how many times the phase list repeats
    # ── Multi-ion mode ────────────────────────────────────────────────────────
    ion_mix:                Optional[List[IonComponent]] = None   # None = single-species mode
    # ── Carbon-etch mode (arbitrary initial config file) ─────────────────────
    initial_config_file:    Optional[str]   = None   # abs path to LAMMPS data file; triggers carbon-etch
    anchor_z_max:           Optional[float] = None   # Å; top of frozen anchor region (required for carbon-etch)
    initial_thermalization: bool            = False  # run NVT equilibration before impacts (carbon-etch)
    initial_thermalization_steps: int       = 10000  # NVT steps for initial thermalization
    # ── Single-impact statistics mode ─────────────────────────────────────────
    single_impact:          bool  = False   # repeat single impact from fresh surface N times
    n_trials:               int   = 100     # number of independent single-impact trials
    randomize_velocities:   bool  = False   # re-assign thermal velocities from Boltzmann each trial
    # ── Deposition mask ───────────────────────────────────────────────────────
    mask_type:   Optional[str]                      = None  # "xymask" | "xmask" | "ymask"
    mask_width:  Union[float, Tuple[float, float]]  = 0.1   # fraction of box; (x_frac, y_frac) for xymask
    invert_mask:       bool  = False  # deposit at edges instead of center
    freeze_mask:       bool  = False  # freeze top-layer atoms in masked region into anchor group
    freeze_mask_depth: float = 2.0    # Å depth from surface to freeze (≈1-2 diamond layers)
    # ── Step edge ─────────────────────────────────────────────────────────────
    step_edge:     bool           = False  # add a monoatomic step edge (works with any surface/termination)
    step_angle:    float          = 0.0   # deg; step edge runs at this angle from x-axis
                                           # 0°  → edge ∥ y, lower terrace is −x half
                                           # 90° → edge ∥ x, lower terrace is −y half
                                           # C(111): 0° = 0°-step, 90° = 30°-step in crystal coords
    step_position: Union[float, List[float]] = 0.5  # step location as fraction of box length (0–1); or [start, end] for a finite-width band
    step_invert:   bool           = False  # True → remove upper terrace from +normal half instead
    step_depth:    Optional[float] = None  # bilayer thickness to remove, Å; None = orientation default
    # ── CO₂ auto-desorption ───────────────────────────────────────────────────
    co2_desorb_fraction: float = 0.0  # fraction of CO2-precursor C atoms force-desorbed before each thermalization

    def __post_init__(self):
        # Normalize angle specs to (polar, azimuth) tuples.
        self.ion_angle = _norm_angle(self.ion_angle)
        self.radical_angle = _norm_angle(self.radical_angle)
        # Normalize xymask single-float shorthand to a (wx, wy) tuple.
        if self.mask_type == "xymask" and isinstance(self.mask_width, (int, float)):
            self.mask_width = (float(self.mask_width), float(self.mask_width))

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
        if "mask_invert" in d and "invert_mask" not in d:
            d["invert_mask"] = d.pop("mask_invert")
        if "sweep_z_check" in d and "above_surf_eject" not in d:
            d["above_surf_eject"] = d.pop("sweep_z_check")
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = {k for k in d if k not in valid}
        if unknown:
            warnings.warn(f"Unknown SimSpec fields ignored: {sorted(unknown)}", stacklevel=2)
        # Specs written before radical_energy_dist existed used the gas-phase
        # draw; keep that so regenerating/re-analysing a mid-run sim is consistent.
        d.setdefault("radical_energy_dist", "gas")
        # Bursts written before radical_burst_near existed used fix deposit's
        # default lattice units: "near 2.0" was 2.0 * xlattice = 5.04 Å on C(100).
        d.setdefault("radical_burst_near", _LEGACY_BURST_NEAR)
        # nested dataclasses
        if "phases" in d and d["phases"] is not None:
            d["phases"] = [CyclePhase(**{"radical_energy_dist": "gas",
                                         "radical_burst_near": _LEGACY_BURST_NEAR, **p})
                           for p in d["phases"]]
        if "ion_mix" in d and d["ion_mix"] is not None:
            d["ion_mix"] = [IonComponent(**c) for c in d["ion_mix"]]
        return cls(**{k: v for k, v in d.items() if k in valid})


def compute_ml(orientation: str, box_x: int, box_y: int) -> int:
    """Return the atoms-per-monolayer count for a given orientation and box size.

    Formula: ML = ml_factor * box_x * box_y
    """
    return ORIENT[orientation]["ml_factor"] * box_x * box_y


_LEGACY_BURST_NEAR = 5.04   # Å; old "near 2.0" in lattice units on C(100)
_LAT_A = 3.5654             # Å, diamond
# Surface area per ML atom, in units of lat_a^2 (lattice-cell area / ml_factor)
_AREA_PER_ML_ATOM = {"100": 0.5, "110": 0.3536, "111": 0.4330, "113": 0.4146,
                     # hexagonal diamond (ideal lonsdaleite): cell area / ml_factor
                     "hex0001": 0.4330, "hex1120": 0.3536, "hex1010": 0.4082}
# Random sequential placement of disks jams at ~55% coverage; stay below it so
# every atom of a chunk lands in its single insertion layer.
_BURST_MAX_COVERAGE = 0.45


def burst_max_chunk(spec: "SimSpec", near: float) -> int:
    """Largest burst chunk that fits in one insertion layer at spacing `near` (Å)."""
    if spec.initial_config_file is not None:
        area = spec.ml * 10.0                       # Langmuir ML: A = 10 Å^2 per atom
    else:
        area = spec.ml * _AREA_PER_ML_ATOM[spec.orientation] * _LAT_A ** 2
    return max(1, int(_BURST_MAX_COVERAGE * area / (3.141592653589793 * near * near / 4)))


def burst_chunks(spec: "SimSpec", flux_ratio: float, chunk: int, near: float) -> List[int]:
    """Chunk sizes for one burst of round(flux_ratio) radicals.

    chunk = 0 picks min(0.5 ML, burst_max_chunk); all atoms of a chunk are
    inserted in one layer before any dynamics run.
    """
    size = chunk if chunk > 0 else min(max(1, round(0.5 * spec.ml)), burst_max_chunk(spec, near))
    total = max(1, round(flux_ratio))
    return [size] * (total // size) + ([total % size] if total % size else [])


def parse_data_file_box(path: str) -> Tuple[float, float]:
    """Parse lx, ly (Å) from a LAMMPS data file header.

    Reads until both 'xlo xhi' and 'ylo yhi' lines are found.
    Raises ValueError if either dimension cannot be found.
    """
    lx = ly = None
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if "xlo xhi" in stripped:
                parts = stripped.split()
                lx = float(parts[1]) - float(parts[0])
            elif "ylo yhi" in stripped:
                parts = stripped.split()
                ly = float(parts[1]) - float(parts[0])
            if lx is not None and ly is not None:
                break
    if lx is None or ly is None:
        raise ValueError(f"Could not parse box dimensions from {path}")
    return lx, ly


def compute_ml_langmuir(lx: float, ly: float) -> int:
    """Return ML (impacts per monolayer) from box XY dimensions using Langmuir normalization.

    1 Langmuir ML = 10^15 ions/cm².  Box area A_Å² in Å² = A_cm² * 1e16 cm².
    Impacts/ML = 10^15 * A_cm² = 10^15 * A_Å² * 1e-16 = A_Å² / 10.
    """
    return max(1, round(lx * ly / 10))


def etch_mode(spec: "SimSpec") -> str:
    """Return the etch mode string for a SimSpec.

    Returns:
        "ion-etch"              — single species, no radicals
        "rie-etch"              — single species with O• radicals
        "multi-ion-etch"        — stochastic multi-ion mix, no radicals
        "multi-rie-etch"        — stochastic multi-ion mix with O• radicals
        "burst-rie-etch"        — single species with O• radicals injected in bursts
        "cycle-etch"            — multi-phase cycling
        "carbon-ion-etch"       — carbon-etch + ion-etch
        "carbon-rie-etch"       — carbon-etch + rie-etch
        "carbon-multi-ion-etch" — carbon-etch + multi-ion-etch
        "carbon-multi-rie-etch" — carbon-etch + multi-rie-etch
        "carbon-cycle-etch"     — carbon-etch + cycle-etch
    """
    prefix = "carbon-" if spec.initial_config_file else ""
    if spec.phases is not None:
        return f"{prefix}cycle-etch"
    if spec.ion_mix is not None:
        return f"{prefix}multi-rie-etch" if spec.flux_ratio > 0 else f"{prefix}multi-ion-etch"
    if spec.flux_ratio > 0:
        return f"{prefix}burst-rie-etch" if spec.radical_burst else f"{prefix}rie-etch"
    return f"{prefix}ion-etch"


def _validate_radical_sampling(obj, where: str) -> None:
    """Checks shared by SimSpec and CyclePhase for the Boltzmann-radical options."""
    if obj.radical_energy_dist not in ("flux", "gas"):
        sys.exit(f"{where}radical_energy_dist must be 'flux' or 'gas'; got {obj.radical_energy_dist!r}.")
    if obj.radical_threshold_on not in ("total", "normal"):
        sys.exit(f"{where}radical_threshold_on must be 'total' or 'normal'; got {obj.radical_threshold_on!r}.")
    if obj.radical_energy_threshold is not None:
        if obj.radical_temperature is None:
            sys.exit(f"{where}radical_energy_threshold requires radical_temperature (Boltzmann radicals).")
        if obj.radical_energy_threshold < 0:
            sys.exit(f"{where}radical_energy_threshold must be >= 0 eV; got {obj.radical_energy_threshold}.")


def _validate_burst(spec: "SimSpec", obj, where: str) -> None:
    """Burst insertion geometry (obj = SimSpec or CyclePhase with radical_burst set)."""
    if obj.radical_burst_near <= 0:
        sys.exit(f"{where}radical_burst_near must be > 0 Å; got {obj.radical_burst_near}.")
    fit = burst_max_chunk(spec, obj.radical_burst_near)
    if obj.radical_burst_chunk > fit:
        sys.exit(
            f"{where}radical_burst_chunk={obj.radical_burst_chunk} cannot be placed in one layer "
            f"at radical_burst_near={obj.radical_burst_near} Å (at most {fit} fit in this "
            f"{spec.ml}-atom-ML box); use a smaller chunk or spacing."
        )


def validate(spec: "SimSpec") -> None:
    """Validate a SimSpec; exit with an informative message on any error."""
    is_carbon = spec.initial_config_file is not None

    if spec.dump_mode not in ("all", "etch_only", "none"):
        sys.exit(f"dump_mode must be 'all', 'etch_only', or 'none'; got '{spec.dump_mode}'.")

    if spec.surface_temperature <= 0:
        sys.exit(f"surface_temperature must be > 0 K; got {spec.surface_temperature}.")

    if spec.radical_temperature is not None and spec.radical_temperature <= 0:
        sys.exit(f"radical_temperature must be > 0 K; got {spec.radical_temperature}.")
    _validate_radical_sampling(spec, "")

    if not (0.0 <= spec.co2_desorb_fraction <= 1.0):
        sys.exit(f"co2_desorb_fraction must be between 0 and 1; got {spec.co2_desorb_fraction}.")

    if spec.radical_burst:
        if spec.ml <= 0:
            sys.exit(
                f"radical_burst requires ml > 0 (got {spec.ml}). "
                "Set ml explicitly when using burst mode."
            )
        _validate_burst(spec, spec, "")

    if spec.nice < 1:
        sys.exit(f"nice must be >= 1, got {spec.nice}.")

    if is_carbon:
        if spec.anchor_z_max is None:
            sys.exit("carbon-etch mode requires anchor_z_max (Å above box origin).")
        if spec.anchor_z_max <= 0:
            sys.exit(f"anchor_z_max must be > 0 Å; got {spec.anchor_z_max}.")
        if not os.path.exists(spec.initial_config_file):
            sys.exit(f"initial_config_file not found: {spec.initial_config_file}")
        # ml is computed from data file box at make_sim() time; skip ml <= 0 check
    else:
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
        # (flux_ratio and radical_energy on SimSpec are ignored in cycle-etch)
        if len(spec.phases) < 2:
            sys.exit("Cycling requires at least 2 phases.")
        if spec.cycles <= 0:
            sys.exit("cycles must be > 0.")
        for i, p in enumerate(spec.phases):
            has_species = bool(p.species)
            has_mix     = p.ion_mix is not None
            if has_species and has_mix:
                sys.exit(f"Phase {i}: set either species or ion_mix, not both.")
            if not has_species and not has_mix:
                sys.exit(f"Phase {i}: must set either species or ion_mix.")
            if has_mix:
                if len(p.ion_mix) < 2:
                    sys.exit(f"Phase {i}: ion_mix must have at least 2 components.")
                for j, comp in enumerate(p.ion_mix):
                    if comp.species not in SPECIES:
                        sys.exit(f"Phase {i} ion_mix[{j}] ({comp.species!r}): unknown species.")
                    if comp.fraction <= 0:
                        sys.exit(f"Phase {i} ion_mix[{j}]: fraction must be > 0, got {comp.fraction}.")
                    if comp.energy <= 0:
                        sys.exit(f"Phase {i} ion_mix[{j}]: energy must be > 0, got {comp.energy}.")
                total_frac = sum(c.fraction for c in p.ion_mix)
                if abs(total_frac - 1.0) > 0.01:
                    sys.exit(f"Phase {i}: ion_mix fractions sum to {total_frac:.4f}, must be 1.0.")
            else:
                if p.species not in SPECIES:
                    sys.exit(
                        f"Phase {i} ({p.species!r}): unknown species. "
                        f"Choose from: {list(SPECIES)}"
                    )
                if p.energy <= 0:
                    sys.exit(f"Phase {i}: energy must be > 0 eV, got {p.energy}.")
            if p.fluence_ml <= 0:
                sys.exit(f"Phase {i}: fluence_ml must be > 0, got {p.fluence_ml}.")
            if p.flux_ratio < 0:
                sys.exit(f"Phase {i}: flux_ratio must be >= 0, got {p.flux_ratio}.")
            if p.flux_ratio > 0 and p.radical_energy <= 0:
                sys.exit(f"Phase {i}: radical_energy must be > 0 when flux_ratio > 0, got {p.radical_energy}.")
            if p.radical_temperature is not None and p.radical_temperature <= 0:
                sys.exit(f"Phase {i}: radical_temperature must be > 0 K, got {p.radical_temperature}.")
            _validate_radical_sampling(p, f"Phase {i}: ")
            if p.radical_burst:
                _validate_burst(spec, p, f"Phase {i}: ")
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
        if spec.energy <= 0:
            sys.exit(f"energy must be > 0 eV; got {spec.energy}.")
        if spec.fluence <= 0:
            sys.exit(f"fluence must be > 0 ML; got {spec.fluence}.")
        if spec.flux_ratio < 0:
            sys.exit(f"flux_ratio must be >= 0, got {spec.flux_ratio}.")
        if spec.flux_ratio > 0:
            if spec.radical_energy <= 0:
                sys.exit(
                    f"radical_energy must be > 0 when flux_ratio > 0, "
                    f"got {spec.radical_energy}."
                )


    if spec.mask_type is not None:
        valid_mask_types = ("xymask", "xmask", "ymask")
        if spec.mask_type not in valid_mask_types:
            sys.exit(f"mask_type must be one of {valid_mask_types} or None; got '{spec.mask_type}'.")
        if spec.phases is not None:
            sys.exit("mask_type is not yet supported for cycle-etch mode.")
        if spec.mask_type == "xymask":
            if not (isinstance(spec.mask_width, tuple) and len(spec.mask_width) == 2):
                sys.exit("xymask requires mask_width as a (x_frac, y_frac) tuple or a single float.")
            wx, wy = spec.mask_width
        else:
            if not isinstance(spec.mask_width, (int, float)):
                sys.exit(f"{spec.mask_type} requires mask_width as a single float; got {type(spec.mask_width).__name__}.")
            wx = wy = float(spec.mask_width)
        if spec.mask_type in ("xymask", "xmask") and not (0.0 < wx < 0.5):
            sys.exit(f"mask_width x-fraction must be in (0, 0.5); got {wx}.")
        if spec.mask_type in ("xymask", "ymask") and not (0.0 < wy < 0.5):
            sys.exit(f"mask_width y-fraction must be in (0, 0.5); got {wy}.")
    if spec.invert_mask and spec.mask_type is None:
        sys.exit("invert_mask=True requires mask_type to also be set.")
    if spec.freeze_mask and spec.mask_type is None:
        sys.exit("freeze_mask=True requires mask_type to also be set.")
    if spec.freeze_mask and spec.freeze_mask_depth <= 0.0:
        sys.exit(f"freeze_mask_depth must be > 0; got {spec.freeze_mask_depth}.")

    if spec.step_edge:
        pos = spec.step_position
        if isinstance(pos, (int, float)):
            p1, p2 = float(pos), 1.0
        else:
            if len(pos) != 2:
                sys.exit(f"step_position list must have exactly 2 elements; got {pos}.")
            p1, p2 = float(pos[0]), float(pos[1])
        if not (0.0 < p1 < 1.0):
            sys.exit(f"step_position start must be in (0, 1); got {p1}.")
        if not (p1 < p2 <= 1.0):
            sys.exit(f"step_position end must be in (start, 1]; got {p2}.")
        if spec.step_depth is not None and spec.step_depth <= 0.0:
            sys.exit(f"step_depth must be > 0; got {spec.step_depth}.")


def normalize_ion_mix(mix: List[IonComponent]) -> List[IonComponent]:
    """Return a copy of mix with fractions normalized to sum exactly to 1.0."""
    total = sum(c.fraction for c in mix)
    return [IonComponent(c.species, c.fraction / total, c.energy) for c in mix]
