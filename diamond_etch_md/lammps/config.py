"""
lammps/config.py — generator for config.lmp, the per-condition parameter file.

config.lmp is included at the top of head.lmp and defines all LAMMPS variables
that vary between simulation conditions: box size, energy, temperature, species
type, surface flags (reconstruct, O_terminate, O_ether_terminate), and timing.
"""

from ..orientations import ORIENT
from ..species import SPECIES
from ..spec import SimSpec, CyclePhase, IonComponent
import math

_kB = 8.617333262e-5  # eV/K

# LAMMPS 23 Jun 2022 Kokkos GPU: fix deposit with "region ... near X" (single-atom)
# applies ~12.713x too much kinetic energy to deposited atoms.  The "mol" (molecular)
# insertion path is unaffected.  Burst-mode O• radicals use the region/near path, so
# radical_energy is divided by this factor as a workaround.
#
# IMPORTANT: verify this bug is still present in your LAMMPS build before running
# burst simulations.  In a short test run, check c_ike (compute ike insert ke) at the
# first radical deposit step in the log.  You should see:
#   c_ike ≈ radical_energy_target × 23.06 kcal/mol   (bug absent — remove correction)
#   c_ike ≈ radical_energy_target × 23.06 × 12.713   (bug present — keep correction)
_DEPOSIT_REGION_BUG_FACTOR = 12.713

# Default bilayer removal depth per orientation (Å). Covers the two atom planes
# of one bilayer with a small margin.  Tune by inspection if the step cuts wrong.
_STEP_DEPTH_DEFAULT = {
    "100": 2.0,   # diamond(100): bilayer ≈ a/2 = 1.78 Å
    "110": 2.5,   # diamond(110) bilayer
    "111": 2.2,   # diamond(111): two planes ≈ 0.52 + 1.03 Å apart
    "113": 2.0,   # diamond(113) bilayer
}


def _zbl_mass_vars(species_iter) -> str:
    """Return 'variable M_X equal ...' lines for each unique ZBL species."""
    seen, out = set(), ""
    for sp in species_iter:
        if sp["needs_zbl"] and sp["mass_var"] not in seen:
            out += f"variable    {sp['mass_var']} equal {sp['mass']}\n"
            seen.add(sp["mass_var"])
    return out


def _radical_config_block(
    flux_ratio: int,
    radical_energy: float,
    radical_temperature,  # Optional[float]
    radical_angle,        # float or (polar_deg, azimuth_deg) tuple
    radical_angle_distribution: bool,
    max_inter_neutral_time: float,
    radical_i_above: float,
    inter_neutral_time: float,
    prefix: str = "",            # e.g. "phase_0_" for cycling per-phase vars
    burst_correction: bool = False,  # divide radical_energy by _DEPOSIT_REGION_BUG_FACTOR
) -> str:
    """Return config.lmp lines for radical-related variables.

    When radical_temperature is set, emits kT_rad and sigma_rad instead of
    radical_energy; inter_neutral_time is replaced by per-radical halt time at
    runtime, so only max_inter_neutral_time is emitted as a cap.
    """
    use_boltzmann = radical_temperature is not None
    use_cosine = radical_angle_distribution
    use_stochastic = use_boltzmann or use_cosine

    # Normalize radical_angle to (polar, azimuth) tuple
    if isinstance(radical_angle, (int, float)):
        rad_theta, rad_phi = float(radical_angle), 90.0
    else:
        rad_theta, rad_phi = float(radical_angle[0]), float(radical_angle[1])

    lines = f"variable    {prefix}flux_ratio equal {flux_ratio}\n"

    if use_boltzmann:
        kT = _kB * radical_temperature
        lines += (
            f"variable    {prefix}kT_rad equal {kT:.8f}        "
            f"# eV = kB * {radical_temperature} K\n"
        )
    elif burst_correction:
        corrected = radical_energy / _DEPOSIT_REGION_BUG_FACTOR
        lines += (
            f"# LAMMPS 23Jun2022 Kokkos GPU bug: fix deposit (region/near) gives {_DEPOSIT_REGION_BUG_FACTOR}x too much KE.\n"
            f"# radical_energy set to target/{_DEPOSIT_REGION_BUG_FACTOR} = {radical_energy}/{_DEPOSIT_REGION_BUG_FACTOR} eV.\n"
            f"# VERIFY: check c_ike at first radical deposit; should be {radical_energy * 23.06:.4f} kcal/mol.\n"
            f"# If c_ike = {corrected * 23.06:.4f} kcal/mol instead, the bug is fixed — remove correction.\n"
            f"variable    {prefix}radical_energy equal {corrected:.10g}  # target: {radical_energy} eV\n"
        )
    else:
        lines += f"variable    {prefix}radical_energy equal {radical_energy}\n"

    lines += f"variable    {prefix}radical_i_above equal {radical_i_above}\n"

    if use_stochastic:
        lines += (
            f"variable    {prefix}max_inter_neutral_time equal {max_inter_neutral_time}\n"
        )
    else:
        lines += f"variable    {prefix}inter_neutral_time equal {inter_neutral_time}\n"

    if not use_cosine:
        lines += f"variable    {prefix}rad_angl equal {rad_theta}\n"
        lines += f"variable    {prefix}rad_azimuth equal {rad_phi}\n"

    return lines


def _step_config_block(spec: SimSpec) -> str:
    """Return config.lmp variable lines for the step-edge modifier and sweep options."""
    depth = (spec.step_depth if spec.step_depth is not None
             else _STEP_DEPTH_DEFAULT.get(spec.orientation, 2.0))
    pos = spec.step_position
    if isinstance(pos, (int, float)):
        p1, p2 = float(pos), 1.0
    else:
        p1, p2 = float(pos[0]), float(pos[1])
    return (
        f"\n# Step edge\n"
        f"variable    step_edge      equal {'1' if spec.step_edge else '0'}\n"
        f"variable    step_angle     equal {spec.step_angle}\n"
        f"variable    step_position   equal {p1}\n"
        f"variable    step_position_2 equal {p2}\n"
        f"variable    step_invert    equal {'1' if spec.step_invert else '0'}\n"
        f"variable    step_depth_ang equal {depth}\n"
        f"\n# Sweep options\n"
        f"variable    do_z_check         equal {'1' if spec.sweep_z_check else '0'}\n"
        f"variable    co2_desorb_fraction equal {spec.co2_desorb_fraction}\n"
    )


def _mask_config_block(spec: SimSpec) -> str:
    """Return config.lmp variable lines for the deposition mask, or '' if no mask."""
    if spec.mask_type is None:
        return ""
    lines = "\n# Deposition mask: restrict ion/radical landing to the exposed window\n"
    if spec.mask_type in ("xymask", "xmask"):
        wx = spec.mask_width[0] if isinstance(spec.mask_width, tuple) else float(spec.mask_width)
        lines += (
            f"variable    mask_width_x equal {wx}   # fraction of Lx masked on each x-face\n"
            f"variable    mask_lo_x    equal xlo+(xhi-xlo)*v_mask_width_x\n"
            f"variable    mask_hi_x    equal xhi-(xhi-xlo)*v_mask_width_x\n"
        )
    if spec.mask_type in ("xymask", "ymask"):
        wy = spec.mask_width[1] if isinstance(spec.mask_width, tuple) else float(spec.mask_width)
        lines += (
            f"variable    mask_width_y equal {wy}   # fraction of Ly masked on each y-face\n"
            f"variable    mask_lo_y    equal ylo+(yhi-ylo)*v_mask_width_y\n"
            f"variable    mask_hi_y    equal yhi-(yhi-ylo)*v_mask_width_y\n"
        )
    return lines


def get_config_lmp(spec: SimSpec) -> str:
    """Generate the contents of config.lmp for the given SimSpec.

    Handles both ion-etch (flux_ratio == 0) and RIE-etch (flux_ratio > 0).
    When flux_ratio > 0, additional radical-related variables are appended.
    """
    species = SPECIES[spec.species]
    surf = ORIENT[spec.orientation]["surfaces"][spec.surface]

    recon_flag = "true" if surf["reconstruct"]   else "false"
    o_flag     = "true" if surf["O_terminate"]   else "false"
    o_eth_flag = "true" if surf["O_ether"]       else "false"

    # O2 energy is split across 2 atoms; user specifies total dimer energy
    energy_per_atom = spec.energy / species["energy_divisor"]

    _ion_theta, _ion_phi = spec.ion_angle
    cfg = (
        f"# DiamondEtchMD generated config (single-species)\n"
        f"# orientation={spec.orientation}  surface={spec.surface}\n"
        f"# species={spec.species}  energy={spec.energy}eV  ion_angle={_ion_theta}deg"
        f"  T={spec.surface_temperature}K\n"
        f"\n"
        f"variable    ML equal {spec.ml}            # atoms per monolayer\n"
        f"variable    end_fluence equal {spec.fluence}   # in ML\n"
        f"\n"
        f"variable    energ equal {energy_per_atom}     # incident energy per atom (eV)\n"
        f"variable    ion_angl equal {_ion_theta}   # ion beam angle (deg from normal)\n"
        f"variable    ion_azimuth equal {_ion_phi}  # ion beam azimuth (deg; 90=YZ plane)\n"
        f"variable    T equal {spec.surface_temperature}    # substrate temperature (K)\n"
        f"variable    pot string REAX\n"
        f"\n"
        f"variable    use_starting_data_file equal false\n"
        f"variable    starting_data_file string data.truncated\n"
        f"\n"
        f'if "${{pot}} == REAX" then &\n'
        f'"variable    lat_a file lat_a.txt"\n'
        f"\n"
        f"variable    x equal {spec.box_x}\n"
        f"variable    y equal {spec.box_y}\n"
        f"variable    lat_top equal {spec.box_depth}\n"
        f"variable    z equal ${{lat_top}}+5\n"
        f"\n"
        f"variable    reconstruct equal {recon_flag}\n"
        f"variable    O_terminate equal {o_flag}\n"
        f"variable    O_ether_terminate equal {o_eth_flag}\n"
        f"\n"
        f"variable    i_above equal {species['i_above']}    # Å above surface to inject ion\n"
        f"variable    impact_time equal {spec.impact_time}\n"
        f"variable    thermalization_time equal {spec.thermalization_time}\n"
        f"\n"
        f"variable    M_C equal 12.011\n"
        f"variable    M_H equal 1.00784\n"
        f"variable    M_O equal 16.0\n"
        + _zbl_mass_vars([species])
        + f"variable    M_incident equal ${{{species['mass_var']}}}\n"
        f"variable    incident_type_index equal {species['type_index']}  # {spec.species}\n"
        f"\n"
        f"variable    seed_adjust equal {spec.seed_adjust}\n"
        f"variable    simdepo equal 1\n"
    )

    if spec.flux_ratio > 0:
        cfg += (
            f"\n"
            f"# RIE-etch: O• radical pre-exposure before each ion impact\n"
            + _radical_config_block(
                flux_ratio=spec.flux_ratio,
                radical_energy=spec.radical_energy,
                radical_temperature=spec.radical_temperature,
                radical_angle=spec.radical_angle,
                radical_angle_distribution=spec.radical_angle_distribution,
                max_inter_neutral_time=spec.max_inter_neutral_time,
                radical_i_above=spec.radical_i_above,
                inter_neutral_time=spec.inter_neutral_time,
                burst_correction=spec.radical_burst,
            )
        )

    return cfg + _step_config_block(spec) + _mask_config_block(spec)


def get_config_lmp_multi_ion(spec: SimSpec) -> str:
    """Generate config.lmp for a multi-ion (ion_mix) SimSpec.

    Unlike single-species config, energ/M_incident/incident_type_index are NOT
    defined here — they are set dynamically per-impact by the ion selection block
    in head.lmp.  All mass variables are always defined so head.lmp can reference
    them regardless of which ion is selected.
    """
    surf = ORIENT[spec.orientation]["surfaces"][spec.surface]
    recon_flag = "true" if surf["reconstruct"] else "false"
    o_flag     = "true" if surf["O_terminate"] else "false"
    o_eth_flag = "true" if surf["O_ether"]     else "false"

    mix = spec.ion_mix
    total = sum(c.fraction for c in mix)
    ion_summary = ", ".join(
        f"{c.species}@{c.energy}eV×{c.fraction/total:.0%}" for c in mix
    )

    _ion_theta, _ion_phi = spec.ion_angle
    cfg = (
        f"# DiamondEtchMD generated config (multi-ion)\n"
        f"# orientation={spec.orientation}  surface={spec.surface}  T={spec.surface_temperature}K\n"
        f"# ions: {ion_summary}\n"
        f"\n"
        f"variable    ML equal {spec.ml}            # atoms per monolayer\n"
        f"variable    end_fluence equal {spec.fluence}   # in ML\n"
        f"\n"
        f"variable    ion_angl equal {_ion_theta}   # ion beam angle (deg from normal)\n"
        f"variable    ion_azimuth equal {_ion_phi}  # ion beam azimuth (deg; 90=YZ plane)\n"
        f"variable    T equal {spec.surface_temperature}    # substrate temperature (K)\n"
        f"variable    pot string REAX\n"
        f"\n"
        f"variable    use_starting_data_file equal false\n"
        f"variable    starting_data_file string data.truncated\n"
        f"\n"
        f'if "${{pot}} == REAX" then &\n'
        f'"variable    lat_a file lat_a.txt"\n'
        f"\n"
        f"variable    x equal {spec.box_x}\n"
        f"variable    y equal {spec.box_y}\n"
        f"variable    lat_top equal {spec.box_depth}\n"
        f"variable    z equal ${{lat_top}}+5\n"
        f"\n"
        f"variable    reconstruct equal {recon_flag}\n"
        f"variable    O_terminate equal {o_flag}\n"
        f"variable    O_ether_terminate equal {o_eth_flag}\n"
        f"\n"
        f"variable    i_above equal 6.0            # Å above surface to inject ion\n"
        f"variable    impact_time equal {spec.impact_time}\n"
        f"variable    thermalization_time equal {spec.thermalization_time}\n"
        f"\n"
        f"variable    M_C equal 12.011\n"
        f"variable    M_H equal 1.00784\n"
        f"variable    M_O equal 16.0\n"
        + _zbl_mass_vars(SPECIES[c.species] for c in mix)
        + ("variable    M_Ar equal 39.948\n" if not any(SPECIES[c.species]["needs_zbl"] for c in mix) else "")
        + f"\n"
        f"variable    seed_adjust equal {spec.seed_adjust}\n"
        f"variable    simdepo equal 1\n"
    )

    if spec.flux_ratio > 0:
        cfg += (
            f"\n"
            f"# Multi-ion RIE: O• radical pre-exposure before each ion impact\n"
            + _radical_config_block(
                flux_ratio=spec.flux_ratio,
                radical_energy=spec.radical_energy,
                radical_temperature=spec.radical_temperature,
                radical_angle=spec.radical_angle,
                radical_angle_distribution=spec.radical_angle_distribution,
                max_inter_neutral_time=spec.max_inter_neutral_time,
                radical_i_above=spec.radical_i_above,
                inter_neutral_time=spec.inter_neutral_time,
                burst_correction=spec.radical_burst,
            )
        )

    return cfg + _step_config_block(spec) + _mask_config_block(spec)


def get_config_lmp_cycle_etch(spec: SimSpec) -> str:
    """Generate config.lmp for a cycle-etch (multi-phase) SimSpec."""
    surf = ORIENT[spec.orientation]["surfaces"][spec.surface]
    recon_flag = "true" if surf["reconstruct"] else "false"
    o_flag     = "true" if surf["O_terminate"] else "false"
    o_eth_flag = "true" if surf["O_ether"]     else "false"

    total_phase_ml   = sum(p.fluence_ml for p in spec.phases)
    total_fluence_ml = spec.cycles * total_phase_ml  # total ML across all cycles

    # Per-phase parameter block
    phase_vars = ""
    for i, p in enumerate(spec.phases):
        sp = SPECIES[p.species]
        energy_per_atom = p.energy / sp["energy_divisor"]
        phase_vars += (
            f"variable    phase_{i}_ml equal {p.fluence_ml}       # ML per cycle ({p.species})\n"
            f"variable    phase_{i}_energy equal {energy_per_atom} # eV/atom\n"
        )
        if p.flux_ratio > 0:
            phase_vars += _radical_config_block(
                flux_ratio=p.flux_ratio,
                radical_energy=p.radical_energy,
                radical_temperature=p.radical_temperature,
                radical_angle=p.radical_angle,
                radical_angle_distribution=p.radical_angle_distribution,
                max_inter_neutral_time=p.max_inter_neutral_time,
                radical_i_above=p.radical_i_above,
                inter_neutral_time=spec.inter_neutral_time,
                prefix=f"phase_{i}_",
            )
        else:
            phase_vars += f"variable    phase_{i}_flux_ratio equal 0\n"

    phase_summary = ", ".join(
        f"{p.species}@{p.energy}eV×{p.fluence_ml}ML"
        + (f"+O•R{p.flux_ratio}" if p.flux_ratio > 0 else "")
        for p in spec.phases
    )

    _ion_theta, _ion_phi = spec.ion_angle
    return (
        f"# DiamondEtchMD generated config (cycling mode)\n"
        f"# orientation={spec.orientation}  surface={spec.surface}  T={spec.surface_temperature}K\n"
        f"# phases: {phase_summary}\n"
        f"# {spec.cycles} cycle(s) × {total_phase_ml} ML/cycle = {total_fluence_ml} ML total\n"
        f"\n"
        f"variable    ML equal {spec.ml}            # atoms per monolayer\n"
        f"variable    cycles equal {spec.cycles}\n"
        f"variable    end_fluence equal {total_fluence_ml}  # total ML\n"
        f"\n"
        f"variable    ion_angl equal {_ion_theta}   # ion beam angle (deg from normal)\n"
        f"variable    ion_azimuth equal {_ion_phi}  # ion beam azimuth (deg; 90=YZ plane)\n"
        f"variable    T equal {spec.surface_temperature}    # substrate temperature (K)\n"
        f"variable    pot string REAX\n"
        f"\n"
        f"variable    use_starting_data_file equal false\n"
        f"variable    starting_data_file string data.truncated\n"
        f"\n"
        f'if "${{pot}} == REAX" then &\n'
        f'"variable    lat_a file lat_a.txt"\n'
        f"\n"
        f"variable    x equal {spec.box_x}\n"
        f"variable    y equal {spec.box_y}\n"
        f"variable    lat_top equal {spec.box_depth}\n"
        f"variable    z equal ${{lat_top}}+5\n"
        f"\n"
        f"variable    reconstruct equal {recon_flag}\n"
        f"variable    O_terminate equal {o_flag}\n"
        f"variable    O_ether_terminate equal {o_eth_flag}\n"
        f"\n"
        f"variable    impact_time equal {spec.impact_time}\n"
        f"variable    thermalization_time equal {spec.thermalization_time}\n"
        f"\n"
        f"variable    ion_i_above equal 6.0     # Å above surface to inject ion\n"
        f"variable    radical_i_above equal {spec.radical_i_above}  # Å above surface to inject O• radical\n"
        f"\n"
        f"variable    M_C equal 12.011\n"
        f"variable    M_H equal 1.00784\n"
        f"variable    M_O equal 16.0\n"
        + _zbl_mass_vars(SPECIES[p.species] for p in spec.phases)
        + f"\n"
        f"variable    seed_adjust equal {spec.seed_adjust}\n"
        f"\n"
        f"# Per-phase parameters\n"
        f"{phase_vars}"
    ) + _step_config_block(spec) + _mask_config_block(spec)


def get_config_lmp_single_impact(spec: SimSpec) -> str:
    """Generate config.lmp for single-impact statistics mode.

    Works for both the crystal-builder path (initial_config_file is None) and the
    carbon-etch path.  n_trials replaces end_fluence as the loop termination counter.
    """
    is_carbon = spec.initial_config_file is not None
    species   = SPECIES[spec.species]
    energy_per_atom = spec.energy / species["energy_divisor"]

    _ion_theta, _ion_phi = spec.ion_angle
    common = (
        f"# DiamondEtchMD generated config (single-impact statistics mode)\n"
        f"# species={spec.species}  energy={spec.energy}eV"
        f"  ion_angle={_ion_theta}deg  T={spec.surface_temperature}K\n"
        f"# n_trials={spec.n_trials}"
        f"  randomize_velocities={spec.randomize_velocities}\n"
        f"\n"
        f"variable    n_trials equal {spec.n_trials}\n"
        f"\n"
        f"variable    energ equal {energy_per_atom}     # incident energy per atom (eV)\n"
        f"variable    ion_angl equal {_ion_theta}   # ion beam angle (deg from normal)\n"
        f"variable    ion_azimuth equal {_ion_phi}  # ion beam azimuth (deg; 90=YZ plane)\n"
        f"variable    T equal {spec.surface_temperature}    # substrate temperature (K)\n"
        f"variable    pot string REAX\n"
        f"\n"
        f"variable    impact_time equal {spec.impact_time}\n"
        f"variable    thermalization_time equal {spec.thermalization_time}\n"
        f"\n"
        f"variable    M_C equal 12.011\n"
        f"variable    M_H equal 1.00784\n"
        f"variable    M_O equal 16.0\n"
        + _zbl_mass_vars([species])
        + f"variable    M_incident equal ${{{species['mass_var']}}}\n"
        f"variable    incident_type_index equal {species['type_index']}  # {spec.species}\n"
        f"variable    i_above equal {species['i_above']}    # Å above surface to inject ion\n"
        f"\n"
        f"variable    seed_adjust equal {spec.seed_adjust}\n"
        f"variable    simdepo equal 1\n"
    )

    if is_carbon:
        _sqrt2 = math.sqrt(2)
        geom = (
            f"\n"
            f"variable    anchor_z_max equal {spec.anchor_z_max}\n"
            f"variable    bottom equal {spec.anchor_z_max}   # Å; for thermalize.lmp zbottom formula\n"
            f"variable    lat_a equal {_sqrt2:.7f}           # sqrt(2) → zbottom = ceil(bottom) Å\n"
        )
    else:
        surf      = ORIENT[spec.orientation]["surfaces"][spec.surface]
        recon_flag = "true" if surf["reconstruct"] else "false"
        o_flag     = "true" if surf["O_terminate"] else "false"
        o_eth_flag = "true" if surf["O_ether"]     else "false"
        geom = (
            f"\n"
            f'if "${{pot}} == REAX" then &\n'
            f'"variable    lat_a file lat_a.txt"\n'
            f"\n"
            f"variable    x equal {spec.box_x}\n"
            f"variable    y equal {spec.box_y}\n"
            f"variable    lat_top equal {spec.box_depth}\n"
            f"variable    z equal ${{lat_top}}+5\n"
            f"\n"
            f"variable    reconstruct equal {recon_flag}\n"
            f"variable    O_terminate equal {o_flag}\n"
            f"variable    O_ether_terminate equal {o_eth_flag}\n"
        )

    cfg = common + geom

    if spec.flux_ratio > 0:
        cfg += (
            f"\n"
            f"# RIE pre-exposure: {spec.flux_ratio} O• radicals per trial\n"
            + _radical_config_block(
                flux_ratio=spec.flux_ratio,
                radical_energy=spec.radical_energy,
                radical_temperature=spec.radical_temperature,
                radical_angle=spec.radical_angle,
                radical_angle_distribution=spec.radical_angle_distribution,
                max_inter_neutral_time=spec.max_inter_neutral_time,
                radical_i_above=spec.radical_i_above,
                inter_neutral_time=spec.inter_neutral_time,
                burst_correction=spec.radical_burst,
            )
        )

    return cfg + _step_config_block(spec) + _mask_config_block(spec)


def get_config_lmp_carbon_etch(spec: SimSpec) -> str:
    """Generate config.lmp for carbon-etch mode (any sub-mode).

    Replaces the diamond surface geometry block with anchor_z_max and Langmuir ML.
    Supports single-species, multi-ion, and cycle sub-modes by checking spec fields.
    lat_a is set to sqrt(2) so thermalize.lmp's zbottom formula gives anchor_z_max Å directly.
    """
    _sqrt2 = math.sqrt(2)

    # Determine ZBL species in use for this config before building common block
    if spec.phases is not None:
        _zbl_sp_list = [SPECIES[p.species] for p in spec.phases]
    elif spec.ion_mix is not None:
        _zbl_sp_list = [SPECIES[c.species] for c in spec.ion_mix]
    else:
        _zbl_sp_list = [SPECIES[spec.species]]

    _ion_theta, _ion_phi = spec.ion_angle
    common = (
        f"# DiamondEtchMD generated config (carbon-etch)\n"
        f"# initial_config_file: {spec.initial_config_file}\n"
        f"# anchor_z_max: {spec.anchor_z_max} Å\n"
        f"\n"
        f"variable    ML equal {spec.ml}\n"
        f"variable    ion_angl equal {_ion_theta}\n"
        f"variable    ion_azimuth equal {_ion_phi}\n"
        f"variable    T equal {spec.surface_temperature}\n"
        f"variable    pot string REAX\n"
        f"\n"
        f"variable    anchor_z_max equal {spec.anchor_z_max}\n"
        f"variable    bottom equal {spec.anchor_z_max}   # Å; for thermalize.lmp zbottom formula\n"
        f"variable    lat_a equal {_sqrt2:.7f}           # sqrt(2) → zbottom = ceil(bottom) Å\n"
        f"\n"
        f"variable    impact_time equal {spec.impact_time}\n"
        f"variable    thermalization_time equal {spec.thermalization_time}\n"
        f"\n"
        f"variable    M_C equal 12.011\n"
        f"variable    M_H equal 1.00784\n"
        f"variable    M_O equal 16.0\n"
        + _zbl_mass_vars(_zbl_sp_list)
        + f"\n"
        f"variable    seed_adjust equal {spec.seed_adjust}\n"
        f"variable    simdepo equal 1\n"
    )

    if spec.phases is not None:
        # ── Cycle sub-mode ────────────────────────────────────────────────────
        total_phase_ml   = sum(p.fluence_ml for p in spec.phases)
        total_fluence_ml = spec.cycles * total_phase_ml

        phase_vars = ""
        for i, p in enumerate(spec.phases):
            sp = SPECIES[p.species]
            energy_per_atom = p.energy / sp["energy_divisor"]
            phase_vars += (
                f"variable    phase_{i}_ml equal {p.fluence_ml}\n"
                f"variable    phase_{i}_energy equal {energy_per_atom}\n"
            )
            if p.flux_ratio > 0:
                phase_vars += _radical_config_block(
                    flux_ratio=p.flux_ratio,
                    radical_energy=p.radical_energy,
                    radical_temperature=p.radical_temperature,
                    radical_angle=p.radical_angle,
                    radical_angle_distribution=p.radical_angle_distribution,
                    max_inter_neutral_time=p.max_inter_neutral_time,
                    radical_i_above=p.radical_i_above,
                    inter_neutral_time=spec.inter_neutral_time,
                    prefix=f"phase_{i}_",
                )
            else:
                phase_vars += f"variable    phase_{i}_flux_ratio equal 0\n"

        phase_summary = ", ".join(
            f"{p.species}@{p.energy}eV×{p.fluence_ml}ML"
            + (f"+O•R{p.flux_ratio}" if p.flux_ratio > 0 else "")
            for p in spec.phases
        )
        return (
            common
            + f"\n# Cycle sub-mode\n"
            + f"# phases: {phase_summary}\n"
            + f"# {spec.cycles} cycle(s) × {total_phase_ml} ML/cycle = {total_fluence_ml} ML total\n"
            + f"\n"
            + f"variable    cycles equal {spec.cycles}\n"
            + f"variable    end_fluence equal {total_fluence_ml}\n"
            + f"\n"
            + f"variable    ion_i_above equal 6.0\n"
            + f"variable    radical_i_above equal {spec.radical_i_above}\n"
            + f"\n"
            + f"# Per-phase parameters\n"
            + phase_vars
        ) + _step_config_block(spec) + _mask_config_block(spec)

    elif spec.ion_mix is not None:
        # ── Multi-ion sub-mode ────────────────────────────────────────────────
        total = sum(c.fraction for c in spec.ion_mix)
        ion_summary = ", ".join(
            f"{c.species}@{c.energy}eV×{c.fraction/total:.0%}" for c in spec.ion_mix
        )
        cfg = (
            common
            + f"\n# Multi-ion sub-mode: {ion_summary}\n"
            + f"\n"
            + f"variable    end_fluence equal {spec.fluence}\n"
            + f"variable    i_above equal 6.0\n"
        )
        if spec.flux_ratio > 0:
            cfg += (
                f"\n# Multi-ion RIE: O• radical pre-exposure\n"
                + _radical_config_block(
                    flux_ratio=spec.flux_ratio,
                    radical_energy=spec.radical_energy,
                    radical_temperature=spec.radical_temperature,
                    radical_angle=spec.radical_angle,
                    radical_angle_distribution=spec.radical_angle_distribution,
                    max_inter_neutral_time=spec.max_inter_neutral_time,
                    radical_i_above=spec.radical_i_above,
                    inter_neutral_time=spec.inter_neutral_time,
                    burst_correction=spec.radical_burst,
                )
            )
        return cfg + _step_config_block(spec) + _mask_config_block(spec)

    else:
        # ── Single-species sub-mode ───────────────────────────────────────────
        species = SPECIES[spec.species]
        energy_per_atom = spec.energy / species["energy_divisor"]
        cfg = (
            common
            + f"\n# Single-species sub-mode: {spec.species} @ {spec.energy} eV\n"
            + f"\n"
            + f"variable    end_fluence equal {spec.fluence}\n"
            + f"variable    energ equal {energy_per_atom}\n"
            + f"variable    i_above equal {species['i_above']}\n"
            + f"variable    M_incident equal ${{{species['mass_var']}}}\n"
            + f"variable    incident_type_index equal {species['type_index']}\n"
        )
        if spec.flux_ratio > 0:
            cfg += (
                f"\n# RIE: O• radical pre-exposure\n"
                + _radical_config_block(
                    flux_ratio=spec.flux_ratio,
                    radical_energy=spec.radical_energy,
                    radical_temperature=spec.radical_temperature,
                    radical_angle=spec.radical_angle,
                    radical_angle_distribution=spec.radical_angle_distribution,
                    max_inter_neutral_time=spec.max_inter_neutral_time,
                    radical_i_above=spec.radical_i_above,
                    inter_neutral_time=spec.inter_neutral_time,
                    burst_correction=spec.radical_burst,
                )
            )
        return cfg + _step_config_block(spec) + _mask_config_block(spec)
