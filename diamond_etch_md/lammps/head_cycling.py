"""
lammps/head_cycling.py — generator for head.lmp in cycle-etch (multi-phase) simulations.

Cycling simulations alternate between N ion species within a single LAMMPS run.
Each phase has its own ion species, energy, ML count, and optional O• radical flux.

Phase selection uses index arithmetic on the global impact counter `c`:
  idx_in_cycle = c % impacts_per_cycle
Phase boundaries (cumulative ML thresholds) are defined as LAMMPS variables and
phase is selected using reverse-order if-blocks (last phase is the default; earlier
phases override it when idx_in_cycle is below their cumulative threshold).

If any phase uses Ar, the hybrid ReaxFF+ZBL potential is used with 4 atom types.
Otherwise, plain ReaxFF with 3 atom types is used (faster).

ncarbon.txt format: "c cn ncarbon nhydrogen noxygen"
  - After each O• radical: col2=cn (which radical in the sequence, 1-indexed)
  - After each ion impact: col2=0
This format enables mid-radical-loop restarts via the neut_complete LAMMPS variable.

Note: Boltzmann/cosine stochastic radical sampling for cycling phases is tracked
in TODO.md — the cycling neutral loop currently uses fixed-energy / fixed-angle mode.
"""

from ..orientations import ORIENT
from ..species import SPECIES
from ..spec import SimSpec, CyclePhase


def _has_ar(spec: SimSpec) -> bool:
    return any(SPECIES[p.species]["needs_zbl"] for p in spec.phases)


def _can_switch_potential(spec: SimSpec) -> bool:
    """True when the cycle mixes Ar phases with non-Ar phases.

    In that case the ZBL potential is only needed during Ar phases.  Since
    Ar is always removed after impact (remove_after_impact=True), it is safe
    to drop to plain ReaxFF for non-Ar phases, which is faster.
    """
    has_zbl = any(SPECIES[p.species]["needs_zbl"] for p in spec.phases)
    has_plain = any(not SPECIES[p.species]["needs_zbl"] for p in spec.phases)
    return has_zbl and has_plain


def _potential_switch_block() -> str:
    """LAMMPS snippet that switches pair potential at phase boundaries.

    Emitted once per outer-loop iteration, after phase-selection variables are
    set.  prev_needs_zbl tracks the potential that is currently active; when
    current_needs_zbl differs, the switch commands are executed.
    """
    return (
        f"# Switch qeq group when moving between Ar and non-Ar phases\n"
        f"# (pair_style stays as hybrid reaxff+ZBL throughout — switching pair_style\n"
        f"#  mid-cycle causes 'pair coeffs not set' errors after cold restarts)\n"
        f'if "${{prev_needs_zbl}} == 1 && ${{current_needs_zbl}} == 0" then &\n'
        f'"unfix reax_qeq" &\n'
        f'"fix reax_qeq all qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        f'if "${{prev_needs_zbl}} == 0 && ${{current_needs_zbl}} == 1" then &\n'
        f'"unfix reax_qeq" &\n'
        f'"pair_style hybrid reaxff NULL mincap 200 safezone 1.5 zbl 5.0 6.0" &\n'
        f'"pair_coeff * * reaxff ffield.reax C H O NULL" &\n'
        f'"pair_coeff 1 4 zbl 6.0 18.0" &\n'
        f'"pair_coeff 2 4 zbl 1.0 18.0" &\n'
        f'"pair_coeff 3 4 zbl 8.0 18.0" &\n'
        f'"pair_coeff 4 4 zbl 18.0 18.0" &\n'
        f'"group nonargon type 1 2 3" &\n'
        f'"fix reax_qeq nonargon qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        f"variable prev_needs_zbl equal ${{current_needs_zbl}}\n"
        f"\n"
    )


def _has_o2(spec: SimSpec) -> bool:
    return any(SPECIES[p.species]["is_molecule"] for p in spec.phases)


def _potential_block(has_ar: bool) -> str:
    if has_ar:
        return (
            f'if "${{pot}} == REAX" then &\n'
            f'"pair_style  hybrid reaxff NULL mincap 200 safezone 1.5 zbl 5.0 6.0" &\n'
            f'"pair_coeff * * reaxff ffield.reax C H O NULL" &\n'
            f'"pair_coeff 1 4 zbl 6.0 18.0" &\n'
            f'"pair_coeff 2 4 zbl 1.0 18.0" &\n'
            f'"pair_coeff 3 4 zbl 8.0 18.0" &\n'
            f'"pair_coeff 4 4 zbl 18.0 18.0" &\n'
            f'"group nonargon type 1 2 3" &\n'
            f'"fix reax_qeq nonargon qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        )
    else:
        return (
            f'if "${{pot}} == REAX" then &\n'
            f'"pair_style reaxff NULL mincap 200 safezone 1.5" &\n'
            f'"pair_coeff * * ffield.reax C H O" &\n'
            f'"fix reax_qeq all qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        )


def _phase_boundary_vars(phases: list) -> str:
    """Generate LAMMPS variables for cumulative impact thresholds per phase."""
    lines = []
    for i in range(len(phases)):
        if i == 0:
            lines.append(
                f"variable    phase_0_end equal $(v_ML*v_phase_0_ml)\n"
            )
        else:
            lines.append(
                f"variable    phase_{i}_end equal "
                f"$(v_phase_{i-1}_end+v_ML*v_phase_{i}_ml)\n"
            )
    lines.append(
        f"variable    impacts_per_cycle equal ${{phase_{len(phases)-1}_end}}\n"
    )
    return "".join(lines)


def _phase_radical_vars(p: CyclePhase, i: int) -> tuple:
    """Return (rad_angl_val, inter_neutral_val) strings for phase i.

    Returns LAMMPS variable references when phase has radicals, else defaults.
    radical_i_above is global (same injection height for all phases).
    """
    if p.flux_ratio > 0:
        return (
            f"${{phase_{i}_rad_angl}}",
            f"${{phase_{i}_inter_neutral_time}}",
        )
    else:
        return ("0.0", "1500.0")


def _phase_selection_block(phases: list, has_ar: bool,
                           switch_potential: bool = False) -> str:
    """
    Generate LAMMPS if-blocks that set current-phase properties at runtime.

    Approach: default to last phase, then reverse-order if-blocks for earlier
    phases.  Each earlier phase overrides when idx_in_cycle is below its
    cumulative end threshold.

    Sets current_rad_angl, current_inter_neutral_time
    in addition to the original current_* ion variables.
    radical_i_above is a global variable (same injection height for all phases).
    """
    N = len(phases)
    lines = []

    # Default: last phase
    last = phases[-1]
    last_sp = SPECIES[last.species]
    rad_angl_v, inter_t_v = _phase_radical_vars(last, N - 1)

    lines.append(f"# Phase selection (default: phase {N-1} = {last.species})\n")
    lines.append(f"variable current_ion_type equal {last_sp['type_index']}\n")
    lines.append(f"variable current_use_molecule equal {1 if last_sp['is_molecule'] else 0}\n")
    lines.append(f"variable current_ion_energy equal ${{phase_{N-1}_energy}}\n")
    lines.append(f"variable current_M_ion equal ${{{last_sp['mass_var']}}}\n")
    lines.append(f"variable current_flux_ratio equal ${{phase_{N-1}_flux_ratio}}\n")
    lines.append(f"variable current_radical_energy equal ${{phase_{N-1}_radical_energy}}\n")
    lines.append(f"variable current_rad_angl equal {rad_angl_v}\n")
    lines.append(f"variable current_inter_neutral_time equal {inter_t_v}\n")
    if has_ar:
        lines.append(
            f"variable current_needs_removal equal "
            f"{1 if last_sp['remove_after_impact'] else 0}\n"
        )
    if switch_potential:
        lines.append(
            f"variable current_needs_zbl equal "
            f"{1 if last_sp['needs_zbl'] else 0}\n"
        )
    lines.append("\n")

    # Reverse-order overrides for phases 0..N-2
    for i in range(N - 2, -1, -1):
        p = phases[i]
        sp = SPECIES[p.species]
        rad_angl_v, inter_t_v = _phase_radical_vars(p, i)
        removal_line = (
            f' &\n"variable current_needs_removal equal '
            f'{1 if sp["remove_after_impact"] else 0}"'
        ) if has_ar else ""
        zbl_line = (
            f' &\n"variable current_needs_zbl equal '
            f'{1 if sp["needs_zbl"] else 0}"'
        ) if switch_potential else ""
        lines.append(
            f'if "${{idx_in_cycle}} < ${{phase_{i}_end}}" then &\n'
            f'"variable current_ion_type equal {sp["type_index"]}" &\n'
            f'"variable current_use_molecule equal {1 if sp["is_molecule"] else 0}" &\n'
            f'"variable current_ion_energy equal ${{phase_{i}_energy}}" &\n'
            f'"variable current_M_ion equal ${{{sp["mass_var"]}}}" &\n'
            f'"variable current_flux_ratio equal ${{phase_{i}_flux_ratio}}" &\n'
            f'"variable current_radical_energy equal ${{phase_{i}_radical_energy}}" &\n'
            f'"variable current_rad_angl equal {rad_angl_v}" &\n'
            f'"variable current_inter_neutral_time equal {inter_t_v}"'
            f"{removal_line}"
            f"{zbl_line}\n"
            f"\n"
        )

    return "".join(lines)


def _build_cycle_ion_dump_blocks(spec: SimSpec, is_carbon_etch: bool = False):
    """Return (ion_dump_open, etch_event_block, channeling_block, ion_dump_close)
    for the cycling ion impact section, based on dump_mode."""
    dm = spec.dump_mode
    channeling_region_cmd = (
        f"region channelled block INF INF INF INF INF ${{channeling_z}} units box"
        if is_carbon_etch
        else f"region channelled block INF INF INF INF INF ${{bottom}} units lattice"
    )
    dump_file = f"etch_event_trajs/event_dump_ion${{c}}.dump"
    dump_cols  = "id type x y z vx vy vz fx fy fz q"

    if dm == "none":
        ion_dump_open  = ""
        ion_dump_close = ""
        etch_event_block = (
            f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"variable starting_nclusts equal $(c_nclusts)"\n'
        )
        channeling_block = (
            f'if "${{one_clust}} == 0" then &\n'
            f'"include sweep.lmp" &\n'
            f'"{channeling_region_cmd}" &\n'
            f'"group channelled_group region channelled" &\n'
            f'"variable n_channelled equal count(channelled_group)" &\n'
            f'"region channelled delete"\n'
            f"\n"
            f'if "${{n_channelled}} > 0" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"delete_atoms group channelled_group" &\n'
            f'"run 0" &\n'
            f'"group channelled_group delete" &\n'
            f'"variable n_channelled equal 0"\n'
        )

    elif dm == "all":
        ion_dump_open  = f"dump        current_dump_ion all custom 100 {dump_file} {dump_cols}\n"
        ion_dump_close = f"undump      current_dump_ion\n"
        etch_event_block = (
            f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"variable starting_nclusts equal $(c_nclusts)"\n'
        )
        channeling_block = (
            f'if "${{one_clust}} == 0" then &\n'
            f'"include sweep.lmp" &\n'
            f'"{channeling_region_cmd}" &\n'
            f'"group channelled_group region channelled" &\n'
            f'"variable n_channelled equal count(channelled_group)" &\n'
            f'"region channelled delete"\n'
            f"\n"
            f'if "${{n_channelled}} > 0" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"delete_atoms group channelled_group" &\n'
            f'"run 0" &\n'
            f'"group channelled_group delete" &\n'
            f'"variable n_channelled equal 0"\n'
        )

    else:  # etch_only
        ion_dump_open = (
            f"variable    keep_dump equal 0\n"
            f"dump        current_dump_ion all custom 100 {dump_file} {dump_cols}\n"
        )
        ion_dump_close = (
            f'if "${{keep_dump}} == 0" then "shell rm {dump_file}"\n'
            f"undump      current_dump_ion\n"
        )
        etch_event_block = (
            f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"variable starting_nclusts equal $(c_nclusts)" &\n'
            f'"variable keep_dump equal 1"\n'
        )
        channeling_block = (
            f'if "${{one_clust}} == 0" then &\n'
            f'"include sweep.lmp" &\n'
            f'"{channeling_region_cmd}" &\n'
            f'"group channelled_group region channelled" &\n'
            f'"variable n_channelled equal count(channelled_group)" &\n'
            f'"region channelled delete"\n'
            f"\n"
            f'if "${{n_channelled}} > 0" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"delete_atoms group channelled_group" &\n'
            f'"run 0" &\n'
            f'"group channelled_group delete" &\n'
            f'"variable n_channelled equal 0" &\n'
            f'"variable keep_dump equal 1"\n'
        )

    return ion_dump_open, etch_event_block, channeling_block, ion_dump_close


def get_head_lmp_cycle_etch(spec: SimSpec) -> str:
    """Generate the contents of head.lmp for a cycle-etch SimSpec."""
    cfg = ORIENT[spec.orientation]
    lattice_cmd = cfg["lattice_cmd"]
    bottom_expr = cfg["bottom_expr"]

    has_ar = _has_ar(spec)
    has_o2 = _has_o2(spec)
    switch_pot = _can_switch_potential(spec)
    has_any_radicals = any(p.flux_ratio > 0 for p in spec.phases)
    n_types = 4 if has_ar else 3
    dm = spec.dump_mode

    phase_names = " → ".join(
        f"{p.species}@{p.energy}eV×{p.fluence_ml}ML"
        + (f"+O•R{p.flux_ratio}" if p.flux_ratio > 0 else "")
        for p in spec.phases
    )

    molecule_decl = "molecule    O2 O2.molecule\n" if has_o2 else ""

    # nonargon group refresh: only needed when Ar is present
    nonargon_refresh = "group nonargon type 1 2 3\n" if has_ar else ""

    masses = (
        f"mass        1 ${{M_C}}\n"
        f"mass        2 ${{M_H}}\n"
        f"mass        3 ${{M_O}}\n"
        + (f"mass        4 ${{M_Ar}}\n" if has_ar else "")
    )

    ar_removal_block = (
        f'\nif "${{current_needs_removal}} == 1" then &\n'
        f'"group       ArRemove type 4" &\n'
        f'"delete_atoms group ArRemove" &\n'
        f'"group       ArRemove delete"\n'
    ) if has_ar else ""

    ion_dump_open, etch_event_block, channeling_block, ion_dump_close = (
        _build_cycle_ion_dump_blocks(spec)
    )

    # Neutral dump: only created in "all" mode, renamed to event_dump_n${c}_${cn}.dump
    neutral_dump_open  = (
        f"dump        current_dump_n all custom 100 "
        f"etch_event_trajs/event_dump_n${{c}}_${{cn}}.dump "
        f"id type x y z vx vy vz fx fy fz q\n"
    ) if dm == "all" else ""
    neutral_dump_close = "undump      current_dump_n\n" if dm == "all" else ""

    return (
        f"# head.lmp — generated by DiamondEtchMD (cycling mode)\n"
        f"# orientation={spec.orientation}  phases: {phase_names}\n"
        f"# {spec.cycles} cycle(s)  T={spec.surface_temperature}K  ion_angle={spec.ion_angle}deg\n"
        f"package     kokkos neigh/qeq full neigh half newton on\n"
        f"units       real\n"
        f"include     config.lmp\n"
        f'if "${{pot}} == COMB3" then &\n'
        f'"atom_style charge" &\n'
        f'elif "${{pot}} == REAX" &\n'
        f'"atom_style charge" &\n'
        f"else &\n"
        f'"atom_style atomic"\n'
        f"\n"
        f"boundary    p p m\n"
        f"\n"
        f"region      dummy_region block 0 1 0 1 0 1\n"
        f"create_box  {n_types} dummy_region\n"
        f"read_data   ${{data_file}} add merge\n"
        f"\n"
        f"# Phase boundary variables (cumulative impacts per cycle)\n"
        f"{_phase_boundary_vars(spec.phases)}"
        f"\n"
        f"variable    lp equal $(v_end_fluence*v_ML)-${{n_complete}} # impacts left\n"
        f"variable    a loop ${{lp}}\n"
        f"variable    bottom equal {bottom_expr}\n"
        f"print       'BOTTOM ${{bottom}}'\n"
        f"print       'N_LAT_0 ${{n_lat_0}}'\n"
        f"\n"
        f"# Regions\n"
        f"{lattice_cmd}\n"
        f"region      bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        f"variable    sublat equal ${{bottom}}+1/2\n"
        f"region      anchor block INF INF INF INF ${{bottom}} ${{sublat}} units lattice\n"
        f"\n"
        f"# Groups\n"
        f"group       anchor region anchor\n"
        f"group       insert empty\n"
        f"group       mobile subtract all anchor\n"
        f"group       carbon type 1\n"
        f"\n"
        f"# Masses\n"
        f"{masses}"
        f"\n"
        f"# Potential\n"
        f"{_potential_block(has_ar)}"
        + (  # track which potential is active for runtime switching
            f"variable    prev_needs_zbl equal 1\n"
            if switch_pot else ""
        ) +
        f"\n"
        f"# Restart state\n"
        f"variable    c equal ${{n_complete}}\n"
        f"variable    cn_start equal ${{neut_complete}}\n"
        f"variable    cn equal 0\n"
        f"variable    event_count equal ${{n_events}}\n"
        f"\n"
        f"# Atom count diagnostics\n"
        f"variable    nfixed  equal count(anchor)\n"
        f"variable    nmobile equal count(mobile)\n"
        f"variable    ninject equal count(insert)\n"
        f"\n"
        f"# Energy conservation check\n"
        f"compute     ake all ke\n"
        f"compute     ape all pe\n"
        f"compute     ike insert ke\n"
        f"variable    ate equal c_ake+c_ape+ecouple\n"
        f"\n"
        f"# Cluster detection\n"
        f"variable    checkevery equal 1000\n"
        f"compute     clusts all cluster/atom 3.0\n"
        f"compute     clust_min all reduce min c_clusts\n"
        f"compute     clust_max all reduce max c_clusts\n"
        f'variable    one_clust equal "c_clust_min == c_clust_max"\n'
        f"fix         stopclust all halt ${{checkevery}} v_one_clust == 0 error continue message yes\n"
        f"\n"
        f"# Carbon cluster event tracking\n"
        f"compute     cclusts carbon cluster/atom 1.8\n"
        f"compute     cc1 carbon chunk/atom c_cclusts compress yes\n"
        f"compute     nclusts carbon reduce max c_cc1\n"
        f"\n"
        f"# Thermo\n"
        f"thermo_style    custom step time v_ninject temp c_ike dt ecouple v_ate v_one_clust c_nclusts\n"
        f"compute     mtemp mobile temp\n"
        f"compute_modify  mtemp dynamic/dof yes\n"
        f"thermo_modify   temp mtemp\n"
        f"thermo_modify   lost warn flush yes\n"
        f"\n"
        f'print       "${{c}}" file begin.txt\n'
        f"{molecule_decl}"
        f"\n"
        # ======================================================
        f"# ========================= Begin Per-Impact Outer Loop =========================\n"
        f"label       loop\n"
        f"{nonargon_refresh}"
        f'if "${{cn_start}} > 0" then "variable cn equal ${{cn_start}}" else "variable cn equal 0"\n'
        f"\n"
        f"# Compute position within current cycle\n"
        f"variable    cycle_idx equal floor(v_c/v_impacts_per_cycle)\n"
        f"variable    idx_in_cycle equal $(v_c-v_cycle_idx*v_impacts_per_cycle)\n"
        f"\n"
        f"{_phase_selection_block(spec.phases, has_ar, switch_pot)}"
        f"# Ion velocities for current phase\n"
        f"variable    vel_ion equal sqrt(2*${{current_ion_energy}}*6.02214129*1.0e+7/${{current_M_ion}}/6242)/1000\n"
        f"variable    velz_ion equal cos(${{ion_angl}}*PI/180)*${{vel_ion}}\n"
        f"variable    vely_ion equal sin(${{ion_angl}}*PI/180)*${{vel_ion}}\n"
        f"\n"
        + (  # potential switch block — only when mixing Ar and non-Ar phases
            _potential_switch_block() if switch_pot else ""
        ) +
        f"# Adaptive timestep (used for both neutral and ion loops)\n"
        f"fix         ats all dt/reset 1 NULL 1 0.01 units box\n"
        f"\n"
        + (  # neutral deposition loop — omitted entirely when no phase has radicals
            f"# ========================= Begin neutral deposition loop =========================\n"
            f'if "${{current_flux_ratio}} == 0" then "jump SELF skip_chem"\n'
            f'if "${{cn_start}} > 0 && ${{cn_start}} < ${{current_flux_ratio}}" then &\n'
            f'"variable neutral_lp loop $(v_current_flux_ratio-v_cn_start)" &\n'
            f'elif "${{cn_start}} == ${{current_flux_ratio}}" &\n'
            f'"jump SELF skip_chem" &\n'
            f"else &\n"
            f'"variable neutral_lp loop ${{current_flux_ratio}}"\n'
            f"\n"
            f"label       neutral_loop\n"
            f"variable    cn equal ${{cn}}+1\n"
            f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{cn}}))\n"
            f"group       insert clear\n"
            f"group       mobile subtract all anchor\n"
            f"{nonargon_refresh}"
            f"\n"
            f"# Radical velocity (fixed-energy, per-phase angle)\n"
            f"variable    vel_chem equal sqrt(2*${{current_radical_energy}}*6.02214129*1.0e+7/${{M_O}}/6242)/1000\n"
            f"variable    velz_chem equal cos(${{current_rad_angl}}*PI/180)*${{vel_chem}}\n"
            f"variable    vely_chem equal sin(${{current_rad_angl}}*PI/180)*${{vel_chem}}\n"
            f"\n"
            f"# Deposit O• radical (always type 3)\n"
            f"fix         depo insert deposit 1 3 1 ${{deposeed}} global "
            f"${{radical_i_above}} ${{radical_i_above}} "
            f"vx 0.0 0.0 vy ${{vely_chem}} ${{vely_chem}} vz -${{velz_chem}} -${{velz_chem}} "
            f"region bbox units box\n"
            f"fix         2 mobile nve\n"
            f"fix         3 insert nve\n"
            f"{neutral_dump_open}"
            f"\n"
            f"timestep    1e-10\n"
            f"run         1 post no\n"
            f"{nonargon_refresh}"
            f"run         0\n"
            f"variable    starting_nclusts equal $(c_nclusts)\n"
            f"variable    t0 equal $(time)\n"
            f"variable    time_elapsed equal time-${{t0}}\n"
            f"fix         thalt all halt 1 v_time_elapsed > ${{current_inter_neutral_time}} "
            f"error continue message yes\n"
            f"\n"
            f"# ======================== Neutral inner loop ========================\n"
            f"label       continue_n_impact\n"
            f"run         500 pre no post no\n"
            f"run         0\n"
            f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"variable starting_nclusts equal $(c_nclusts)"\n'
            f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
            f'if "$(time-v_t0) < ${{current_inter_neutral_time}}" then "jump SELF continue_n_impact"\n'
            f"\n"
            f"group       carbon type 1\n"
            f"group       hydrogen type 2\n"
            f"group       oxygen type 3\n"
            f"variable    ncarbon equal count(carbon)\n"
            f"variable    nhydrogen equal count(hydrogen)\n"
            f"variable    noxygen equal count(oxygen)\n"
            f"\n"
            f'print       "Neutral run ${{cn}} complete"\n'
            f'print       "C_COUNT_neutral: ${{ncarbon}}"\n'
            f'print       "${{c}} ${{cn}} ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
            f"write_data  impact_snaps/${{c}}_${{cn}}.data nofix nocoeff\n"
            f"\n"
            f"unfix       thalt\n"
            f"{neutral_dump_close}"
            f"unfix       depo\n"
            f"# Thermalize after each radical\n"
            f"include     thermalize.lmp\n"
            f"unfix       2\n"
            f"unfix       3\n"
            f"# ======================== End neutral inner loop ========================\n"
            f"next        neutral_lp\n"
            f"jump        SELF neutral_loop\n"
            f"\n"
            f"label       skip_chem\n"
            f"variable    cn_start equal 0\n"
            f"variable    cn equal 0\n"
            f"# ========================= End neutral deposition loop =========================\n"
            f"\n"
            if has_any_radicals else ""
        ) +
        f"# Final thermalize before ion impact\n"
        f"include     thermalize.lmp\n"
        f"\n"
        f"# Increment ion impact counter\n"
        f"variable    c equal ${{c}}+1\n"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{c}}))\n"
        f"\n"
        f"{ion_dump_open}"
        f"\n"
        f"# Deposit ion (O2 via molecule file, all others as single atom)\n"
        f'if "${{current_use_molecule}} == 1" then &\n'
        f'"fix depo insert deposit 1 0 1 ${{deposeed}} global '
        f'${{ion_i_above}} ${{ion_i_above}} vx 0.0 0.0 vy ${{vely_ion}} ${{vely_ion}} '
        f'vz -${{velz_ion}} -${{velz_ion}} region bbox units box mol O2" &\n'
        f"else &\n"
        f'"fix depo insert deposit 1 ${{current_ion_type}} 1 ${{deposeed}} global '
        f'${{ion_i_above}} ${{ion_i_above}} vx 0.0 0.0 vy ${{vely_ion}} ${{vely_ion}} '
        f'vz -${{velz_ion}} -${{velz_ion}} region bbox units box"\n'
        f"fix         2 mobile nve\n"
        f"fix         3 insert nve\n"
        f"\n"
        f"timestep    1e-10\n"
        f"run         1 post no\n"
        f"{nonargon_refresh}"
        f"run         0\n"
        f"variable    starting_nclusts equal $(c_nclusts)\n"
        f"thermo      100\n"
        f"variable    t0 equal $(time)\n"
        f"variable    time_elapsed equal time-${{t0}}\n"
        f"fix         thalt all halt 1 v_time_elapsed > ${{impact_time}} error continue message yes\n"
        f"run         0\n"
        f"variable    n_channelled equal 0\n"
        f"\n"
        f"# =========================== Ion inner loop ===========================\n"
        f"label       continue_impact\n"
        f"run         1000000000 pre no post no\n"
        f"run         0\n"
        f"variable    n_channelled equal 0\n"
        f"{etch_event_block}"
        f"{channeling_block}"
        f"\n"
        f'if "$(time-v_t0) < ${{impact_time}}" then "jump SELF continue_impact"\n'
        f"unfix       thalt\n"
        f"# =========================== End ion inner loop ===========================\n"
        f"\n"
        f"unfix       depo\n"
        f"unfix       ats\n"
        f"{ar_removal_block}"
        f"\n"
        f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
        f"\n"
        f"# Thermalize\n"
        f"include     thermalize.lmp\n"
        f"\n"
        f"unfix       2\n"
        f"unfix       3\n"
        f"\n"
        f"# Replenish carbon if etched below baseline\n"
        f"group       carbon type 1\n"
        f"group       hydrogen type 2\n"
        f"group       oxygen type 3\n"
        f"variable    ncarbon equal count(carbon)\n"
        f"variable    nhydrogen equal count(hydrogen)\n"
        f"variable    noxygen equal count(oxygen)\n"
        f"label       addfix\n"
        f'if "${{ncarbon}}<${{n_lat_0}}" then &\n'
        f'"include addfix.lmp" &\n'
        f'"fix 2 mobile nve" &\n'
        f'"include thermalize.lmp" &\n'
        f'"unfix 2"\n'
        f'if "${{ncarbon}}<${{n_lat_0}}" then "jump SELF addfix"\n'
        f"\n"
        f'print       "Run ${{c}} complete"\n'
        f'print       "C_COUNT: ${{ncarbon}}"\n'
        f'print       "${{c}} 0 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        f"\n"
        f"write_data  impact_snaps/${{c}}_0.data nofix nocoeff\n"
        f"if '$(v_c%v_ML) == 0' then "
        f"\"write_dump all custom ML_impacts.dump id type x y z vx vy vz q modify sort id append yes\"\n"
        f"\n"
        f"{ion_dump_close}"
        f"# ========================= End Per-Impact Outer Loop =========================\n"
        f"next        a\n"
        f"jump        SELF loop\n"
    )


def get_head_lmp_carbon_etch_cycle(spec: SimSpec) -> str:
    """Generate head.lmp for carbon-etch + cycle-etch mode.

    Differences vs get_head_lmp_cycle_etch:
    - read_data (patched 4-type file) instead of create_box + read_data add merge
    - anchor defined by anchor_z_max (Å, box units) not lattice units
    - thin-slab stop condition at top of each loop iteration
    - no addfix replenishment block
    """
    from .head import (_carbon_etch_slab_check, _carbon_etch_end_label,
                       _carbon_etch_initial_therm_block)

    has_ar = _has_ar(spec)
    has_o2 = _has_o2(spec)
    switch_pot = _can_switch_potential(spec)
    has_any_radicals = any(p.flux_ratio > 0 for p in spec.phases)
    dm = spec.dump_mode

    phase_names = " → ".join(
        f"{p.species}@{p.energy}eV×{p.fluence_ml}ML"
        + (f"+O•R{p.flux_ratio}" if p.flux_ratio > 0 else "")
        for p in spec.phases
    )

    molecule_decl    = "molecule    O2 O2.molecule\n" if has_o2 else ""
    nonargon_refresh = "group nonargon type 1 2 3\n" if has_ar else ""

    masses = (
        f"mass        1 ${{M_C}}\n"
        f"mass        2 ${{M_H}}\n"
        f"mass        3 ${{M_O}}\n"
        + (f"mass        4 ${{M_Ar}}\n" if has_ar else "")
    )

    ar_removal_block = (
        f'\nif "${{current_needs_removal}} == 1" then &\n'
        f'"group       ArRemove type 4" &\n'
        f'"delete_atoms group ArRemove" &\n'
        f'"group       ArRemove delete"\n'
    ) if has_ar and spec.remove_ar else ""

    ion_dump_open, etch_event_block, channeling_block, ion_dump_close = (
        _build_cycle_ion_dump_blocks(spec, is_carbon_etch=True)
    )

    neutral_dump_open  = (
        f"dump        current_dump_n all custom 100 "
        f"etch_event_trajs/event_dump_n${{c}}_${{cn}}.dump "
        f"id type x y z vx vy vz fx fy fz q\n"
    ) if dm == "all" else ""
    neutral_dump_close = "undump      current_dump_n\n" if dm == "all" else ""

    return (
        f"# head.lmp — generated by DiamondEtchMD (carbon-etch cycling)\n"
        f"# config_file={spec.initial_config_file}  phases: {phase_names}\n"
        f"# {spec.cycles} cycle(s)  T={spec.surface_temperature}K  ion_angle={spec.ion_angle}deg\n"
        f"package     kokkos neigh/qeq full neigh half newton on\n"
        f"units       real\n"
        f"include     config.lmp\n"
        f'if "${{pot}} == COMB3" then &\n'
        f'"atom_style charge" &\n'
        f'elif "${{pot}} == REAX" &\n'
        f'"atom_style charge" &\n'
        f"else &\n"
        f'"atom_style atomic"\n'
        f"\n"
        f"boundary    p p m\n"
        f"\n"
        f"# Load user-supplied config (patched to 4 atom types by builder)\n"
        f"read_data   ${{data_file}}\n"
        f"\n"
        f"# Phase boundary variables\n"
        f"{_phase_boundary_vars(spec.phases)}"
        f"\n"
        f"variable    lp equal $(v_end_fluence*v_ML)-${{n_complete}}\n"
        f"variable    a loop ${{lp}}\n"
        f"\n"
        f"# Regions\n"
        f"region      bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        f"region      anchor block INF INF INF INF INF ${{anchor_z_max}} units box\n"
        f"\n"
        f"# Groups\n"
        f"group       anchor region anchor\n"
        f"group       insert empty\n"
        f"group       mobile subtract all anchor\n"
        f"group       carbon type 1\n"
        f"\n"
        f"# Thin-slab threshold\n"
        f"variable    n_anchor_2x equal 2*count(anchor)\n"
        f"# Channeling threshold: 2 Å below the lowest anchor atom (avoids false positives)\n"
        f"variable    channeling_z equal bound(anchor,zmin)-2.0\n"
        f"\n"
        f"# Masses\n"
        f"{masses}"
        f"\n"
        f"# Potential\n"
        f"{_potential_block(has_ar)}"
        + (
            f"variable    prev_needs_zbl equal 1\n"
            if switch_pot else ""
        ) +
        f"\n"
        f"# Restart state\n"
        f"variable    c equal ${{n_complete}}\n"
        f"variable    cn_start equal ${{neut_complete}}\n"
        f"variable    cn equal 0\n"
        f"variable    event_count equal ${{n_events}}\n"
        f"\n"
        f"variable    nfixed  equal count(anchor)\n"
        f"variable    nmobile equal count(mobile)\n"
        f"variable    ninject equal count(insert)\n"
        f"\n"
        f"compute     ake all ke\n"
        f"compute     ape all pe\n"
        f"compute     ike insert ke\n"
        f"variable    ate equal c_ake+c_ape+ecouple\n"
        f"\n"
        f"variable    checkevery equal 1000\n"
        f"compute     clusts all cluster/atom 3.0\n"
        f"compute     clust_min all reduce min c_clusts\n"
        f"compute     clust_max all reduce max c_clusts\n"
        f'variable    one_clust equal "c_clust_min == c_clust_max"\n'
        f"fix         stopclust all halt ${{checkevery}} v_one_clust == 0 error continue message yes\n"
        f"\n"
        f"compute     cclusts carbon cluster/atom 1.8\n"
        f"compute     cc1 carbon chunk/atom c_cclusts compress yes\n"
        f"compute     nclusts carbon reduce max c_cc1\n"
        f"\n"
        f"thermo_style    custom step time v_ninject temp c_ike dt ecouple v_ate v_one_clust c_nclusts\n"
        f"compute     mtemp mobile temp\n"
        f"compute_modify  mtemp dynamic/dof yes\n"
        f"thermo_modify   temp mtemp\n"
        f"thermo_modify   lost warn flush yes\n"
        f"\n"
        f'print       "${{c}}" file begin.txt\n'
        f"{molecule_decl}"
        f"{_carbon_etch_initial_therm_block(spec.initial_thermalization)}"
        f"\n"
        f"# ========================= Begin Per-Impact Outer Loop =========================\n"
        f"label       loop\n"
        f"{_carbon_etch_slab_check()}"
        f"{nonargon_refresh}"
        f'if "${{cn_start}} > 0" then "variable cn equal ${{cn_start}}" else "variable cn equal 0"\n'
        f"\n"
        f"variable    cycle_idx equal floor(v_c/v_impacts_per_cycle)\n"
        f"variable    idx_in_cycle equal $(v_c-v_cycle_idx*v_impacts_per_cycle)\n"
        f"\n"
        f"{_phase_selection_block(spec.phases, has_ar, switch_pot)}"
        f"variable    vel_ion equal sqrt(2*${{current_ion_energy}}*6.02214129*1.0e+7/${{current_M_ion}}/6242)/1000\n"
        f"variable    velz_ion equal cos(${{ion_angl}}*PI/180)*${{vel_ion}}\n"
        f"variable    vely_ion equal sin(${{ion_angl}}*PI/180)*${{vel_ion}}\n"
        f"\n"
        + (
            _potential_switch_block() if switch_pot else ""
        ) +
        f"fix         ats all dt/reset 1 NULL 1 0.01 units box\n"
        f"\n"
        + (
            f"# ========================= Begin neutral deposition loop =========================\n"
            f'if "${{current_flux_ratio}} == 0" then "jump SELF skip_chem"\n'
            f'if "${{cn_start}} > 0 && ${{cn_start}} < ${{current_flux_ratio}}" then &\n'
            f'"variable neutral_lp loop $(v_current_flux_ratio-v_cn_start)" &\n'
            f'elif "${{cn_start}} == ${{current_flux_ratio}}" &\n'
            f'"jump SELF skip_chem" &\n'
            f"else &\n"
            f'"variable neutral_lp loop ${{current_flux_ratio}}"\n'
            f"\n"
            f"label       neutral_loop\n"
            f"variable    cn equal ${{cn}}+1\n"
            f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{cn}}))\n"
            f"group       insert clear\n"
            f"group       mobile subtract all anchor\n"
            f"{nonargon_refresh}"
            f"\n"
            f"variable    vel_chem equal sqrt(2*${{current_radical_energy}}*6.02214129*1.0e+7/${{M_O}}/6242)/1000\n"
            f"variable    velz_chem equal cos(${{current_rad_angl}}*PI/180)*${{vel_chem}}\n"
            f"variable    vely_chem equal sin(${{current_rad_angl}}*PI/180)*${{vel_chem}}\n"
            f"\n"
            f"fix         depo insert deposit 1 3 1 ${{deposeed}} global "
            f"${{radical_i_above}} ${{radical_i_above}} "
            f"vx 0.0 0.0 vy ${{vely_chem}} ${{vely_chem}} vz -${{velz_chem}} -${{velz_chem}} "
            f"region bbox units box\n"
            f"fix         2 mobile nve\n"
            f"fix         3 insert nve\n"
            f"{neutral_dump_open}"
            f"\n"
            f"timestep    1e-10\n"
            f"run         1 post no\n"
            f"{nonargon_refresh}"
            f"run         0\n"
            f"variable    starting_nclusts equal $(c_nclusts)\n"
            f"variable    t0 equal $(time)\n"
            f"variable    time_elapsed equal time-${{t0}}\n"
            f"fix         thalt all halt 1 v_time_elapsed > ${{current_inter_neutral_time}} "
            f"error continue message yes\n"
            f"\n"
            f"label       continue_n_impact\n"
            f"run         500 pre no post no\n"
            f"run         0\n"
            f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"variable starting_nclusts equal $(c_nclusts)"\n'
            f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
            f'if "$(time-v_t0) < ${{current_inter_neutral_time}}" then "jump SELF continue_n_impact"\n'
            f"\n"
            f"group       carbon type 1\n"
            f"group       hydrogen type 2\n"
            f"group       oxygen type 3\n"
            f"variable    ncarbon equal count(carbon)\n"
            f"variable    nhydrogen equal count(hydrogen)\n"
            f"variable    noxygen equal count(oxygen)\n"
            f"\n"
            f'print       "Neutral run ${{cn}} complete"\n'
            f'print       "C_COUNT_neutral: ${{ncarbon}}"\n'
            f'print       "${{c}} ${{cn}} ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
            f"write_data  impact_snaps/${{c}}_${{cn}}.data nofix nocoeff\n"
            f"\n"
            f"unfix       thalt\n"
            f"{neutral_dump_close}"
            f"unfix       depo\n"
            f"include     thermalize.lmp\n"
            f"unfix       2\n"
            f"unfix       3\n"
            f"next        neutral_lp\n"
            f"jump        SELF neutral_loop\n"
            f"\n"
            f"label       skip_chem\n"
            f"variable    cn_start equal 0\n"
            f"variable    cn equal 0\n"
            f"# ========================= End neutral deposition loop =========================\n"
            f"\n"
            if has_any_radicals else ""
        ) +
        f"include     thermalize.lmp\n"
        f"\n"
        f"variable    c equal ${{c}}+1\n"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{c}}))\n"
        f"\n"
        f"{ion_dump_open}"
        f"\n"
        f'if "${{current_use_molecule}} == 1" then &\n'
        f'"fix depo insert deposit 1 0 1 ${{deposeed}} global '
        f'${{ion_i_above}} ${{ion_i_above}} vx 0.0 0.0 vy ${{vely_ion}} ${{vely_ion}} '
        f'vz -${{velz_ion}} -${{velz_ion}} region bbox units box mol O2" &\n'
        f"else &\n"
        f'"fix depo insert deposit 1 ${{current_ion_type}} 1 ${{deposeed}} global '
        f'${{ion_i_above}} ${{ion_i_above}} vx 0.0 0.0 vy ${{vely_ion}} ${{vely_ion}} '
        f'vz -${{velz_ion}} -${{velz_ion}} region bbox units box"\n'
        f"fix         2 mobile nve\n"
        f"fix         3 insert nve\n"
        f"\n"
        f"timestep    1e-10\n"
        f"run         1 post no\n"
        f"{nonargon_refresh}"
        f"run         0\n"
        f"variable    starting_nclusts equal $(c_nclusts)\n"
        f"thermo      100\n"
        f"variable    t0 equal $(time)\n"
        f"variable    time_elapsed equal time-${{t0}}\n"
        f"fix         thalt all halt 1 v_time_elapsed > ${{impact_time}} error continue message yes\n"
        f"run         0\n"
        f"variable    n_channelled equal 0\n"
        f"\n"
        f"label       continue_impact\n"
        f"run         1000000000 pre no post no\n"
        f"run         0\n"
        f"variable    n_channelled equal 0\n"
        f"{etch_event_block}"
        f"{channeling_block}"
        f"\n"
        f'if "$(time-v_t0) < ${{impact_time}}" then "jump SELF continue_impact"\n'
        f"unfix       thalt\n"
        f"\n"
        f"unfix       depo\n"
        f"unfix       ats\n"
        f"{ar_removal_block}"
        f"\n"
        f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
        f"\n"
        f"include     thermalize.lmp\n"
        f"\n"
        f"unfix       2\n"
        f"unfix       3\n"
        f"\n"
        f"# Atom counts (no replenishment in carbon-etch)\n"
        f"group       carbon type 1\n"
        f"group       hydrogen type 2\n"
        f"group       oxygen type 3\n"
        f"variable    ncarbon equal count(carbon)\n"
        f"variable    nhydrogen equal count(hydrogen)\n"
        f"variable    noxygen equal count(oxygen)\n"
        f"\n"
        f'print       "Run ${{c}} complete"\n'
        f'print       "C_COUNT: ${{ncarbon}}"\n'
        f'print       "${{c}} 0 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        f"\n"
        f"write_data  impact_snaps/${{c}}_0.data nofix nocoeff\n"
        f"if '$(v_c%v_ML) == 0' then "
        f"\"write_dump all custom ML_impacts.dump id type x y z vx vy vz q modify sort id append yes\"\n"
        f"\n"
        f"{ion_dump_close}"
        f"# ========================= End Per-Impact Outer Loop =========================\n"
        f"next        a\n"
        f"jump        SELF loop\n"
        f"{_carbon_etch_end_label()}"
    )
