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


def _phase_needs_zbl(p: CyclePhase) -> bool:
    """True when any ion in this phase requires ZBL (i.e. is a non-C/H/O element)."""
    if p.ion_mix is not None:
        return any(SPECIES[c.species]["needs_zbl"] for c in p.ion_mix)
    return SPECIES[p.species]["needs_zbl"]


def _phase_has_molecule(p: CyclePhase) -> bool:
    """True when any ion in this phase is a molecule (e.g. O2)."""
    if p.ion_mix is not None:
        return any(SPECIES[c.species]["is_molecule"] for c in p.ion_mix)
    return SPECIES[p.species]["is_molecule"]


def _phase_label(p: CyclePhase, i: int) -> str:
    """Short display label for phase i (used in head.lmp comments)."""
    suffix = f"+O•R{p.flux_ratio}" if p.flux_ratio > 0 else ""
    if p.ion_mix is not None:
        mix = "+".join(f"{c.species}({c.fraction:.0%})" for c in p.ion_mix)
        return f"[{mix}]×{p.fluence_ml}ML{suffix}"
    return f"{p.species}@{p.energy}eV×{p.fluence_ml}ML{suffix}"


def _has_ar(spec: SimSpec) -> bool:
    return any(_phase_needs_zbl(p) for p in spec.phases)


def _zbl_atomic_number(spec: SimSpec) -> float:
    """Return the atomic number of the first ZBL-needing species in spec.phases."""
    for p in spec.phases:
        if p.ion_mix is not None:
            for c in p.ion_mix:
                sp = SPECIES[c.species]
                if sp["needs_zbl"]:
                    return sp["atomic_number"]
        else:
            sp = SPECIES[p.species]
            if sp["needs_zbl"]:
                return sp["atomic_number"]
    return 18.0


def _can_switch_potential(spec: SimSpec) -> bool:
    """True when the cycle mixes Ar phases with non-Ar phases.

    In that case the ZBL potential is only needed during Ar phases.  Since
    Ar is always removed after impact (remove_after_impact=True), it is safe
    to drop to plain ReaxFF for non-Ar phases, which is faster.
    """
    has_zbl = any(_phase_needs_zbl(p) for p in spec.phases)
    has_plain = any(not _phase_needs_zbl(p) for p in spec.phases)
    return has_zbl and has_plain


def _potential_switch_block(spec: SimSpec) -> str:
    """LAMMPS snippet that switches pair potential at phase boundaries.

    Emitted once per outer-loop iteration, after phase-selection variables are
    set.  prev_needs_zbl tracks the potential that is currently active; when
    current_needs_zbl differs, the switch commands are executed.
    """
    z = _zbl_atomic_number(spec)
    return (
        f"# Switch pair potential and qeq group at Ar↔non-Ar phase boundaries.\n"
        f"# ZBL→plain: drop hybrid pair_style; type-4 (Ar, now absent) maps to C.\n"
        f'if "${{prev_needs_zbl}} == 1 && ${{current_needs_zbl}} == 0" then &\n'
        f'"unfix reax_qeq" &\n'
        f'"pair_style reaxff NULL" &\n'
        f'"pair_coeff * * ffield.reax C H O C" &\n'
        f'"fix reax_qeq all qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        f'if "${{prev_needs_zbl}} == 0 && ${{current_needs_zbl}} == 1" then &\n'
        f'"unfix reax_qeq" &\n'
        f'"pair_style hybrid reaxff NULL zbl 5.0 6.0" &\n'
        f'"pair_coeff * * reaxff ffield.reax C H O NULL" &\n'
        f'"pair_coeff 1 4 zbl 6.0 {z}" &\n'
        f'"pair_coeff 2 4 zbl 1.0 {z}" &\n'
        f'"pair_coeff 3 4 zbl 8.0 {z}" &\n'
        f'"pair_coeff 4 4 zbl {z} {z}" &\n'
        f'"fix reax_qeq nonargon qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        f"variable prev_needs_zbl equal ${{current_needs_zbl}}\n"
        f"\n"
    )


def _has_o2(spec: SimSpec) -> bool:
    return any(_phase_has_molecule(p) for p in spec.phases)


def _potential_block(spec: SimSpec) -> str:
    has_zbl = _has_ar(spec)
    if has_zbl:
        z = _zbl_atomic_number(spec)
        return (
            f'if "${{pot}} == REAX" then &\n'
            f'"pair_style  hybrid reaxff NULL zbl 5.0 6.0" &\n'
            f'"pair_coeff * * reaxff ffield.reax C H O NULL" &\n'
            f'"pair_coeff 1 4 zbl 6.0 {z}" &\n'
            f'"pair_coeff 2 4 zbl 1.0 {z}" &\n'
            f'"pair_coeff 3 4 zbl 8.0 {z}" &\n'
            f'"pair_coeff 4 4 zbl {z} {z}" &\n'
            f'"group nonargon type 1 2 3" &\n'
            f'"fix reax_qeq nonargon qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        )
    else:
        return (
            f'if "${{pot}} == REAX" then &\n'
            f'"pair_style reaxff NULL" &\n'
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
    """Return (rad_angl_val, inter_neutral_val, rad_azimuth_val) strings for phase i.

    Returns LAMMPS variable references when phase has radicals, else defaults.
    radical_i_above is global (same injection height for all phases).
    """
    if p.flux_ratio > 0:
        return (
            f"${{phase_{i}_rad_angl}}",
            f"${{phase_{i}_inter_neutral_time}}",
            f"${{phase_{i}_rad_azimuth}}",
        )
    else:
        return ("0.0", "1500.0", "90.0")


def _phase_selection_block(phases: list, has_ar: bool,
                           switch_potential: bool = False) -> str:
    """
    Generate LAMMPS if-blocks that set current-phase properties at runtime.

    Approach: default to last phase, then reverse-order if-blocks for earlier
    phases.  Each earlier phase overrides when idx_in_cycle is below its
    cumulative end threshold.

    For ion_mix phases: sets current_phase_mix_idx (1-indexed by mix-phase order)
    and current_phase_idx (absolute phase index) instead of current_ion_type /
    current_use_molecule / current_ion_energy / current_M_ion.  The mix dispatch
    block (_phase_ion_mix_dispatch_block) fills in those vars after this block runs.

    For radical_burst phases: sets current_phase_is_burst = 1 and current_phase_idx
    so the burst dispatch block can emit the right per-phase burst loop.

    Sets current_rad_angl, current_inter_neutral_time in addition to ion variables.
    radical_i_above is a global variable (same injection height for all phases).
    """
    N = len(phases)
    lines = []
    has_any_radicals = any(p.flux_ratio > 0 for p in phases)
    has_any_mix = any(p.ion_mix is not None for p in phases)
    has_any_burst = any(p.radical_burst for p in phases)

    # Assign 1-indexed mix dispatch slots to ion_mix phases
    mix_idx_of = {}  # phase_index -> mix dispatch index (1-based)
    mix_counter = 1
    for i, p in enumerate(phases):
        if p.ion_mix is not None:
            mix_idx_of[i] = mix_counter
            mix_counter += 1

    def _ion_defaults_for(p, i):
        """Direct-assignment lines for a non-mix phase (no quotes, for default block)."""
        lines_out = []
        if p.ion_mix is not None:
            lines_out.append(f"variable current_phase_mix_idx equal {mix_idx_of[i]}\n")
            lines_out.append(f"variable current_phase_idx equal {i}\n")
            if has_ar:
                mix_needs_removal = any(
                    SPECIES[c.species]["remove_after_impact"] for c in p.ion_mix
                )
                lines_out.append(f"variable current_needs_removal equal {1 if mix_needs_removal else 0}\n")
            if switch_potential:
                lines_out.append(f"variable current_needs_zbl equal {1 if _phase_needs_zbl(p) else 0}\n")
        else:
            sp = SPECIES[p.species]
            lines_out.append(f"variable current_ion_type equal {sp['type_index']}\n")
            lines_out.append(f"variable current_use_molecule equal {1 if sp['is_molecule'] else 0}\n")
            lines_out.append(f"variable current_ion_energy equal ${{phase_{i}_energy}}\n")
            lines_out.append(f"variable current_M_ion equal ${{{sp['mass_var']}}}\n")
            lines_out.append(f"variable current_phase_mix_idx equal 0\n")
            lines_out.append(f"variable current_phase_idx equal {i}\n")
            if has_ar:
                lines_out.append(f"variable current_needs_removal equal {1 if sp['remove_after_impact'] else 0}\n")
            if switch_potential:
                lines_out.append(f"variable current_needs_zbl equal {1 if sp['needs_zbl'] else 0}\n")
        if has_any_burst:
            lines_out.append(f"variable current_phase_is_burst equal {1 if p.radical_burst else 0}\n")
        return lines_out

    def _ion_quoted_for(p, i, radical_energy_line, removal_line, zbl_line):
        """Quoted if-then clause fragments for phase i override block."""
        if p.ion_mix is not None:
            base = (
                f'"variable current_phase_mix_idx equal {mix_idx_of[i]}" &\n'
                f'"variable current_phase_idx equal {i}" &\n'
            )
        else:
            sp = SPECIES[p.species]
            base = (
                f'"variable current_ion_type equal {sp["type_index"]}" &\n'
                f'"variable current_use_molecule equal {1 if sp["is_molecule"] else 0}" &\n'
                f'"variable current_ion_energy equal ${{phase_{i}_energy}}" &\n'
                f'"variable current_M_ion equal ${{{sp["mass_var"]}}}" &\n'
                f'"variable current_phase_mix_idx equal 0" &\n'
                f'"variable current_phase_idx equal {i}" &\n'
            )
        burst_line = (
            f'"variable current_phase_is_burst equal {1 if p.radical_burst else 0}" &\n'
        ) if has_any_burst else ""
        return base + burst_line

    # Default: last phase (direct assignment, no if-wrapper)
    last = phases[-1]
    rad_angl_v, inter_t_v, rad_azimuth_v = _phase_radical_vars(last, N - 1)

    last_sp = SPECIES[last.species] if last.ion_mix is None else None

    lines.append(f"# Phase selection (default: phase {N-1} = {_phase_label(last, N-1)})\n")
    for ln in _ion_defaults_for(last, N - 1):
        lines.append(ln)
    lines.append(f"variable current_flux_ratio equal ${{phase_{N-1}_flux_ratio}}\n")
    if has_any_radicals and last.flux_ratio > 0:
        lines.append(f"variable current_radical_energy equal ${{phase_{N-1}_radical_energy}}\n")
    lines.append(f"variable current_rad_angl equal {rad_angl_v}\n")
    lines.append(f"variable current_rad_azimuth equal {rad_azimuth_v}\n")
    lines.append(f"variable current_inter_neutral_time equal {inter_t_v}\n")
    lines.append("\n")

    # Reverse-order overrides for phases 0..N-2
    for i in range(N - 2, -1, -1):
        p = phases[i]
        rad_angl_v, inter_t_v, rad_azimuth_v = _phase_radical_vars(p, i)
        radical_energy_line = (
            f'"variable current_radical_energy equal ${{phase_{i}_radical_energy}}" &\n'
            if (has_any_radicals and p.flux_ratio > 0) else ""
        )

        sp = SPECIES[p.species] if p.ion_mix is None else None
        removal_line = (
            f' &\n"variable current_needs_removal equal '
            f'{1 if sp["remove_after_impact"] else 0}"'
        ) if has_ar and sp is not None else (
            f' &\n"variable current_needs_removal equal 0"'
        ) if has_ar else ""
        zbl_line = (
            f' &\n"variable current_needs_zbl equal '
            f'{1 if sp["needs_zbl"] else 0}"'
        ) if switch_potential and sp is not None else (
            f' &\n"variable current_needs_zbl equal 0"'
        ) if switch_potential else ""

        ion_block = _ion_quoted_for(p, i, radical_energy_line, removal_line, zbl_line)
        lines.append(
            f'if "${{idx_in_cycle}} < ${{phase_{i}_end}}" then &\n'
            f"{ion_block}"
            f'"variable current_flux_ratio equal ${{phase_{i}_flux_ratio}}" &\n'
            f"{radical_energy_line}"
            f'"variable current_rad_angl equal {rad_angl_v}" &\n'
            f'"variable current_rad_azimuth equal {rad_azimuth_v}" &\n'
            f'"variable current_inter_neutral_time equal {inter_t_v}"'
            f"{removal_line}"
            f"{zbl_line}\n"
            f"\n"
        )

    return "".join(lines)


def _phase_ion_mix_dispatch_block(phases: list) -> str:
    """Generate the per-impact stochastic ion selection block for cycling mix phases.

    Called once per outer-loop iteration after _phase_selection_block.  When
    current_phase_mix_idx == 0 the block is a no-op (non-mix phase).  For mix
    phases, a random draw selects one ion component and sets current_ion_type,
    current_use_molecule, current_ion_energy, current_M_ion.
    """
    # Collect mix phases in order of their dispatch index
    mix_phases = [(i, p) for i, p in enumerate(phases) if p.ion_mix is not None]
    if not mix_phases:
        return ""

    lines = [
        "# ── Per-phase ion mix stochastic dispatch ────────────────────────────────────\n",
        'if "${current_phase_mix_idx} == 0" then "jump SELF phase_mix_dispatch_done"\n',
        "\n",
    ]

    for dispatch_idx, (phase_i, p) in enumerate(mix_phases, start=1):
        lines.append(
            f'if "${{current_phase_mix_idx}} == {dispatch_idx}" then "jump SELF phase_mix_{dispatch_idx}"\n'
        )
    lines.append("\n")

    for dispatch_idx, (phase_i, p) in enumerate(mix_phases, start=1):
        mix = p.ion_mix
        total = sum(c.fraction for c in mix)
        lines.append(f"label       phase_mix_{dispatch_idx}\n")
        lines.append(f"variable    r_mix equal random(0,1,${{c}}+50000+${{seed_adjust}})\n")
        # Jump table for components except last
        cumulative = 0.0
        for j, comp in enumerate(mix[:-1]):
            cumulative += comp.fraction / total
            lines.append(
                f'if "${{r_mix}} < {cumulative:.8f}" then "jump SELF phase_mix_{dispatch_idx}_ion_{j}"\n'
            )
        lines.append(f"jump        SELF phase_mix_{dispatch_idx}_ion_{len(mix)-1}\n\n")

        for j, comp in enumerate(mix):
            sp = SPECIES[comp.species]
            energy_per_atom = comp.energy / sp["energy_divisor"]
            lines.append(f"label       phase_mix_{dispatch_idx}_ion_{j}\n")
            lines.append(f"variable    current_ion_type equal {sp['type_index']}\n")
            lines.append(f"variable    current_use_molecule equal {1 if sp['is_molecule'] else 0}\n")
            lines.append(f"variable    current_ion_energy equal {energy_per_atom}\n")
            lines.append(f"variable    current_M_ion equal ${{{sp['mass_var']}}}\n")
            lines.append(f"jump        SELF phase_mix_dispatch_done\n\n")

    lines.append("label       phase_mix_dispatch_done\n")
    lines.append("# ─────────────────────────────────────────────────────────────────────────────\n\n")
    return "".join(lines)


def _cycle_burst_blocks(phases: list, spec: SimSpec, nonargon_refresh: str) -> str:
    """Generate burst-mode radical injection blocks for all burst phases.

    Returns empty string when no phase uses radical_burst.
    The outer loop dispatches to the right label via current_phase_is_burst
    and current_phase_idx.
    """
    burst_phases = [(i, p) for i, p in enumerate(phases) if p.radical_burst]
    if not burst_phases:
        return ""

    from ..lammps.config import _DEPOSIT_REGION_BUG_FACTOR

    dm = spec.dump_mode
    dump_cols = "id type x y z vx vy vz fx fy fz q"
    ml = spec.ml

    lines = [
        "# ── Burst radical injection dispatch ─────────────────────────────────────────\n",
        'if "${current_phase_is_burst} == 0" then "jump SELF burst_dispatch_done"\n',
        'if "${cn_start} >= 1" then "jump SELF skip_chem"\n',
    ]
    for i, _p in burst_phases:
        lines.append(
            f'if "${{current_phase_idx}} == {i}" then "jump SELF burst_phase_{i}"\n'
        )
    lines.append('jump        SELF burst_dispatch_done\n\n')

    for phase_i, p in burst_phases:
        chunk_size = p.radical_burst_chunk if p.radical_burst_chunk > 0 else max(1, round(0.5 * ml))
        total = max(1, round(p.flux_ratio))
        n_full = total // chunk_size
        remainder = total % chunk_size
        chunks = [chunk_size] * n_full + ([remainder] if remainder > 0 else [])
        attempt = p.radical_burst_attempt

        lines.append(f"label       burst_phase_{phase_i}\n")
        lines.append(
            f"# burst: {len(chunks)} chunk(s) × ≤{chunk_size} atoms = {total} O• total\n"
            f"# Fixed velocity from current_radical_energy / current_rad_angl / current_rad_azimuth\n"
            f"variable    vel_chem_burst equal "
            f"sqrt(2*${{current_radical_energy}}*6.02214129*1.0e+7/${{M_O}}/6242)/1000\n"
            f"variable    vx_burst equal v_vel_chem_burst*sin(${{current_rad_angl}}*PI/180)*cos(${{current_rad_azimuth}}*PI/180)\n"
            f"variable    vy_burst equal v_vel_chem_burst*sin(${{current_rad_angl}}*PI/180)*sin(${{current_rad_azimuth}}*PI/180)\n"
            f"variable    vz_burst equal -v_vel_chem_burst*cos(${{current_rad_angl}}*PI/180)\n"
            f"timestep    1e-10\n"
            f"\n"
        )

        for ci, csize in enumerate(chunks):
            chunk_dump_file = f"etch_event_trajs/event_dump_burst_${{c}}_{ci}.dump"
            if dm == "none":
                chunk_dump_open  = ""
                chunk_dump_close = ""
                chunk_etch_event = ""
            elif dm == "all":
                chunk_dump_open  = f"dump        current_dump_burst all custom 100 {chunk_dump_file} {dump_cols}\n"
                chunk_dump_close = f"undump      current_dump_burst\n"
                chunk_etch_event = ""
            else:  # etch_only — keep if cluster count increases during this chunk's thermalization
                _first_guard = (
                    f'if "${{c}} == 1" then "variable keep_dump_burst equal 1"\n'
                    if (spec.dump_first_impact and ci == 0) else ""
                )
                chunk_dump_open = (
                    f"variable    keep_dump_burst equal 0\n"
                    + _first_guard
                    + f"dump        current_dump_burst all custom 100 {chunk_dump_file} {dump_cols}\n"
                )
                chunk_etch_event = (
                    f'if "$(c_nclusts) > ${{burst_nclusts0}}" then "variable keep_dump_burst equal 1"\n'
                )
                chunk_dump_close = (
                    f'if "${{keep_dump_burst}} == 0" then "shell rm {chunk_dump_file}"\n'
                    f"undump      current_dump_burst\n"
                )

            lines.append(
                f"# --- Burst chunk {ci+1}/{len(chunks)}: {csize} atoms (deposit) ---\n"
                f"if \"$(bound(all,zmax)+v_radical_i_above+2.0) > $(zhi)\" then "
                f"\"change_box all z delta 0 $(bound(all,zmax)+v_radical_i_above+2.0-zhi) units box\" "
                f"\"region bbox delete\" "
                f"\"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\"\n"
                f"variable    z_ins{phase_i}_{ci} equal bound(all,zmax)+${{radical_i_above}}\n"
                f"region      bzone{phase_i}_{ci} block EDGE EDGE EDGE EDGE "
                f"$(v_z_ins{phase_i}_{ci} - 0.1) $(v_z_ins{phase_i}_{ci} + 0.1) units box\n"
                f"group       insert clear\n"
                f"group       mobile subtract all anchor\n"
                f"variable    burst_seed{phase_i}_{ci}_1 equal "
                f"floor(random(1,72099+${{seed_adjust}},${{c}}*100000+{ci}*10000+{ci+1}))\n"
                f"fix         burst_depo insert deposit 1 3 1 ${{burst_seed{phase_i}_{ci}_1}} "
                f"attempt {attempt} "
                f"vx ${{vx_burst}} ${{vx_burst}} "
                f"vy ${{vy_burst}} ${{vy_burst}} "
                f"vz ${{vz_burst}} ${{vz_burst}} "
                f"region bzone{phase_i}_{ci} near 2.0\n"
                f"fix         2 mobile nve\n"
                f"fix         3 insert nve\n"
                f"run         1 post no\n"
                f"run         0\n"
                f"{nonargon_refresh}"
                f"unfix       burst_depo\n"
                f"unfix       2\n"
                f"unfix       3\n"
            )
            if csize > 1:
                lines.append(
                    f"variable    burst_lp{phase_i}_{ci} loop {csize - 1}\n"
                    f"label       burst_dep_{phase_i}_{ci}\n"
                    f"group       insert clear\n"
                    f"group       mobile subtract all anchor\n"
                    f"variable    burst_seed{phase_i}_{ci} equal "
                    f"floor(random(1,72099+${{seed_adjust}},${{c}}*100000+{ci}*10000+v_burst_lp{phase_i}_{ci}))\n"
                    f"fix         burst_depo insert deposit 1 3 1 ${{burst_seed{phase_i}_{ci}}} "
                    f"attempt {attempt} "
                    f"vx ${{vx_burst}} ${{vx_burst}} "
                    f"vy ${{vy_burst}} ${{vy_burst}} "
                    f"vz ${{vz_burst}} ${{vz_burst}} "
                    f"region bzone{phase_i}_{ci} near 2.0\n"
                    f"fix         2 mobile nve\n"
                    f"fix         3 insert nve\n"
                    f"run         1 post no\n"
                    f"run         0\n"
                    f"{nonargon_refresh}"
                    f"unfix       burst_depo\n"
                    f"unfix       2\n"
                    f"unfix       3\n"
                    f"next        burst_lp{phase_i}_{ci}\n"
                    f"jump        SELF burst_dep_{phase_i}_{ci}\n\n"
                )
            # Dynamics phase: c_nclusts is current after the last run 0 above
            lines.append(
                f"region      bzone{phase_i}_{ci} delete\n"
                f"# --- Burst chunk {ci+1}/{len(chunks)}: thermalize ---\n"
                f"{chunk_dump_open}"
                f"variable    burst_nclusts0 equal $(c_nclusts)\n"
                f"fix         2 mobile nve\n"
                f"include     thermalize.lmp\n"
                f"unfix       2\n"
                f"{chunk_etch_event}"
                f"{chunk_dump_close}"
                f"\n"
            )

        lines.append(
            f"group       carbon type 1\n"
            f"group       hydrogen type 2\n"
            f"group       oxygen type 3\n"
            f"variable    ncarbon_b equal count(carbon)\n"
            f"variable    nhydrogen_b equal count(hydrogen)\n"
            f"variable    noxygen_b equal count(oxygen)\n"
            f'print       "Burst complete (${{c}} impacts done)"\n'
            f'print       "${{c}} 1 ${{ncarbon_b}} ${{nhydrogen_b}} ${{noxygen_b}}" append ncarbon.txt\n'
            f"write_data  impact_snaps/${{c}}_1.data nofix nocoeff\n"
            f"variable    cn_start equal 0\n"
            f"jump        SELF skip_chem\n\n"
        )

    lines.append(
        "label       burst_dispatch_done\n"
        "# ─────────────────────────────────────────────────────────────────────────────\n\n"
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
        _first_guard = (
            'if "${c} == 1" then "variable keep_dump equal 1"\n'
            if spec.dump_first_impact else ""
        )
        ion_dump_open = (
            f"variable    keep_dump equal 0\n"
            f"{_first_guard}"
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
    has_any_mix = any(p.ion_mix is not None for p in spec.phases)
    has_any_burst = any(p.radical_burst for p in spec.phases)
    n_types = 4 if has_ar else 3
    dm = spec.dump_mode

    phase_names = " → ".join(_phase_label(p, i) for i, p in enumerate(spec.phases))

    molecule_decl = "molecule    O2 O2.molecule\n" if has_o2 else ""

    # nonargon group refresh: only needed when Ar is present
    nonargon_refresh = "group nonargon type 1 2 3\n" if has_ar else ""

    # For ZBL mass line: find the first ZBL species' mass var
    _zbl_mass_line = ""
    if has_ar:
        for p in spec.phases:
            srcs = p.ion_mix if p.ion_mix is not None else [type('_', (), {'species': p.species})()]
            for comp in srcs:
                sp = SPECIES[comp.species]
                if sp["needs_zbl"]:
                    _zbl_mass_line = f"mass        4 ${{{sp['mass_var']}}}\n"
                    break
            if _zbl_mass_line:
                break

    masses = (
        f"mass        1 ${{M_C}}\n"
        f"mass        2 ${{M_H}}\n"
        f"mass        3 ${{M_O}}\n"
        + _zbl_mass_line
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

    # Neutral dump: always open in "all"/"etch_only"; delete if no etch in "etch_only"
    _neutral_dump_file = f"etch_event_trajs/event_dump_n${{c}}_${{cn}}.dump"
    if dm == "all":
        neutral_dump_open  = (
            f"dump        current_dump_n all custom 100 {_neutral_dump_file} "
            f"id type x y z vx vy vz fx fy fz q\n"
        )
        neutral_dump_close = "undump      current_dump_n\n"
    elif dm == "etch_only":
        _rad_first_guard = (
            'if "${c} == 1 && ${cn} == 1" then "variable keep_dump_n equal 1"\n'
            if spec.dump_first_impact else ""
        )
        neutral_dump_open  = (
            f"variable    keep_dump_n equal 0\n"
            f"{_rad_first_guard}"
            f"dump        current_dump_n all custom 100 {_neutral_dump_file} "
            f"id type x y z vx vy vz fx fy fz q\n"
        )
        neutral_dump_close = (
            f'if "${{keep_dump_n}} == 0" then "shell rm {_neutral_dump_file}"\n'
            f"undump      current_dump_n\n"
        )
    else:
        neutral_dump_open  = ""
        neutral_dump_close = ""

    return (
        f"# head.lmp — generated by DiamondEtchMD (cycling mode)\n"
        f"# orientation={spec.orientation}  phases: {phase_names}\n"
        f"# {spec.cycles} cycle(s)  T={spec.surface_temperature}K  ion_angle={spec.ion_angle[0]}deg\n"
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
        f"{_potential_block(spec)}"
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
        + (  # stochastic ion selection for ion_mix phases
            f"{_phase_ion_mix_dispatch_block(spec.phases)}"
            if has_any_mix else ""
        ) +
        f"# Ion velocities for current phase\n"
        f"variable    vel_ion equal sqrt(2*${{current_ion_energy}}*6.02214129*1.0e+7/${{current_M_ion}}/6242)/1000\n"
        f"variable    velx_ion equal sin(${{ion_angl}}*PI/180)*cos(${{ion_azimuth}}*PI/180)*${{vel_ion}}\n"
        f"variable    vely_ion equal sin(${{ion_angl}}*PI/180)*sin(${{ion_azimuth}}*PI/180)*${{vel_ion}}\n"
        f"variable    velz_ion equal cos(${{ion_angl}}*PI/180)*${{vel_ion}}\n"
        f"\n"
        + (  # potential switch block — only when mixing Ar and non-Ar phases
            _potential_switch_block(spec) if switch_pot else ""
        ) +
        f"# Adaptive timestep (used for both neutral and ion loops)\n"
        f"fix         ats all dt/reset 1 NULL 1 0.01 units box\n"
        f"\n"
        + (  # neutral/burst deposition — omitted entirely when no phase has radicals or burst
            f"# ========================= Begin neutral deposition loop =========================\n"
            f'if "${{current_flux_ratio}} == 0" then "jump SELF skip_chem"\n'
            + (  # burst dispatch — jumps to per-phase burst block, skips regular loop
                _cycle_burst_blocks(spec.phases, spec, nonargon_refresh)
                if has_any_burst else ""
            ) +
            f"# Stochastic floor/ceil: draw floor(R) or ceil(R) radicals so long-run avg = R\n"
            f"variable    flux_lo equal floor(v_current_flux_ratio)\n"
            f"variable    flux_hi equal ceil(v_current_flux_ratio)\n"
            f"variable    p_hi equal v_current_flux_ratio-v_flux_lo\n"
            f"variable    r_fr equal $(random(0,1,v_c+80000+v_seed_adjust))\n"
            f"variable    target_flux equal ${{flux_lo}}\n"
            f'if "${{p_hi}} > 0 && ${{r_fr}} < ${{p_hi}}" then "variable target_flux equal ${{flux_hi}}"\n'
            f"\n"
            f'if "${{cn_start}} > 0 && ${{cn_start}} < ${{target_flux}}" then &\n'
            f'"variable neutral_lp loop $(v_target_flux-v_cn_start)" &\n'
            f'elif "${{cn_start}} == ${{target_flux}}" &\n'
            f'"jump SELF skip_chem" &\n'
            f"else &\n"
            f'"variable neutral_lp loop ${{target_flux}}"\n'
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
            f"variable    velx_chem equal sin(${{current_rad_angl}}*PI/180)*cos(${{current_rad_azimuth}}*PI/180)*${{vel_chem}}\n"
            f"variable    vely_chem equal sin(${{current_rad_angl}}*PI/180)*sin(${{current_rad_azimuth}}*PI/180)*${{vel_chem}}\n"
            f"variable    velz_chem equal cos(${{current_rad_angl}}*PI/180)*${{vel_chem}}\n"
            f"\n"
            f"# Deposit O• radical (always type 3)\n"
            f"fix         depo insert deposit 1 3 1 ${{deposeed}} global "
            f"${{radical_i_above}} ${{radical_i_above}} "
            f"vx ${{velx_chem}} ${{velx_chem}} vy ${{vely_chem}} ${{vely_chem}} vz -${{velz_chem}} -${{velz_chem}} "
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
            f'"variable starting_nclusts equal $(c_nclusts)"'
            + (' &\n"variable keep_dump_n equal 1"' if dm == "etch_only" else "")
            + "\n"
            + f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
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
            + (
                ""
                if spec.skip_radical_thermalization else
                f"# Thermalize after each radical\n"
                f"include     thermalize.lmp\n"
            ) +
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
            if (has_any_radicals or has_any_burst) else ""
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
        f'${{ion_i_above}} ${{ion_i_above}} vx ${{velx_ion}} ${{velx_ion}} vy ${{vely_ion}} ${{vely_ion}} '
        f'vz -${{velz_ion}} -${{velz_ion}} region bbox units box mol O2" &\n'
        f"else &\n"
        f'"fix depo insert deposit 1 ${{current_ion_type}} 1 ${{deposeed}} global '
        f'${{ion_i_above}} ${{ion_i_above}} vx ${{velx_ion}} ${{velx_ion}} vy ${{vely_ion}} ${{vely_ion}} '
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

    _neutral_dump_file = f"etch_event_trajs/event_dump_n${{c}}_${{cn}}.dump"
    if dm == "all":
        neutral_dump_open  = (
            f"dump        current_dump_n all custom 100 {_neutral_dump_file} "
            f"id type x y z vx vy vz fx fy fz q\n"
        )
        neutral_dump_close = "undump      current_dump_n\n"
    elif dm == "etch_only":
        _rad_first_guard = (
            'if "${c} == 1 && ${cn} == 1" then "variable keep_dump_n equal 1"\n'
            if spec.dump_first_impact else ""
        )
        neutral_dump_open  = (
            f"variable    keep_dump_n equal 0\n"
            f"{_rad_first_guard}"
            f"dump        current_dump_n all custom 100 {_neutral_dump_file} "
            f"id type x y z vx vy vz fx fy fz q\n"
        )
        neutral_dump_close = (
            f'if "${{keep_dump_n}} == 0" then "shell rm {_neutral_dump_file}"\n'
            f"undump      current_dump_n\n"
        )
    else:
        neutral_dump_open  = ""
        neutral_dump_close = ""

    return (
        f"# head.lmp — generated by DiamondEtchMD (carbon-etch cycling)\n"
        f"# config_file={spec.initial_config_file}  phases: {phase_names}\n"
        f"# {spec.cycles} cycle(s)  T={spec.surface_temperature}K  ion_angle={spec.ion_angle[0]}deg\n"
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
        f"{_potential_block(spec)}"
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
        f"variable    velx_ion equal sin(${{ion_angl}}*PI/180)*cos(${{ion_azimuth}}*PI/180)*${{vel_ion}}\n"
        f"variable    vely_ion equal sin(${{ion_angl}}*PI/180)*sin(${{ion_azimuth}}*PI/180)*${{vel_ion}}\n"
        f"variable    velz_ion equal cos(${{ion_angl}}*PI/180)*${{vel_ion}}\n"
        f"\n"
        + (
            _potential_switch_block(spec) if switch_pot else ""
        ) +
        f"fix         ats all dt/reset 1 NULL 1 0.01 units box\n"
        f"\n"
        + (
            f"# ========================= Begin neutral deposition loop =========================\n"
            f'if "${{current_flux_ratio}} == 0" then "jump SELF skip_chem"\n'
            f"# Stochastic floor/ceil: draw floor(R) or ceil(R) radicals so long-run avg = R\n"
            f"variable    flux_lo equal floor(v_current_flux_ratio)\n"
            f"variable    flux_hi equal ceil(v_current_flux_ratio)\n"
            f"variable    p_hi equal v_current_flux_ratio-v_flux_lo\n"
            f"variable    r_fr equal $(random(0,1,v_c+80000+v_seed_adjust))\n"
            f"variable    target_flux equal ${{flux_lo}}\n"
            f'if "${{p_hi}} > 0 && ${{r_fr}} < ${{p_hi}}" then "variable target_flux equal ${{flux_hi}}"\n'
            f"\n"
            f'if "${{cn_start}} > 0 && ${{cn_start}} < ${{target_flux}}" then &\n'
            f'"variable neutral_lp loop $(v_target_flux-v_cn_start)" &\n'
            f'elif "${{cn_start}} == ${{target_flux}}" &\n'
            f'"jump SELF skip_chem" &\n'
            f"else &\n"
            f'"variable neutral_lp loop ${{target_flux}}"\n'
            f"\n"
            f"label       neutral_loop\n"
            f"variable    cn equal ${{cn}}+1\n"
            f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{cn}}))\n"
            f"group       insert clear\n"
            f"group       mobile subtract all anchor\n"
            f"{nonargon_refresh}"
            f"\n"
            f"variable    vel_chem equal sqrt(2*${{current_radical_energy}}*6.02214129*1.0e+7/${{M_O}}/6242)/1000\n"
            f"variable    velx_chem equal sin(${{current_rad_angl}}*PI/180)*cos(${{current_rad_azimuth}}*PI/180)*${{vel_chem}}\n"
            f"variable    vely_chem equal sin(${{current_rad_angl}}*PI/180)*sin(${{current_rad_azimuth}}*PI/180)*${{vel_chem}}\n"
            f"variable    velz_chem equal cos(${{current_rad_angl}}*PI/180)*${{vel_chem}}\n"
            f"\n"
            f"fix         depo insert deposit 1 3 1 ${{deposeed}} global "
            f"${{radical_i_above}} ${{radical_i_above}} "
            f"vx ${{velx_chem}} ${{velx_chem}} vy ${{vely_chem}} ${{vely_chem}} vz -${{velz_chem}} -${{velz_chem}} "
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
            f'"variable starting_nclusts equal $(c_nclusts)"'
            + (' &\n"variable keep_dump_n equal 1"' if dm == "etch_only" else "")
            + "\n"
            + f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
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
        f'${{ion_i_above}} ${{ion_i_above}} vx ${{velx_ion}} ${{velx_ion}} vy ${{vely_ion}} ${{vely_ion}} '
        f'vz -${{velz_ion}} -${{velz_ion}} region bbox units box mol O2" &\n'
        f"else &\n"
        f'"fix depo insert deposit 1 ${{current_ion_type}} 1 ${{deposeed}} global '
        f'${{ion_i_above}} ${{ion_i_above}} vx ${{velx_ion}} ${{velx_ion}} vy ${{vely_ion}} ${{vely_ion}} '
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
