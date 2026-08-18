"""
lammps/head.py — generator for head.lmp, the main LAMMPS driver script.

head.lmp is the top-level script passed to lmp via -in.  It includes config.lmp,
sets up atom groups, the force field, computes, and runs the per-impact outer loop.

Supports two single-species modes:
  ion-etch  (flux_ratio == 0): ions only; 4-col ncarbon.txt format.
  RIE-etch     (flux_ratio > 0) : O• radicals deposited before each ion impact;
               5-col ncarbon.txt format (same as cycle-etch).

  For each impact (ion-etch):
    1. deposit incident particle
    2. adaptive-timestep NVE until impact_time reached or cluster separates
    3. thermalize substrate
    4. replenish carbon if etched below baseline (addfix loop)
    5. write ncarbon.txt (4-col), save impact_snaps/${c}.data

  For each impact (RIE-etch) — adds a radical pre-exposure loop:
    0. deposit flux_ratio O• radicals; after each: write 5-col ncarbon.txt,
       save impact_snaps/${c}_${cn}.data
    1–5. same as ion-etch, but ncarbon.txt uses 5-col; save impact_snaps/${c}_0.data

The lattice command and bottom expression embedded in head.lmp are
orientation-specific and are pulled from the ORIENT registry.

Species-specific behaviour (Ar ZBL, O2 molecule injection) is resolved at
generation time via the SPECIES registry — no LAMMPS-level conditionals.

Note: LAMMPS ${variable} syntax must be escaped as ${{variable}} in Python
f-strings.  LAMMPS $(expression) uses parentheses and needs no escaping.
"""

from ..orientations import ORIENT
from ..species import SPECIES
from ..spec import SimSpec, IonComponent


def _potential_block(species: dict) -> str:
    """Return the pair_style / pair_coeff / QEQ block for the given species."""
    if species["needs_zbl"]:
        # Inert ion (Ar, Er, ...): hybrid ReaxFF + ZBL for short-range nuclear repulsion
        Z = species["atomic_number"]
        return (
            f'if "${{pot}} == REAX" then &\n'
            f'"pair_style  hybrid reaxff NULL zbl 5.0 6.0" &\n'
            f'"pair_coeff * * reaxff ffield.reax C H O NULL" &\n'
            f'"pair_coeff 1 4 zbl 6.0 {Z}" &\n'
            f'"pair_coeff 2 4 zbl 1.0 {Z}" &\n'
            f'"pair_coeff 3 4 zbl 8.0 {Z}" &\n'
            f'"pair_coeff 4 4 zbl {Z} {Z}" &\n'
            f'"group nonargon type 1 2 3" &\n'
            f'"fix reax_qeq nonargon qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        )
    else:
        # O, O2, H: plain ReaxFF — type 4 slot maps to C so all pair coeffs are set
        return (
            f'if "${{pot}} == REAX" then &\n'
            f'"pair_style reaxff NULL" &\n'
            f'"pair_coeff * * ffield.reax C H O C" &\n'
            f'"fix reax_qeq all qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        )


def _expose_zone_def(spec: SimSpec) -> str:
    """Return LAMMPS commands that define region expose_zone at the current box bounds.

    expose_zone is a sub-region of bbox restricted to the unmasked xy area.
    When spec.mask_type is None, returns "" (deposit uses bbox directly).
    Must be called every time bbox is redefined since EDGE captures box state.
    When spec.invert_mask is True, 'side out' inverts the region so atoms land
    at the edges/frame instead of the center window.
    """
    if spec.mask_type is None:
        return ""
    mt = spec.mask_type
    side = " side out" if spec.invert_mask else ""
    if mt == "xymask":
        return (
            f"region      expose_zone block "
            f"$(v_mask_lo_x) $(v_mask_hi_x) "
            f"$(v_mask_lo_y) $(v_mask_hi_y) "
            f"EDGE EDGE{side} units box\n"
        )
    elif mt == "xmask":
        return (
            f"region      expose_zone block "
            f"$(v_mask_lo_x) $(v_mask_hi_x) "
            f"EDGE EDGE EDGE EDGE{side} units box\n"
        )
    else:  # ymask
        return (
            f"region      expose_zone block "
            f"EDGE EDGE "
            f"$(v_mask_lo_y) $(v_mask_hi_y) "
            f"EDGE EDGE{side} units box\n"
        )


def _expose_zone_redef(spec: SimSpec) -> str:
    """Return LAMMPS commands to delete and redefine expose_zone.

    Use this wherever bbox is deleted and recreated mid-run.
    Returns "" when no mask is active.
    """
    if spec.mask_type is None:
        return ""
    return f"region      expose_zone delete\n" + _expose_zone_def(spec)


def _deposit_region(spec: SimSpec) -> str:
    """Return the LAMMPS region name to use for ion/radical deposit commands."""
    return "expose_zone" if spec.mask_type is not None else "bbox"


def _bzone_xy_bounds(spec: SimSpec) -> str:
    """Return the x y lo/hi part of a bzone region block, masked or full-box."""
    mt = spec.mask_type if spec is not None else None
    if mt == "xymask":
        return "$(v_mask_lo_x) $(v_mask_hi_x) $(v_mask_lo_y) $(v_mask_hi_y)"
    elif mt == "xmask":
        return "$(v_mask_lo_x) $(v_mask_hi_x) EDGE EDGE"
    elif mt == "ymask":
        return "EDGE EDGE $(v_mask_lo_y) $(v_mask_hi_y)"
    else:
        return "EDGE EDGE EDGE EDGE"


def _bzone_inner_xy(spec: SimSpec) -> str:
    """Return x y bounds of the inner (center) mask block for intersect-based inverted bzone."""
    mt = spec.mask_type
    if mt == "xymask":
        return "$(v_mask_lo_x) $(v_mask_hi_x) $(v_mask_lo_y) $(v_mask_hi_y)"
    elif mt == "xmask":
        return "$(v_mask_lo_x) $(v_mask_hi_x) EDGE EDGE"
    else:  # ymask
        return "EDGE EDGE $(v_mask_lo_y) $(v_mask_hi_y)"


def _bzone_def(spec: SimSpec, name: str, z_lo: str, z_hi: str) -> str:
    """Return LAMMPS region definition commands for a burst z-slab region.

    Normal: single block region (xy masked or full, z = slab).
    Inverted mask: frame = intersect(full-xy z-slab, complement-of-center z-slab).
    The complement is achieved by 'side out' on the center block; 'region subtract' is
    not a valid LAMMPS style — use 'region intersect' with side out instead.
    """
    if not spec.invert_mask or spec.mask_type is None:
        return (
            f"region      {name} block {_bzone_xy_bounds(spec)} {z_lo} {z_hi} units box\n"
        )
    inner_xy = _bzone_inner_xy(spec)
    return (
        f"region      {name}_full block EDGE EDGE EDGE EDGE {z_lo} {z_hi} units box\n"
        f"region      {name}_center block {inner_xy} {z_lo} {z_hi} side out units box\n"
        f"region      {name} intersect 2 {name}_full {name}_center\n"
    )


def _bzone_cleanup(spec: SimSpec, name: str) -> str:
    """Return LAMMPS region delete commands for a bzone region created by _bzone_def."""
    if not spec.invert_mask or spec.mask_type is None:
        return f"region      {name} delete\n"
    return (
        f"region      {name} delete\n"
        f"region      {name}_center delete\n"
        f"region      {name}_full delete\n"
    )


def _freeze_zone_def(spec: SimSpec, name: str, z_lo_expr: str, z_hi_expr: str) -> str:
    """Return LAMMPS region definition for the masked (non-deposition) zone at a z slab.

    This is the mirror of _bzone_def: bzone covers where atoms land; freeze_zone covers
    the complementary masked area where atoms are blocked.
    Normal mask  (invert_mask=False): frame = intersect(full z-slab, complement-of-inner z-slab).
    Inverted mask (invert_mask=True): masked region is the center → plain inner block.
    'region subtract' is not a valid LAMMPS style; use intersect + side out instead.
    """
    inner_xy = _bzone_inner_xy(spec)
    if spec.invert_mask or spec.mask_type is None:
        return (
            f"region      {name} block {inner_xy} {z_lo_expr} {z_hi_expr} units box\n"
        )
    return (
        f"region      {name}_full block EDGE EDGE EDGE EDGE {z_lo_expr} {z_hi_expr} units box\n"
        f"region      {name}_inner block {inner_xy} {z_lo_expr} {z_hi_expr} side out units box\n"
        f"region      {name} intersect 2 {name}_full {name}_inner\n"
    )


def _freeze_zone_cleanup(spec: SimSpec, name: str) -> str:
    """Return LAMMPS region delete commands for a region created by _freeze_zone_def."""
    if spec.invert_mask or spec.mask_type is None:
        return f"region      {name} delete\n"
    return (
        f"region      {name} delete\n"
        f"region      {name}_inner delete\n"
        f"region      {name}_full delete\n"
    )


def _freeze_mask_block(spec: SimSpec) -> str:
    """Return LAMMPS commands to freeze the top surface layer in the masked region.

    Must be inserted immediately after 'group anchor region anchor'.
    The z floor is computed from bound(all,zmax) on the first run (n_complete==0) and
    written to freeze_mask_z.txt so restarts use the original surface height — not the
    top of any amorphous carbon that has since deposited on the masked area.
    """
    if not spec.freeze_mask or spec.mask_type is None:
        return ""
    depth = spec.freeze_mask_depth
    z_hi  = depth + 0.5
    z_lo_expr = "$(v_z_freeze_lo)"
    z_hi_expr = f"$(v_z_freeze_lo+{z_hi})"
    return (
        f"# Freeze top-layer atoms in masked region\n"
        f"# z floor saved on first run (n_complete==0) and reloaded on restart\n"
        f'if "${{n_complete}} == 0" then &\n'
        f'"variable z_freeze_lo equal $(bound(all,zmax) - {depth})" &\n'
        f'"print \'$(v_z_freeze_lo)\' file freeze_mask_z.txt" &\n'
        f"else &\n"
        f'"variable z_freeze_lo file freeze_mask_z.txt"\n'
        + _freeze_zone_def(spec, "freeze_mask_zone", z_lo_expr, z_hi_expr)
        + f"group       freeze_mask_atoms region freeze_mask_zone\n"
        f"group       anchor union anchor freeze_mask_atoms\n"
        f"group       freeze_mask_atoms delete\n"
        + _freeze_zone_cleanup(spec, "freeze_mask_zone")
        + f"variable    z_freeze_lo delete\n"
    )


def _expose_zone_redef_if_cmds(spec: SimSpec) -> str:
    """Return space-separated quoted LAMMPS commands for expose_zone delete+redef.

    Used inside burst-mode 'if ... then "cmd1" "cmd2" ...' strings where
    _expose_zone_redef() (multi-line) cannot be used.  Returns "" if no mask.
    """
    if spec.mask_type is None:
        return ""
    mt = spec.mask_type
    side = " side out" if spec.invert_mask else ""
    if mt == "xymask":
        block = f"$(v_mask_lo_x) $(v_mask_hi_x) $(v_mask_lo_y) $(v_mask_hi_y) EDGE EDGE{side} units box"
    elif mt == "xmask":
        block = f"$(v_mask_lo_x) $(v_mask_hi_x) EDGE EDGE EDGE EDGE{side} units box"
    else:  # ymask
        block = f"EDGE EDGE $(v_mask_lo_y) $(v_mask_hi_y) EDGE EDGE{side} units box"
    return (
        f'"region expose_zone delete" '
        f'"region expose_zone block {block}" '
    )


def _deposit_line(species: dict, spec: SimSpec = None) -> str:
    """Return the fix deposit command for the given species."""
    region = _deposit_region(spec) if spec is not None else "bbox"
    if species["is_molecule"]:
        # O2: inject as molecule; type 0 means types come from molecule file
        return (
            f"fix     depo insert deposit 1 0 10000000000 ${{deposeed}} global "
            f"${{i_above}} ${{i_above}} vx 0.0 0.0 vy ${{vely}} ${{vely}} "
            f"vz -${{velz}} -${{velz}} region {region} units box mol O2\n"
        )
    else:
        # single atom: inject by type index
        return (
            f"fix     depo insert deposit 1 ${{incident_type_index}} 10000000000 ${{deposeed}} global "
            f"${{i_above}} ${{i_above}} vx 0.0 0.0 vy ${{vely}} ${{vely}} "
            f"vz -${{velz}} -${{velz}} region {region} units box\n"
        )


def _radical_loop_block(spec: SimSpec) -> str:
    """Return the LAMMPS radical pre-exposure loop for RIE-etch mode.

    Supports 4 sampling modes (use_boltzmann × use_cosine):
    - fixed energy + fixed angle: monoenergetic, fixed direction
    - Boltzmann + fixed angle: Maxwell-Boltzmann speed, fixed direction
    - fixed energy + cosine: fixed speed, Lambert cosine angle distribution
    - Boltzmann + cosine: MB speed + Lambert cosine (full stochastic)

    In stochastic mode (Boltzmann or cosine), per-radical halt time is computed
    as min(radical_i_above / |vz_rad|, max_inter_neutral_time) so slow radicals
    still reach the surface.

    Logs energy and angles of every radical to radical_log.txt.
    Radical etch-event trajectories saved when dump_mode == "all" or "etch_only".
    """
    use_boltzmann = spec.radical_temperature is not None
    use_cosine    = spec.radical_angle_distribution
    use_stochastic = use_boltzmann or use_cosine
    dm = spec.dump_mode
    # Determine whether this sim uses hybrid ZBL (Ar ions) → QEQ group is "nonargon"
    # (Kokkos fix qeq/reaxff doesn't resize its internal arrays when atoms are
    # deposited mid-run, so we must unfix/refix after each deposit to force a
    # full reinitialisation that picks up the new atom.)
    if spec.ion_mix:
        _has_zbl = any(SPECIES[c.species]["needs_zbl"] for c in spec.ion_mix)
    else:
        _has_zbl = SPECIES[spec.species]["needs_zbl"]
    _qeq_group = "nonargon" if _has_zbl else "all"

    blk = (
        f"# ========= Begin RIE-etch O• radical deposition loop =========\n"
        f'if "${{cn_start}} > 0 && ${{cn_start}} < ${{flux_ratio}}" then &\n'
        f'"variable neutral_lp loop $(v_flux_ratio-v_cn_start)" &\n'
        f'elif "${{cn_start}} == ${{flux_ratio}}" &\n'
        f'"jump SELF skip_radicals" &\n'
        f"else &\n"
        f'"variable neutral_lp loop ${{flux_ratio}}"\n'
        f"\n"
        f"label       neutral_loop\n"
        f"# Refresh bbox: each radical can shrink zhi; must update before next deposit\n"
        f"region bbox delete\n"
        f"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_redef(spec)
        + f"variable    cn equal ${{cn}}+1\n"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{cn}}))\n"
        f"group       insert clear\n"
        f"group       mobile subtract all anchor\n"
        f"\n"
        f"# Sample O• radical velocity\n"
    )

    # ── Speed sampling ────────────────────────────────────────────────────────
    if use_boltzmann:
        # 3 Box-Muller pairs → 3 Gaussian velocity component squares → MB speed magnitude.
        # LAMMPS re-evaluates equal-style variables on every reference: v_gx*v_gx calls
        # the gx formula twice and, in some LAMMPS/Kokkos builds, the two random() calls
        # with the same seed return different values, making the product negative and
        # crashing sqrt(gx^2+gy^2+gz^2).
        # Fix: use cos^2(x) = (1+cos(2x))/2 so each random variable appears exactly once,
        # guaranteeing gx_sq >= 0 without any formula-doubling.
        blk += (
            f"variable    sigma_rad equal sqrt(${{kT_rad}}*6.02214129e7/${{M_O}}/6242)/1000\n"
            f"variable    bms equal v_c*100000+v_cn+${{seed_adjust}}*1000000\n"
            # Freeze each draw immediately with $() so every downstream reference
            # sees a literal constant — the bare random() form advances the persistent
            # RNG on every variable evaluation, making logged values inconsistent with
            # the actually deposited velocity.
            f"variable    u1a equal $(random(1e-10,1.0,v_bms+1))\n"
            f"variable    u2a equal $(random(1e-10,1.0,v_bms+2))\n"
            f"variable    u1b equal $(random(1e-10,1.0,v_bms+3))\n"
            f"variable    u2b equal $(random(1e-10,1.0,v_bms+4))\n"
            f"variable    u1c equal $(random(1e-10,1.0,v_bms+5))\n"
            f"variable    u2c equal $(random(1e-10,1.0,v_bms+6))\n"
            f"variable    gx_sq equal (-2*ln(v_u1a))*(1+cos(4*PI*v_u2a))/2*v_sigma_rad*v_sigma_rad\n"
            f"variable    gy_sq equal (-2*ln(v_u1b))*(1+cos(4*PI*v_u2b))/2*v_sigma_rad*v_sigma_rad\n"
            f"variable    gz_sq equal (-2*ln(v_u1c))*(1+cos(4*PI*v_u2c))/2*v_sigma_rad*v_sigma_rad\n"
            f"variable    rad_speed equal sqrt(v_gx_sq+v_gy_sq+v_gz_sq)\n"
        )
    else:
        blk += (
            f"variable    vel_chem equal sqrt(2*${{radical_energy}}*6.02214129*1.0e+7/${{M_O}}/6242)/1000\n"
            f"variable    rad_speed equal v_vel_chem\n"
        )

    # ── Direction sampling ────────────────────────────────────────────────────
    if use_cosine:
        # Lambert cosine: theta = arcsin(sqrt(U)), phi = 2*pi*U2
        # seed offsets 7-8 (Boltzmann) or 1-2 (cosine-only, own seed)
        if use_boltzmann:
            blk += (
                f"variable    u_th equal $(random(0.0,1.0,v_bms+7))\n"
                f"variable    u_ph equal $(random(0.0,1.0,v_bms+8))\n"
            )
        else:
            blk += (
                f"variable    bms_ang equal v_c*100000+v_cn+${{seed_adjust}}*1000000\n"
                f"variable    u_th equal $(random(0.0,1.0,v_bms_ang+1))\n"
                f"variable    u_ph equal $(random(0.0,1.0,v_bms_ang+2))\n"
            )
        blk += (
            f"variable    cos_theta equal sqrt(v_u_th)\n"
            f"variable    sin_theta equal sqrt(1.0-v_u_th)\n"
            f"variable    phi_rad equal 2*PI*v_u_ph\n"
            f"variable    vx_rad equal v_rad_speed*v_sin_theta*cos(v_phi_rad)\n"
            f"variable    vy_rad equal v_rad_speed*v_sin_theta*sin(v_phi_rad)\n"
            f"variable    vz_rad equal -v_rad_speed*v_cos_theta\n"
        )
    else:
        blk += (
            f"variable    vx_rad equal 0.0\n"
            f"variable    vy_rad equal v_rad_speed*sin(${{rad_angl}}*PI/180)\n"
            f"variable    vz_rad equal -v_rad_speed*cos(${{rad_angl}}*PI/180)\n"
        )

    # ── Per-radical halt time ─────────────────────────────────────────────────
    # rad_halt_t = min(1.5*i_above/|vz|, max_inter_neutral_time)
    # i_above/|vz| is the one-way travel time; ×1.5 gives time for surface interaction.
    # |vz| = speed×cos(θ) so the window is automatically longer for glancing radicals.
    # The +1e-10 guards against exact θ=90° (vz=0) without affecting any real case.
    # This is evaluated as a frozen equal-style variable since vz_rad is a frozen constant.

    # ── Logging to radical_log.txt ────────────────────────────────────────────
    # Columns: impact  radical_idx  energy_eV  polar_deg  azimuthal_deg
    # Energy: use gx_sq+gy_sq+gz_sq (= |v|^2) in Boltzmann mode to avoid
    # re-evaluating vx_rad^2+... which re-evaluates the random() chain.
    # Angles: derive from the sampling variables directly rather than from velocity
    # components — computing acos(-vz/|v|) would re-evaluate rad_speed independently
    # for the numerator and denominator, and a tiny discrepancy pushes the ratio
    # outside [-1,1], crashing acos.
    if use_boltzmann:
        rad_spd_sq_formula = "v_gx_sq+v_gy_sq+v_gz_sq"
    else:
        rad_spd_sq_formula = "v_vx_rad*v_vx_rad+v_vy_rad*v_vy_rad+v_vz_rad*v_vz_rad"

    if use_cosine:
        # cos_theta = sqrt(u_th) in [0,1] → acos always valid; phi_rad already computed
        polar_formula   = "acos(v_cos_theta)*180/PI"
        azimuth_formula = "v_phi_rad*180/PI"
    else:
        # fixed angle; azimuth is always 0 (radicals in the yz plane)
        polar_formula   = f"${{rad_angl}}"
        azimuth_formula = "0.0"

    blk += (
        f"variable    rad_spd_sq equal {rad_spd_sq_formula}\n"
        f"variable    rad_energy_ev equal v_rad_spd_sq*${{M_O}}*6242*500000/6.02214129e7\n"
        f"variable    rad_polar_deg equal {polar_formula}\n"
        f"variable    rad_azimuth_deg equal {azimuth_formula}\n"
        f'print       "${{c}} ${{cn}} $(v_rad_energy_ev) $(v_rad_polar_deg) $(v_rad_azimuth_deg)"'
        f" append radical_log.txt\n"
    )

    # ── Fix deposit ───────────────────────────────────────────────────────────
    _rad_region = _deposit_region(spec)
    blk += (
        f"\n"
        f"# Deposit O• radical (type 3) with sampled velocity\n"
        f"fix         depo insert deposit 1 3 1 ${{deposeed}} global "
        f"${{radical_i_above}} ${{radical_i_above}} "
        f"vx ${{vx_rad}} ${{vx_rad}} vy ${{vy_rad}} ${{vy_rad}} vz ${{vz_rad}} ${{vz_rad}} "
        f"region {_rad_region} units box\n"
        f"fix         2 mobile nve\n"
        f"fix         3 insert nve\n"
    )

    # ── Radical dump ("all": always record; "etch_only": record, delete if no etch) ──
    if dm == "all":
        blk += (
            f"dump        current_dump_n all custom 100 "
            f"etch_event_trajs/event_dump_n${{c}}_${{cn}}.dump "
            f"id type x y z vx vy vz fx fy fz q\n"
        )
    elif dm == "etch_only":
        blk += (
            f"variable    keep_dump_n equal 0\n"
            f"dump        current_dump_n all custom 100 "
            f"etch_event_trajs/event_dump_n${{c}}_${{cn}}.dump "
            f"id type x y z vx vy vz fx fy fz q\n"
        )

    # ── Run radical impact ────────────────────────────────────────────────────
    # For ZBL (Ar+) sims, refresh the nonargon group AFTER the deposit so the
    # Kokkos fix qeq/reaxff picks up the new O• atom before the dynamics run
    # (mirrors what nonargon_regroup does for O2+ ions in the outer loop).
    nonargon_refresh_rad = (
        f"group       nonargon type 1 2 3\n" if _has_zbl else ""
    )
    blk += (
        f"\n"
        f"timestep    1e-10\n"
        f"run         1 post no\n"
        f"{nonargon_refresh_rad}"
        f"run         0\n"
        f"fix         ats_n all dt/reset 1 NULL 1.00 0.01 units box\n"
        f"variable    starting_nclusts equal $(c_nclusts)\n"
        f"variable    t0 equal $(time)\n"
        f"variable    time_elapsed equal time-${{t0}}\n"
    )

    # ── Halt time: stochastic uses angle-dependent rad_halt_t; fixed uses inter_neutral_time ──
    if use_stochastic:
        # min() and abs() are not available as pairwise math functions in this build.
        # Use sqrt(x*x) for |x|; cap with an if statement instead of min(a,b).
        # ${} substitution calls the full variable evaluator (sqrt works); the if
        # then overwrites rad_halt_t with the cap if the travel time exceeds it.
        blk += (
            f"variable    rad_halt_t equal 1.5*${{radical_i_above}}/(sqrt(v_vz_rad*v_vz_rad)+1e-10)\n"
            f'if "${{rad_halt_t}} > ${{max_inter_neutral_time}}" then "variable rad_halt_t equal ${{max_inter_neutral_time}}"\n'
            f"fix         thalt all halt 1 v_time_elapsed > ${{rad_halt_t}} "
            f"error continue message yes\n"
        )
        halt_var = "${rad_halt_t}"
    else:
        blk += (
            f"fix         thalt all halt 1 v_time_elapsed > ${{inter_neutral_time}} "
            f"error continue message yes\n"
        )
        halt_var = "${inter_neutral_time}"

    blk += (
        f"\n"
        f"# ===================== Radical inner loop =====================\n"
        f"label       continue_n_impact\n"
        f"run         500 pre no post no\n"
        f"run         0\n"
        f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
        f'"variable event_count equal ${{event_count}}+1" &\n'
        f'"variable starting_nclusts equal $(c_nclusts)"'
        + (' &\n"variable keep_dump_n equal 1"' if dm == "etch_only" else "")
        + "\n"
        + f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
    )

    blk += (
        f'if "$(time-v_t0) < {halt_var}" then "jump SELF continue_n_impact"\n'
    )

    blk += (
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
        f"unfix       ats_n\n"
    )

    if dm == "all":
        blk += f"undump      current_dump_n\n"
    elif dm == "etch_only":
        blk += (
            f'if "${{keep_dump_n}} == 0" then '
            f'"shell rm etch_event_trajs/event_dump_n${{c}}_${{cn}}.dump"\n'
            f"undump      current_dump_n\n"
        )

    blk += (
        f"unfix       depo\n"
    )
    if not spec.skip_radical_thermalization:
        blk += f"# Thermalize after each radical\ninclude     thermalize.lmp\n"
    blk += (
        f"unfix       2\n"
        f"unfix       3\n"
        f"# ===================== End radical inner loop =====================\n"
        f"next        neutral_lp\n"
        f"jump        SELF neutral_loop\n"
        f"\n"
        f"label       skip_radicals\n"
        f"variable    cn_start equal 0\n"
        f"variable    cn equal 0\n"
        f"# ========= End RIE-etch O• radical deposition loop =========\n"
        f"\n"
        f"# Final thermalize before ion impact\n"
        f"include     thermalize.lmp\n"
        f"\n"
        f"# Refresh bbox: thermalize (boundary p p m) may shrink zhi\n"
        f"region bbox delete\n"
        f"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_redef(spec)
        + f"\n"
    )

    return blk


def _radical_burst_block(spec: SimSpec) -> str:
    """Return the LAMMPS burst O• injection block for burst-mode RIE.

    Injects flux_ratio O• atoms in chunks of radical_burst_chunk before each ion
    impact.  All atoms in a chunk are deposited at the same z height:
    bound(all,zmax) + radical_i_above, evaluated once at the start of each chunk.
    A narrow region (±0.1 Å) is used instead of 'global lo hi' so the box never
    grows during deposition — atoms land inside the existing vacuum above the surface.
    Only valid for mono-energetic fixed-angle mode (no Boltzmann, no cosine).

    Output: one ncarbon.txt entry (cn=1) per burst + one write_data snapshot.
    Restart: if cn_start >= 1 the burst is skipped (already complete for this impact).
    Per-chunk thermalization is controlled by skip_radical_thermalization.
    """
    ml         = spec.ml
    auto_chunk = max(1, round(0.5 * ml))
    chunk_size = spec.radical_burst_chunk if spec.radical_burst_chunk > 0 else auto_chunk
    total      = max(1, spec.flux_ratio)
    n_full     = total // chunk_size
    remainder  = total % chunk_size
    chunks     = [chunk_size] * n_full + ([remainder] if remainder > 0 else [])

    # ZBL (Ar ions) → must refresh nonargon group after each deposit
    if spec.ion_mix:
        _has_zbl = any(SPECIES[c.species]["needs_zbl"] for c in spec.ion_mix)
    else:
        _has_zbl = SPECIES[spec.species]["needs_zbl"]
    nonargon_refresh = f"group       nonargon type 1 2 3\n" if _has_zbl else ""

    dm        = spec.dump_mode
    dump_cols = "id type x y z vx vy vz fx fy fz q"

    blk = (
        f"# ========= Begin radical burst deposition"
        f" ({len(chunks)} chunk(s) × ≤{chunk_size} atoms = {total} total) =========\n"
        f'if "${{cn_start}} >= 1" then "jump SELF skip_burst"\n'
        f"\n"
        f"# Fixed velocity for all burst atoms\n"
        f"variable    vel_chem_burst equal sqrt(2*${{radical_energy}}*6.02214129*1.0e+7/${{M_O}}/6242)/1000\n"
        f"variable    vx_burst equal 0.0\n"
        f"variable    vy_burst equal v_vel_chem_burst*sin(${{rad_angl}}*PI/180)\n"
        f"variable    vz_burst equal -v_vel_chem_burst*cos(${{rad_angl}}*PI/180)\n"
        f"timestep    1e-10\n"
        f"\n"
        f"# Record pre-burst O count to detect incomplete placement\n"
        f"group       oxygen_pre_burst type 3\n"
        f"variable    n_oxy_pre equal $(count(oxygen_pre_burst))\n"
        f"group       oxygen_pre_burst delete\n"
        f"\n"
    )

    for ci, csize in enumerate(chunks):
        dump_file = f"etch_event_trajs/event_dump_burst_${{c}}_{ci}.dump"

        # Per-chunk dump open / etch-event / dump close
        if dm == "none":
            chunk_dump_open  = ""
            chunk_dump_close = ""
            etch_event_line  = (
                f'if "$(c_nclusts) > ${{burst_nclusts0}}" then &\n'
                f'"variable event_count equal ${{event_count}}+1" &\n'
                f'"variable burst_nclusts0 equal $(c_nclusts)"\n'
            )
        elif dm == "all":
            chunk_dump_open  = f"dump        current_dump_burst all custom 100 {dump_file} {dump_cols}\n"
            chunk_dump_close = f"undump      current_dump_burst\n"
            etch_event_line  = (
                f'if "$(c_nclusts) > ${{burst_nclusts0}}" then &\n'
                f'"variable event_count equal ${{event_count}}+1" &\n'
                f'"variable burst_nclusts0 equal $(c_nclusts)"\n'
            )
        else:  # etch_only
            chunk_dump_open  = (
                f"variable    keep_dump_burst equal 0\n"
                f"dump        current_dump_burst all custom 100 {dump_file} {dump_cols}\n"
            )
            chunk_dump_close = (
                f'if "${{keep_dump_burst}} == 0" then "shell rm {dump_file}"\n'
                f"undump      current_dump_burst\n"
            )
            etch_event_line  = (
                f'if "$(c_nclusts) > ${{burst_nclusts0}}" then &\n'
                f'"variable event_count equal ${{event_count}}+1" &\n'
                f'"variable burst_nclusts0 equal $(c_nclusts)" &\n'
                f'"variable keep_dump_burst equal 1"\n'
            )

        # ── All atoms in chunk: narrow z-region at bound(all,zmax)+radical_i_above ──
        # z_ins{ci} is captured once per chunk; all csize atoms land at the same z.
        # Unlike 'global', 'region' placement requires the region to already be inside
        # the simulation box.  With p p m shrink-wrap, zhi ≈ zmax + skin, so bzone
        # (at zmax + radical_i_above) can sit above zhi.  Expand the box first if needed.
        blk += (
            f"# --- Burst chunk {ci+1}/{len(chunks)}: {csize} atoms (deposit phase) ---\n"
            f"if \"$(bound(all,zmax)+v_radical_i_above+2.0) > $(zhi)\" then "
            f"\"change_box all z delta 0 $(bound(all,zmax)+v_radical_i_above+2.0-zhi) units box\" "
            f"\"region bbox delete\" "
            f"\"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\" "
            + _expose_zone_redef_if_cmds(spec)
            + f"\n"
            f"variable    z_ins{ci} equal bound(all,zmax)+${{radical_i_above}}\n"
            + _bzone_def(spec, f"bzone{ci}", f"$(v_z_ins{ci} - 0.1)", f"$(v_z_ins{ci} + 0.1)")
            + f"group       insert clear\n"
            f"group       mobile subtract all anchor\n"
            f"variable    burst_seed{ci}_1 equal "
            f"floor(random(1,72099+${{seed_adjust}},${{c}}*100000+{ci}*10000+{ci+1}))\n"
            f"fix         burst_depo insert deposit 1 3 1 ${{burst_seed{ci}_1}} "
            f"attempt {spec.radical_burst_attempt} "
            f"vx ${{vx_burst}} ${{vx_burst}} "
            f"vy ${{vy_burst}} ${{vy_burst}} "
            f"vz ${{vz_burst}} ${{vz_burst}} "
            f"region bzone{ci} near 2.0\n"
            f"fix         2 mobile nve\n"
            f"fix         3 insert nve\n"
            f"run         1 post no\n"
            f"run         0\n"
            f"{nonargon_refresh}"
            f"unfix       burst_depo\n"
            f"unfix       2\n"
            f"unfix       3\n"
        )
        # ── Atoms 2..csize: same bzone region, same z ─────────────────────────────
        if csize > 1:
            blk += (
                f"variable    burst_lp{ci} loop {csize - 1}\n"
                f"label       burst_dep_{ci}\n"
                f"group       insert clear\n"
                f"group       mobile subtract all anchor\n"
                f"variable    burst_seed{ci} equal "
                f"floor(random(1,72099+${{seed_adjust}},${{c}}*100000+{ci}*10000+v_burst_lp{ci}))\n"
                f"fix         burst_depo insert deposit 1 3 1 ${{burst_seed{ci}}} "
                f"attempt {spec.radical_burst_attempt} "
                f"vx ${{vx_burst}} ${{vx_burst}} "
                f"vy ${{vy_burst}} ${{vy_burst}} "
                f"vz ${{vz_burst}} ${{vz_burst}} "
                f"region bzone{ci} near 2.0\n"
                f"fix         2 mobile nve\n"
                f"fix         3 insert nve\n"
                f"run         1 post no\n"
                f"run         0\n"
                f"{nonargon_refresh}"
                f"unfix       burst_depo\n"
                f"unfix       2\n"
                f"unfix       3\n"
                f"next        burst_lp{ci}\n"
                f"jump        SELF burst_dep_{ci}\n"
            )
        blk += (
            _bzone_cleanup(spec, f"bzone{ci}")
            + f"variable    z_ins{ci} delete\n"
        )
        # ── Dynamics phase ───────────────────────────────────────────────────────
        blk += (
            f"\n"
            f"# --- Burst chunk {ci+1}/{len(chunks)}: dynamics phase ---\n"
            f"# All {csize} deposited atoms are now in mobile; run dynamics together\n"
            f"group       insert clear\n"
            f"group       mobile subtract all anchor\n"
            f"fix         2 mobile nve\n"
            f"fix         3 insert nve\n"
            f"{chunk_dump_open}"
            f"fix         ats_burst all dt/reset 1 NULL 1.00 0.01 units box\n"
            f"variable    t0_burst equal $(time)\n"
            f"variable    burst_elapsed equal time-${{t0_burst}}\n"
            f"variable    burst_nclusts0 equal $(c_nclusts)\n"
            f"fix         burst_thalt all halt 1 v_burst_elapsed > ${{inter_neutral_time}} "
            f"error continue message yes\n"
            f"run         0\n"
            f"label       burst_inner_{ci}\n"
            f"run         500 pre no post no\n"
            f"run         0\n"
            f"{etch_event_line}"
            f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
            f'if "$(time-v_t0_burst) < ${{inter_neutral_time}}" then "jump SELF burst_inner_{ci}"\n'
            f"unfix       burst_thalt\n"
            f"unfix       ats_burst\n"
            f"{chunk_dump_close}"
        )
        if not spec.skip_radical_thermalization:
            blk += f"include     thermalize.lmp\n"
        blk += (
            f"unfix       2\n"
            f"unfix       3\n"
            f"timestep    1e-10\n"
            f"\n"
        )

    # ── Post-chunk: shortfall check + optional top-up ────────────────────────
    blk += (
        f"\n"
        f"group       carbon type 1\n"
        f"group       hydrogen type 2\n"
        f"group       oxygen type 3\n"
        f"variable    ncarbon equal count(carbon)\n"
        f"variable    nhydrogen equal count(hydrogen)\n"
        f"variable    noxygen equal count(oxygen)\n"
        f"\n"
        f"# Shortfall check: how many O atoms were actually placed?\n"
        f"variable    n_placed_burst equal count(oxygen)-v_n_oxy_pre\n"
        f"variable    shortfall_burst equal {total}-v_n_placed_burst\n"
        f'if "${{shortfall_burst}} == 0" then "jump SELF burst_topup_done"\n'
        f"\n"
        f"# Record incomplete burst (atom count, not halting)\n"
        f'print       "c=${{c}} placed=$(v_n_placed_burst) expected={total} shortfall=$(v_shortfall_burst)" '
        f"append BURST_INCOMPLETE\n"
        f"\n"
        f"# Top-up: deposit missing atoms with higher attempt count\n"
        f"if \"$(bound(all,zmax)+v_radical_i_above+2.0) > $(zhi)\" then "
        f"\"change_box all z delta 0 $(bound(all,zmax)+v_radical_i_above+2.0-zhi) units box\" "
        f"\"region bbox delete\" "
        f"\"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\" "
        + _expose_zone_redef_if_cmds(spec)
        + f"\n"
        f"variable    z_topup equal bound(all,zmax)+${{radical_i_above}}\n"
        + _bzone_def(spec, "bzone_topup", "$(v_z_topup - 0.1)", "$(v_z_topup + 0.1)")
        + f"variable    topup_lp loop $(v_shortfall_burst)\n"
        f"label       burst_topup_loop\n"
        f"group       insert clear\n"
        f"group       mobile subtract all anchor\n"
        f"variable    topup_seed equal "
        f"floor(random(1,72099+${{seed_adjust}},${{c}}*100000+99999+v_topup_lp))\n"
        f"fix         burst_depo insert deposit 1 3 1 ${{topup_seed}} "
        f"attempt 1000 "
        f"vx ${{vx_burst}} ${{vx_burst}} "
        f"vy ${{vy_burst}} ${{vy_burst}} "
        f"vz ${{vz_burst}} ${{vz_burst}} "
        f"region bzone_topup near 2.0\n"
        f"fix         2 mobile nve\n"
        f"fix         3 insert nve\n"
        f"run         1 post no\n"
        f"run         0\n"
        f"{nonargon_refresh}"
        f"unfix       burst_depo\n"
        f"unfix       2\n"
        f"unfix       3\n"
        f"next        topup_lp\n"
        f"jump        SELF burst_topup_loop\n"
        + _bzone_cleanup(spec, "bzone_topup")
        + f"variable    z_topup delete\n"
        f"\n"
        f"# Top-up dynamics: run deposited atoms until inter_neutral_time\n"
        f"group       insert clear\n"
        f"group       mobile subtract all anchor\n"
        f"fix         2 mobile nve\n"
        f"fix         3 insert nve\n"
        f"fix         ats_topup all dt/reset 1 NULL 1.00 0.01 units box\n"
        f"variable    t0_topup equal $(time)\n"
        f"variable    topup_elapsed equal time-${{t0_topup}}\n"
        f"variable    burst_nclusts0 equal $(c_nclusts)\n"
        f"fix         topup_thalt all halt 1 v_topup_elapsed > ${{inter_neutral_time}} "
        f"error continue message yes\n"
        f"run         0\n"
        f"label       burst_topup_inner\n"
        f"run         500 pre no post no\n"
        f"run         0\n"
        f'if "$(c_nclusts) > ${{burst_nclusts0}}" then &\n'
        f'"variable event_count equal ${{event_count}}+1" &\n'
        f'"variable burst_nclusts0 equal $(c_nclusts)"\n'
        f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
        f'if "$(time-v_t0_topup) < ${{inter_neutral_time}}" then "jump SELF burst_topup_inner"\n'
        f"unfix       topup_thalt\n"
        f"unfix       ats_topup\n"
    )
    if not spec.skip_radical_thermalization:
        blk += f"include     thermalize.lmp\n"
    blk += (
        f"unfix       2\n"
        f"unfix       3\n"
        f"timestep    1e-10\n"
        f"\n"
        f"label       burst_topup_done\n"
        f"\n"
    )

    blk += (
        f'print       "Burst complete"\n'
        f'print       "C_COUNT_burst: ${{ncarbon}}"\n'
        f'print       "${{c}} 1 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        f"write_data  impact_snaps/${{c}}_1.data nofix nocoeff\n"
        f"\n"
        f"label       skip_burst\n"
        f"variable    cn_start equal 0\n"
        f"variable    cn equal 0\n"
        f"# ========= End radical burst deposition =========\n"
        f"\n"
        f"# Final thermalize before ion impact\n"
        f"include     thermalize.lmp\n"
        f"\n"
        f"# Refresh bbox: thermalize (boundary p p m) may shrink zhi\n"
        f"region bbox delete\n"
        f"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_redef(spec)
        + f"\n"
    )

    return blk


def _build_ion_dump_blocks(spec: SimSpec, is_carbon_etch: bool = False):
    """Return (ion_dump_open, etch_event_block, channeling_block, ion_dump_close)
    strings for get_head_lmp / get_head_lmp_multi_ion based on dump_mode."""
    dm = spec.dump_mode
    # Carbon-etch: threshold is 2 Å below the lowest anchor atom (set at startup as
    # channeling_z), so anchor atoms never false-trigger the channeling check.
    channeling_region_cmd = (
        f"region channelled block INF INF INF INF INF ${{channeling_z}} units box"
        if is_carbon_etch
        else f"region channelled block INF INF INF INF INF ${{bottom}} units lattice"
    )
    notify_cmd = (
        f"\"print 'Impact ${{c}}: atom channeled below anchor region.' append ATOM_CHANNELED\""
        if is_carbon_etch
        else f'"include notify_channeled.lmp"'
    )
    dump_file = f"etch_event_trajs/event_dump_${{c}}.dump"
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
            f'"region channelled delete" &\n'
            f"\n"
            f'if "${{n_channelled}} > 0" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"delete_atoms group channelled_group" &\n'
            f'"run 0" &\n'
            f'"group channelled_group delete" &\n'
            f'"variable n_channelled equal 0" &\n'
            f"{notify_cmd}\n"
        )

    elif dm == "all":
        ion_dump_open  = f"dump current_dump all custom 100 {dump_file} {dump_cols}\n"
        ion_dump_close = f"undump current_dump\n"
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
            f'"region channelled delete" &\n'
            f"\n"
            f'if "${{n_channelled}} > 0" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"delete_atoms group channelled_group" &\n'
            f'"run 0" &\n'
            f'"group channelled_group delete" &\n'
            f'"variable n_channelled equal 0" &\n'
            f"{notify_cmd}\n"
        )

    else:  # etch_only
        ion_dump_open = (
            f"variable    keep_dump equal 0\n"
            f"dump current_dump all custom 100 {dump_file} {dump_cols}\n"
        )
        ion_dump_close = (
            f'if "${{keep_dump}} == 0" then "shell rm {dump_file}"\n'
            f"undump current_dump\n"
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
            f'"region channelled delete" &\n'
            f"\n"
            f'if "${{n_channelled}} > 0" then &\n'
            f'"variable event_count equal ${{event_count}}+1" &\n'
            f'"delete_atoms group channelled_group" &\n'
            f'"run 0" &\n'
            f'"group channelled_group delete" &\n'
            f'"variable n_channelled equal 0" &\n'
            f'"variable keep_dump equal 1" &\n'
            f"{notify_cmd}\n"
        )

    return ion_dump_open, etch_event_block, channeling_block, ion_dump_close


def get_head_lmp(spec: SimSpec) -> str:
    """Generate the contents of head.lmp for the given SimSpec.

    Supports ion-etch (flux_ratio == 0) and RIE-etch (flux_ratio > 0).
    """
    cfg = ORIENT[spec.orientation]
    lattice_cmd = cfg["lattice_cmd"]
    bottom_expr = cfg["bottom_expr"]
    species = SPECIES[spec.species]
    is_burst     = spec.flux_ratio > 0 and spec.radical_burst
    is_rie       = spec.flux_ratio > 0 and not spec.radical_burst
    has_radicals = spec.flux_ratio > 0

    # O2 molecule declaration (before the loop)
    molecule_decl = ""
    if species["is_molecule"]:
        molecule_decl = f"molecule O2 O2.molecule\n"

    # Ar: re-establish nonargon group at top of each loop iteration
    nonargon_regroup = ""
    if species["needs_zbl"]:
        nonargon_regroup = f"group nonargon type 1 2 3\n"

    # RIE/burst: cn (radical counter) restart variables before the loop
    rie_pre_loop = ""
    if has_radicals:
        rie_pre_loop = (
            f"variable    cn_start equal ${{neut_complete}}\n"
            f"variable    cn equal 0\n"
        )

    # RIE/burst: cn_start reset at top of loop iteration
    rie_loop_top = ""
    if has_radicals:
        rie_loop_top = (
            f'if "${{cn_start}} > 0" then "variable cn equal ${{cn_start}}" else "variable cn equal 0"\n'
            f"\n"
        )

    # ncarbon.txt output format: 5-col for RIE/burst, 4-col for ion-etch
    if has_radicals:
        ncarbon_print = f'print "${{c}} 0 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}_0.data nofix nocoeff\n"
    else:
        ncarbon_print = f'print "${{c}} ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}.data nofix nocoeff\n"

    # Radical/burst block (inserted before the ion deposit section)
    if is_burst:
        radical_loop = _radical_burst_block(spec)
    elif is_rie:
        radical_loop = _radical_loop_block(spec)
    else:
        radical_loop = ""

    loop_counter_line = f"variable    c equal ${{c}}+1\n"

    ion_dump_open, etch_event_block, channeling_block, ion_dump_close = (
        _build_ion_dump_blocks(spec)
    )

    return (
        f"# head.lmp — generated by DiamondEtchMD\n"
        f"# orientation={spec.orientation}  species={spec.species}"
        f"  {spec.energy}eV  {spec.surface_temperature}K  ion_angle={spec.ion_angle}deg\n"
        f"package kokkos neigh/qeq full neigh half newton on\n"
        f"units\t\treal\n"
        f"include     config.lmp\n"
        f'if "${{pot}} == COMB3" then &\n'
        f'"atom_style charge" &\n'
        f'elif "${{pot}} == REAX" &\n'
        f'"atom_style charge" &\n'
        f"else &\n"
        f'"atom_style atomic"\n'
        f"\n"
        f"boundary\tp p m\n"
        f"\n"
        f"# create_box with 4 types then merge data file so type 4 (Ar) is available\n"
        f"region      dummy_region block 0 1 0 1 0 1\n"
        f"create_box  4 dummy_region\n"
        f"read_data   ${{data_file}} add merge\n"
        f"\n"
        f"variable \tlp equal $(v_end_fluence*v_ML)-${{n_complete}} # impacts left\n"
        f"variable    a loop ${{lp}}\n"
        f"variable    bottom equal {bottom_expr}\n"
        f"print       'BOTTOM ${{bottom}}'\n"
        f"print       'N_LAT_0 ${{n_lat_0}}'\n"
        f"\n"
        f"# Regions — lattice must match make_surf.lmp for this orientation\n"
        f"{lattice_cmd}\n"
        f"region          bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_def(spec)
        + f"variable        sublat equal ${{bottom}}+1/2\n"
        f"region          anchor block INF INF INF INF ${{bottom}} ${{sublat}} units lattice\n"
        f"\n"
        f"# Groups\n"
        f"group \tanchor region anchor\n"
        + _freeze_mask_block(spec)
        + f"group\tinsert empty\n"
        f"group \tmobile subtract all anchor\n"
        f"group   carbon type 1\n"
        f"\n"
        f"# Particle masses (4 types: C, H, O, {spec.species})\n"
        f"mass        1 ${{M_C}}\n"
        f"mass        2 ${{M_H}}\n"
        f"mass        3 ${{M_O}}\n"
        f"mass        4 ${{{species['mass_var']}}}\n"
        f"\n"
        f"# Potential\n"
        f"{_potential_block(species)}"
        f"\n"
        f"# Pre-run counters\n"
        f"variable c equal ${{n_complete}}\n"
        f"variable event_count equal ${{n_events}}\n"
        f"{rie_pre_loop or 'variable    cn equal 0\n'}"
        f"\n"
        f"# Atom counts\n"
        f"variable\tnfixed   equal count(anchor)\n"
        f"variable\tnmobile  equal count(mobile)\n"
        f"variable \tninject  equal count(insert)\n"
        f"\n"
        f"# Incident particle velocity components\n"
        f"variable \tvel equal sqrt(2*${{energ}}*6.02214129*1.0e+7/${{M_incident}}/6242)/1000\n"
        f"variable \tvelz equal cos(${{ion_angl}}*PI/180)*${{vel}}\n"
        f"variable \tvely equal sin(${{ion_angl}}*PI/180)*${{vel}}\n"
        f"\n"
        f"# Energy conservation check\n"
        f"compute     ake all ke\n"
        f"compute     ape all pe\n"
        f"compute     ike insert ke\n"
        f"variable    ate equal c_ake+c_ape+ecouple\n"
        f"\n"
        f"# Clustering\n"
        f"variable\tcheckevery equal 1000\n"
        f"compute\t    clusts all cluster/atom 3.0\n"
        f"compute\t    clust_min all reduce min c_clusts\n"
        f"compute\t    clust_max all reduce max c_clusts\n"
        f"variable    one_clust equal \"c_clust_min == c_clust_max\"\n"
        f"fix\t\t    stopclust all halt ${{checkevery}} v_one_clust == 0 error continue message yes\n"
        f"\n"
        f"# Carbon cluster event tracking\n"
        f"compute     cclusts carbon cluster/atom 1.8\n"
        f"compute     cc1 carbon chunk/atom c_cclusts compress yes\n"
        f"compute     nclusts carbon reduce max c_cc1\n"
        f"\n"
        f"# Thermo\n"
        f"thermo_style    custom step time v_ninject temp c_ike dt ecouple v_ate v_one_clust c_nclusts\n"
        f"compute     \tmtemp mobile temp\n"
        f"compute_modify  mtemp dynamic/dof yes\n"
        f"thermo_modify   temp mtemp\n"
        f"thermo_modify   lost warn flush yes\n"
        f"\n"
        f'print "${{c}}" file begin.txt\n'
        f"{molecule_decl}"
        f"\n"
        f"# ========================= Begin Per-Impact Outer Loop =========================\n"
        f"label\t\tloop\n"
        f"# Refresh bbox after thermalize shrinks the box (prevents stale-region errors)\n"
        f"region bbox delete\n"
        f"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_redef(spec)
        + f"{rie_loop_top}"
        f"\n"
        f"group       insert clear\n"
        f"group \t    mobile subtract all anchor\n"
        f"{nonargon_regroup}"
        f"\n"
        f"{radical_loop}"
        f"{loop_counter_line}"
        f"{ion_dump_open}"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{c}}))\n"
        f"{_deposit_line(species, spec)}"
        f"\n"
        f"fix     2 mobile nve\n"
        f"fix     3 insert nve\n"
        f"\n"
        f"# Run 1 fs timestep to place incident particle\n"
        f"timestep 1e-10\n"
        f"run 1 post no\n"
        f"variable starting_nclusts equal $(c_nclusts)\n"
        f"\n"
        f"# Adaptive timestep\n"
        f"fix ats all dt/reset 1 NULL 1.00 0.01 units box\n"
        f"thermo      100\n"
        f"\n"
        f"variable    t0 equal $(time)\n"
        f"variable    time_elapsed equal time-${{t0}}\n"
        f"fix         thalt all halt 1 v_time_elapsed > ${{impact_time}} error continue message yes\n"
        f"\n"
        f"run         0 post no\n"
        f"variable    n_channelled equal 0\n"
        f"# =========================== begin inner loop ===========================\n"
        f"label\t\tcontinue_impact\n"
        f"\n"
        f"run         1000000000 pre no post no\n"
        f"run         0\n"
        f"{etch_event_block}"
        f"{channeling_block}"
        f"\n"
        f'if "$(time-v_t0) < ${{impact_time}}" then "jump SELF continue_impact"\n'
        f"unfix thalt\n"
        f"# ============================ end inner loop ============================\n"
        f"\n"
        f"unfix   depo\n"
        f"unfix   ats\n"
        f"\n"
        f"# Thermalize\n"
        f"include thermalize.lmp\n"
        f"\n"
        f"unfix       2\n"
        f"unfix       3\n"
        f"\n"
        + (
        f"# Remove implanted {spec.species} (does not bond; would accumulate across impacts)\n"
        f"group       incident_ion type 4\n"
        f"delete_atoms group incident_ion\n"
        f"group       incident_ion delete\n"
        f"\n"
        if species["needs_zbl"] else ""
        ) +
        f"# Replenish carbon if etched below baseline\n"
        f"group       carbon type 1\n"
        f"group       hydrogen type 2\n"
        f"group       oxygen type 3\n"
        f"variable    ncarbon equal count(carbon)\n"
        f"variable    nhydrogen equal count(hydrogen)\n"
        f"variable    noxygen equal count(oxygen)\n"
        f"label addfix\n"
        f'if "${{ncarbon}}<${{n_lat_0}}" then &\n'
        f'"include addfix.lmp" &\n'
        f'"fix 2 mobile nve" &\n'
        f'"include thermalize.lmp" &\n'
        f'"unfix 2"\n'
        f'if "${{ncarbon}}<${{n_lat_0}}" then "jump SELF addfix"\n'
        f"\n"
        f'print "Run ${{c}} complete"\n'
        f'print "C_COUNT: ${{ncarbon}}"\n'
        f"{ncarbon_print}"
        f"\n"
        f"{write_data}"
        f"if '$(v_c%v_ML) == 0' then \"write_dump all custom ML_impacts.dump id type q x y z vx vy vz modify sort id append yes\"\n"
        f"\n"
        f"{ion_dump_close}"
        f"# ========================= End Per-Impact Outer Loop =========================\n"
        f"next\t\ta\n"
        f"jump\t\tSELF loop\n"
    )


# ── Multi-ion helpers ─────────────────────────────────────────────────────────


def _multi_ion_select_block(spec: SimSpec) -> str:
    """Generate the per-impact stochastic ion selection block for multi-ion runs.

    Draws a uniform random number r and jumps to the matching ion label based on
    cumulative fraction thresholds.  Each label sets cur_ion (string), energ,
    M_incident, incident_type_index, and cur_ion_is_mol, then jumps to a common
    'ion_sel_done' label where vel/velz/vely are recomputed.  The selected ion
    name is also logged to ion_impacts.txt for per-impact analysis.

    Seed formula: ${c}+50000+${seed_adjust} — independent from deposeed (which
    uses seed=${c}).
    """
    mix = spec.ion_mix
    total = sum(c.fraction for c in mix)

    lines = [
        "# ── Stochastic ion selection ─────────────────────────────────────────────────\n",
        "variable    r equal random(0,1,${c}+50000+${seed_adjust})\n",
        "\n",
    ]

    # Jump table: cumulative thresholds for all ions except the last (fallback)
    cumulative = 0.0
    for i, comp in enumerate(mix[:-1]):
        cumulative += comp.fraction / total
        lines.append(
            f'if "${{r}} < {cumulative:.8f}" then "jump SELF ion_sel_{i}"\n'
        )
    lines.append(f"jump        SELF ion_sel_{len(mix) - 1}\n")
    lines.append("\n")

    # Per-ion setup labels
    for i, comp in enumerate(mix):
        sp = SPECIES[comp.species]
        energy_per_atom = comp.energy / sp["energy_divisor"]
        mass_var        = sp["mass_var"]
        type_idx        = sp["type_index"]
        is_mol          = 1 if sp["is_molecule"] else 0

        lines.append(f"label       ion_sel_{i}\n")
        lines.append(f"variable    cur_ion string {comp.species}\n")
        lines.append(f"variable    energ equal {energy_per_atom}\n")
        lines.append(f"variable    M_incident equal ${{{mass_var}}}\n")
        lines.append(f"variable    incident_type_index equal {type_idx}\n")
        lines.append(f"variable    cur_ion_is_mol equal {is_mol}\n")
        lines.append(f"jump        SELF ion_sel_done\n")
        lines.append("\n")

    lines.extend([
        "label       ion_sel_done\n",
        'print       "${c} ${cur_ion}" append ion_impacts.txt\n',
        "variable    vel equal sqrt(2*${energ}*6.02214129*1.0e+7/${M_incident}/6242)/1000\n",
        "variable    velz equal cos(${ion_angl}*PI/180)*${vel}\n",
        "variable    vely equal sin(${ion_angl}*PI/180)*${vel}\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "\n",
    ])

    return "".join(lines)


def _multi_ion_deposit_line(spec: SimSpec = None) -> str:
    """Return a conditional fix deposit that handles both atom and molecule ions."""
    region = _deposit_region(spec) if spec is not None else "bbox"
    return (
        f'if "${{cur_ion_is_mol}} == 1" then &\n'
        f'"fix     depo insert deposit 1 0 10000000000 ${{deposeed}} global '
        f'${{i_above}} ${{i_above}} vx 0.0 0.0 vy ${{vely}} ${{vely}} '
        f'vz -${{velz}} -${{velz}} region {region} units box mol O2" &\n'
        f"else &\n"
        f'"fix     depo insert deposit 1 ${{incident_type_index}} 10000000000 ${{deposeed}} global '
        f'${{i_above}} ${{i_above}} vx 0.0 0.0 vy ${{vely}} ${{vely}} '
        f'vz -${{velz}} -${{velz}} region {region} units box"\n'
    )


def get_head_lmp_multi_ion(spec: SimSpec) -> str:
    """Generate head.lmp for a multi-ion (ion_mix) SimSpec.

    Structurally identical to get_head_lmp() except:
    - Potential block uses ZBL hybrid if any ion in the mix needs it.
    - Molecule declarations are emitted for every molecule species in the mix.
    - vel/velz/vely are NOT pre-computed; they are set per-impact by the ion
      selection block (_multi_ion_select_block) which runs after deposeed is set.
    - fix deposit is conditional on cur_ion_is_mol (set by the selection block).
    - Ar removal block guards with a count check (Ar may not be deposited this
      impact if a non-Ar ion was selected).
    - ion_impacts.txt is written once per impact with: ${c} ${cur_ion}.
    """
    cfg = ORIENT[spec.orientation]
    lattice_cmd  = cfg["lattice_cmd"]
    bottom_expr  = cfg["bottom_expr"]

    mix     = spec.ion_mix
    has_zbl      = any(SPECIES[c.species]["needs_zbl"]   for c in mix)
    has_molecule = any(SPECIES[c.species]["is_molecule"] for c in mix)
    is_burst     = spec.flux_ratio > 0 and spec.radical_burst
    is_rie       = spec.flux_ratio > 0 and not spec.radical_burst
    has_radicals = spec.flux_ratio > 0

    # Potential block — ZBL mix needs hybrid; O/O2 mix uses plain ReaxFF
    zbl_sp    = next((SPECIES[c.species] for c in mix if SPECIES[c.species]["needs_zbl"]), None)
    potential = _potential_block(zbl_sp if has_zbl else SPECIES["O"])
    mass4_var = zbl_sp["mass_var"] if has_zbl else "M_Ar"

    molecule_decl   = "molecule O2 O2.molecule\n" if has_molecule else ""
    nonargon_regroup = "group nonargon type 1 2 3\n" if has_zbl else ""

    rie_pre_loop = ""
    rie_loop_top = ""
    if has_radicals:
        rie_pre_loop = (
            f"variable    cn_start equal ${{neut_complete}}\n"
            f"variable    cn equal 0\n"
        )
        rie_loop_top = (
            f'if "${{cn_start}} > 0" then "variable cn equal ${{cn_start}}" else "variable cn equal 0"\n'
            f"\n"
        )

    if has_radicals:
        ncarbon_print = f'print "${{c}} 0 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}_0.data nofix nocoeff\n"
    else:
        ncarbon_print = f'print "${{c}} ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}.data nofix nocoeff\n"

    if is_burst:
        radical_loop = _radical_burst_block(spec)
    elif is_rie:
        radical_loop = _radical_loop_block(spec)
    else:
        radical_loop = ""
    loop_counter_line = f"variable    c equal ${{c}}+1\n"

    ion_dump_open, etch_event_block, channeling_block, ion_dump_close = (
        _build_ion_dump_blocks(spec)
    )

    ion_label = "_".join(c.species for c in mix)

    return (
        f"# head.lmp — generated by DiamondEtchMD (multi-ion)\n"
        f"# orientation={spec.orientation}  "
        f"ions=[{', '.join(f'{c.species}@{c.energy}eV×{c.fraction:.0%}' for c in mix)}]\n"
        f"# T={spec.surface_temperature}K  ion_angle={spec.ion_angle}deg\n"
        f"package kokkos neigh/qeq full neigh half newton on\n"
        f"units\t\treal\n"
        f"include     config.lmp\n"
        f'if "${{pot}} == COMB3" then &\n'
        f'"atom_style charge" &\n'
        f'elif "${{pot}} == REAX" &\n'
        f'"atom_style charge" &\n'
        f"else &\n"
        f'"atom_style atomic"\n'
        f"\n"
        f"boundary\tp p m\n"
        f"\n"
        f"region      dummy_region block 0 1 0 1 0 1\n"
        f"create_box  4 dummy_region\n"
        f"read_data   ${{data_file}} add merge\n"
        f"\n"
        f"variable \tlp equal $(v_end_fluence*v_ML)-${{n_complete}}\n"
        f"variable    a loop ${{lp}}\n"
        f"variable    bottom equal {bottom_expr}\n"
        f"print       'BOTTOM ${{bottom}}'\n"
        f"print       'N_LAT_0 ${{n_lat_0}}'\n"
        f"\n"
        f"{lattice_cmd}\n"
        f"region          bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_def(spec)
        + f"variable        sublat equal ${{bottom}}+1/2\n"
        f"region          anchor block INF INF INF INF ${{bottom}} ${{sublat}} units lattice\n"
        f"\n"
        f"group \tanchor region anchor\n"
        + _freeze_mask_block(spec)
        + f"group\tinsert empty\n"
        f"group \tmobile subtract all anchor\n"
        f"group   carbon type 1\n"
        f"\n"
        f"mass        1 ${{M_C}}\n"
        f"mass        2 ${{M_H}}\n"
        f"mass        3 ${{M_O}}\n"
        f"mass        4 ${{{mass4_var}}}\n"
        f"\n"
        f"{potential}"
        f"\n"
        f"variable c equal ${{n_complete}}\n"
        f"variable event_count equal ${{n_events}}\n"
        f"{rie_pre_loop or 'variable    cn equal 0\n'}"
        f"\n"
        f"variable\tnfixed   equal count(anchor)\n"
        f"variable\tnmobile  equal count(mobile)\n"
        f"variable \tninject  equal count(insert)\n"
        f"\n"
        f"compute     ake all ke\n"
        f"compute     ape all pe\n"
        f"compute     ike insert ke\n"
        f"variable    ate equal c_ake+c_ape+ecouple\n"
        f"\n"
        f"variable\tcheckevery equal 1000\n"
        f"compute\t    clusts all cluster/atom 3.0\n"
        f"compute\t    clust_min all reduce min c_clusts\n"
        f"compute\t    clust_max all reduce max c_clusts\n"
        f"variable    one_clust equal \"c_clust_min == c_clust_max\"\n"
        f"fix\t\t    stopclust all halt ${{checkevery}} v_one_clust == 0 error continue message yes\n"
        f"\n"
        f"compute     cclusts carbon cluster/atom 1.8\n"
        f"compute     cc1 carbon chunk/atom c_cclusts compress yes\n"
        f"compute     nclusts carbon reduce max c_cc1\n"
        f"\n"
        f"thermo_style    custom step time v_ninject temp c_ike dt ecouple v_ate v_one_clust c_nclusts\n"
        f"compute     \tmtemp mobile temp\n"
        f"compute_modify  mtemp dynamic/dof yes\n"
        f"thermo_modify   temp mtemp\n"
        f"thermo_modify   lost warn flush yes\n"
        f"\n"
        f'print "${{c}}" file begin.txt\n'
        f"{molecule_decl}"
        f"\n"
        f"# ========================= Begin Per-Impact Outer Loop =========================\n"
        f"label\t\tloop\n"
        f"# Refresh bbox after thermalize shrinks the box (prevents stale-region errors)\n"
        f"region bbox delete\n"
        f"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_redef(spec)
        + f"{rie_loop_top}"
        f"\n"
        f"group       insert clear\n"
        f"group \t    mobile subtract all anchor\n"
        f"{nonargon_regroup}"
        f"\n"
        f"{radical_loop}"
        f"{loop_counter_line}"
        f"{ion_dump_open}"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{c}}))\n"
        f"{_multi_ion_select_block(spec)}"
        f"{_multi_ion_deposit_line(spec)}"
        f"\n"
        f"fix     2 mobile nve\n"
        f"fix     3 insert nve\n"
        f"\n"
        f"timestep 1e-10\n"
        f"run 1 post no\n"
        f"variable starting_nclusts equal $(c_nclusts)\n"
        f"\n"
        f"fix ats all dt/reset 1 NULL 1.00 0.01 units box\n"
        f"thermo      100\n"
        f"\n"
        f"variable    t0 equal $(time)\n"
        f"variable    time_elapsed equal time-${{t0}}\n"
        f"fix         thalt all halt 1 v_time_elapsed > ${{impact_time}} error continue message yes\n"
        f"\n"
        f"run         0 post no\n"
        f"variable    n_channelled equal 0\n"
        f"# =========================== begin inner loop ===========================\n"
        f"label\t\tcontinue_impact\n"
        f"\n"
        f"run         1000000000 pre no post no\n"
        f"run         0\n"
        f"{etch_event_block}"
        f"{channeling_block}"
        f"\n"
        f'if "$(time-v_t0) < ${{impact_time}}" then "jump SELF continue_impact"\n'
        f"unfix thalt\n"
        f"# ============================ end inner loop ============================\n"
        f"\n"
        f"unfix   depo\n"
        f"unfix   ats\n"
        f"\n"
        f"include thermalize.lmp\n"
        f"\n"
        f"unfix       2\n"
        f"unfix       3\n"
        f"\n"
        + (
        f"# Remove implanted ZBL ion (does not bond; would accumulate across impacts)\n"
        f"group       incident_ion type 4\n"
        f"delete_atoms group incident_ion\n"
        f"group       incident_ion delete\n"
        f"\n"
        if has_zbl else ""
        ) +
        f"group       carbon type 1\n"
        f"group       hydrogen type 2\n"
        f"group       oxygen type 3\n"
        f"variable    ncarbon equal count(carbon)\n"
        f"variable    nhydrogen equal count(hydrogen)\n"
        f"variable    noxygen equal count(oxygen)\n"
        f"label addfix\n"
        f'if "${{ncarbon}}<${{n_lat_0}}" then &\n'
        f'"include addfix.lmp" &\n'
        f'"fix 2 mobile nve" &\n'
        f'"include thermalize.lmp" &\n'
        f'"unfix 2"\n'
        f'if "${{ncarbon}}<${{n_lat_0}}" then "jump SELF addfix"\n'
        f"\n"
        f'print "Run ${{c}} complete"\n'
        f'print "C_COUNT: ${{ncarbon}}"\n'
        f"{ncarbon_print}"
        f"\n"
        f"{write_data}"
        f"if '$(v_c%v_ML) == 0' then \"write_dump all custom ML_impacts.dump id type q x y z vx vy vz modify sort id append yes\"\n"
        f"\n"
        f"{ion_dump_close}"
        f"# ========================= End Per-Impact Outer Loop =========================\n"
        f"next\t\ta\n"
        f"jump\t\tSELF loop\n"
    )


# ── Carbon-etch helpers ───────────────────────────────────────────────────────

def _carbon_etch_load_block() -> str:
    """Surface-loading block for carbon-etch.

    builder.py patches the copied data file to declare 4 atom types and adds
    Masses entries for types 2-4, so a plain read_data works here.
    data_file = initial_config.data on first run, latest impact_snaps/*.data on restarts.
    """
    return f"read_data   ${{data_file}}\n"


def _carbon_etch_anchor_block() -> str:
    """Anchor-region definition for carbon-etch (box units, anchor_z_max from config.lmp)."""
    return (
        f"region          bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        f"region          anchor block INF INF INF INF INF ${{anchor_z_max}} units box\n"
    )


def _carbon_etch_initial_therm_block(enable: bool) -> str:
    """Emit a one-time include guard for init_thermalization.lmp.

    The actual content lives in a separate file so it is skipped on restarts
    (n_complete > 0).  builder.py writes init_thermalization.lmp when
    initial_thermalization=True.
    """
    if not enable:
        return ""
    return (
        f'if "${{n_complete}} == 0" then "include init_thermalization.lmp"\n'
        f"\n"
    )


def get_init_thermalization_lmp(steps: int = 10000) -> str:
    """Content of init_thermalization.lmp: CG minimize then NVT equilibration.

    FIRE minimizer is not used — incompatible with Kokkos.
    Anchor atoms are frozen during minimization via setforce 0 0 0.
    """
    return (
        f"# CG minimization before thermalization (first run only)\n"
        f"fix         minimfreeze anchor setforce 0.0 0.0 0.0\n"
        f"min_style   cg\n"
        f"minimize    1.0e-4 1.0e-6 1000 10000\n"
        f"unfix       minimfreeze\n"
        f"\n"
        f"# NVT equilibration\n"
        f"fix         therm_init mobile nvt temp ${{T}} ${{T}} 100.0\n"
        f"run         {steps}\n"
        f"unfix       therm_init\n"
        f"write_data  initial_config_${{T}}K.data\n"
    )


def _carbon_etch_slab_check() -> str:
    """Thin-slab check at top of per-impact loop: jump to end_sim if slab nearly depleted."""
    return (
        f"# Carbon-etch thin-slab stop: halt if fewer than 2x anchor atoms remain\n"
        f"variable    n_total_now equal count(all)\n"
        f'if "${{n_total_now}} < ${{n_anchor_2x}}" then "jump SELF end_sim"\n'
        f"\n"
    )


def _carbon_etch_end_label() -> str:
    """End-of-script label written after next/jump loop; writes SLAB_DEPLETED flag."""
    return (
        f"\n"
        f"label end_sim\n"
        f'print "SLAB_DEPLETED: carbon count below 2x anchor region" file SLAB_DEPLETED\n'
    )


def get_head_lmp_carbon_etch(spec: SimSpec) -> str:
    """Generate head.lmp for carbon-etch + single-species (ion-etch or rie-etch).

    Differences vs get_head_lmp:
    - read_data extra/atom/types 3 instead of create_box + read_data add merge
    - anchor defined by anchor_z_max (A, box units) not lattice units
    - no addfix replenishment block
    - thin-slab stop condition + SLAB_DEPLETED flag at end
    """
    species      = SPECIES[spec.species]
    is_burst     = spec.flux_ratio > 0 and spec.radical_burst
    is_rie       = spec.flux_ratio > 0 and not spec.radical_burst
    has_radicals = spec.flux_ratio > 0

    molecule_decl    = "molecule O2 O2.molecule\n" if species["is_molecule"] else ""
    nonargon_regroup = "group nonargon type 1 2 3\n" if species["needs_zbl"] else ""

    rie_pre_loop = ""
    rie_loop_top = ""
    if has_radicals:
        rie_pre_loop = (
            f"variable    cn_start equal ${{neut_complete}}\n"
            f"variable    cn equal 0\n"
        )
        rie_loop_top = (
            f'if "${{cn_start}} > 0" then "variable cn equal ${{cn_start}}" else "variable cn equal 0"\n'
            f"\n"
        )

    if has_radicals:
        ncarbon_print = f'print "${{c}} 0 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}_0.data nofix nocoeff\n"
    else:
        ncarbon_print = f'print "${{c}} ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}.data nofix nocoeff\n"

    if is_burst:
        radical_loop = _radical_burst_block(spec)
    elif is_rie:
        radical_loop = _radical_loop_block(spec)
    else:
        radical_loop = ""
    loop_counter_line = f"variable    c equal ${{c}}+1\n"

    ion_dump_open, etch_event_block, channeling_block, ion_dump_close = (
        _build_ion_dump_blocks(spec, is_carbon_etch=True)
    )

    return (
        f"# head.lmp - generated by DiamondEtchMD (carbon-etch)\n"
        f"# config_file={spec.initial_config_file}  species={spec.species}"
        f"  {spec.energy}eV  {spec.surface_temperature}K  angle={spec.ion_angle}deg\n"
        f"package kokkos neigh/qeq full neigh half newton on\n"
        f"units\t\treal\n"
        f"include     config.lmp\n"
        f'if "${{pot}} == COMB3" then &\n'
        f'"atom_style charge" &\n'
        f'elif "${{pot}} == REAX" &\n'
        f'"atom_style charge" &\n'
        f"else &\n"
        f'"atom_style atomic"\n'
        f"\n"
        f"boundary\tp p m\n"
        f"\n"
        f"# Load user-supplied config (type 1 = C only); allocate types 2-4 for H/O/Ar\n"
        f"{_carbon_etch_load_block()}"
        f"\n"
        f"variable \tlp equal $(v_end_fluence*v_ML)-${{n_complete}}\n"
        f"variable    a loop ${{lp}}\n"
        f"\n"
        f"# Regions\n"
        f"{_carbon_etch_anchor_block()}"
        + _expose_zone_def(spec)
        + f"\n"
        f"# Groups\n"
        f"group \tanchor region anchor\n"
        + _freeze_mask_block(spec)
        + f"group\tinsert empty\n"
        f"group \tmobile subtract all anchor\n"
        f"group   carbon type 1\n"
        f"\n"
        f"# Thin-slab threshold: 2x frozen atom count\n"
        f"variable    n_anchor_2x equal 2*count(anchor)\n"
        f"# Channeling threshold: 2 Å below the lowest anchor atom (avoids false positives)\n"
        f"variable    channeling_z equal bound(anchor,zmin)-2.0\n"
        f"\n"
        f"# mass 1 is set by read_data; set here again to use consistent M_C variable\n"
        f"mass        1 ${{M_C}}\n"
        f"mass        2 ${{M_H}}\n"
        f"mass        3 ${{M_O}}\n"
        f"mass        4 ${{{species['mass_var']}}}\n"
        f"\n"
        f"# Potential\n"
        f"{_potential_block(species)}"
        f"\n"
        f"variable c equal ${{n_complete}}\n"
        f"variable event_count equal ${{n_events}}\n"
        f"{rie_pre_loop or 'variable    cn equal 0\n'}"
        f"\n"
        f"variable\tnfixed   equal count(anchor)\n"
        f"variable\tnmobile  equal count(mobile)\n"
        f"variable \tninject  equal count(insert)\n"
        f"\n"
        f"variable \tvel equal sqrt(2*${{energ}}*6.02214129*1.0e+7/${{M_incident}}/6242)/1000\n"
        f"variable \tvelz equal cos(${{ion_angl}}*PI/180)*${{vel}}\n"
        f"variable \tvely equal sin(${{ion_angl}}*PI/180)*${{vel}}\n"
        f"\n"
        f"compute     ake all ke\n"
        f"compute     ape all pe\n"
        f"compute     ike insert ke\n"
        f"variable    ate equal c_ake+c_ape+ecouple\n"
        f"\n"
        f"variable\tcheckevery equal 1000\n"
        f"compute\t    clusts all cluster/atom 3.0\n"
        f"compute\t    clust_min all reduce min c_clusts\n"
        f"compute\t    clust_max all reduce max c_clusts\n"
        f"variable    one_clust equal \"c_clust_min == c_clust_max\"\n"
        f"fix\t\t    stopclust all halt ${{checkevery}} v_one_clust == 0 error continue message yes\n"
        f"\n"
        f"compute     cclusts carbon cluster/atom 1.8\n"
        f"compute     cc1 carbon chunk/atom c_cclusts compress yes\n"
        f"compute     nclusts carbon reduce max c_cc1\n"
        f"\n"
        f"thermo_style    custom step time v_ninject temp c_ike dt ecouple v_ate v_one_clust c_nclusts\n"
        f"compute     \tmtemp mobile temp\n"
        f"compute_modify  mtemp dynamic/dof yes\n"
        f"thermo_modify   temp mtemp\n"
        f"thermo_modify   lost warn flush yes\n"
        f"\n"
        f'print "${{c}}" file begin.txt\n'
        f"{molecule_decl}"
        f"{_carbon_etch_initial_therm_block(spec.initial_thermalization)}"
        f"\n"
        f"# ========================= Begin Per-Impact Outer Loop =========================\n"
        f"label\t\tloop\n"
        f"region bbox delete\n"
        f"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_redef(spec)
        + f"{_carbon_etch_slab_check()}"
        f"{rie_loop_top}"
        f"\n"
        f"group       insert clear\n"
        f"group \t    mobile subtract all anchor\n"
        f"{nonargon_regroup}"
        f"\n"
        f"{radical_loop}"
        f"{loop_counter_line}"
        f"{ion_dump_open}"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{c}}))\n"
        f"{_deposit_line(species, spec)}"
        f"\n"
        f"fix     2 mobile nve\n"
        f"fix     3 insert nve\n"
        f"\n"
        f"timestep 1e-10\n"
        f"run 1 post no\n"
        f"variable starting_nclusts equal $(c_nclusts)\n"
        f"\n"
        f"fix ats all dt/reset 1 NULL 1.00 0.01 units box\n"
        f"thermo      100\n"
        f"\n"
        f"variable    t0 equal $(time)\n"
        f"variable    time_elapsed equal time-${{t0}}\n"
        f"fix         thalt all halt 1 v_time_elapsed > ${{impact_time}} error continue message yes\n"
        f"\n"
        f"run         0 post no\n"
        f"variable    n_channelled equal 0\n"
        f"# =========================== begin inner loop ===========================\n"
        f"label\t\tcontinue_impact\n"
        f"\n"
        f"run         1000000000 pre no post no\n"
        f"run         0\n"
        f"{etch_event_block}"
        f"{channeling_block}"
        f"\n"
        f'if "$(time-v_t0) < ${{impact_time}}" then "jump SELF continue_impact"\n'
        f"unfix thalt\n"
        f"# ============================ end inner loop ============================\n"
        f"\n"
        f"unfix   depo\n"
        f"unfix   ats\n"
        f"\n"
        f"include thermalize.lmp\n"
        f"\n"
        f"unfix       2\n"
        f"unfix       3\n"
        f"\n"
        + (
        f"# Remove implanted {spec.species} ion\n"
        f"group       incident_ion type 4\n"
        f"delete_atoms group incident_ion\n"
        f"group       incident_ion delete\n"
        f"\n"
        if species["needs_zbl"] and spec.remove_ar else ""
        ) +
        f"# Atom counts (no replenishment in carbon-etch)\n"
        f"group       carbon type 1\n"
        f"group       hydrogen type 2\n"
        f"group       oxygen type 3\n"
        f"variable    ncarbon equal count(carbon)\n"
        f"variable    nhydrogen equal count(hydrogen)\n"
        f"variable    noxygen equal count(oxygen)\n"
        f"\n"
        f'print "Run ${{c}} complete"\n'
        f'print "C_COUNT: ${{ncarbon}}"\n'
        f"{ncarbon_print}"
        f"\n"
        f"{write_data}"
        f"if '$(v_c%v_ML) == 0' then \"write_dump all custom ML_impacts.dump id type q x y z vx vy vz modify sort id append yes\"\n"
        f"\n"
        f"{ion_dump_close}"
        f"# ========================= End Per-Impact Outer Loop =========================\n"
        f"next\t\ta\n"
        f"jump\t\tSELF loop\n"
        f"{_carbon_etch_end_label()}"
    )


def get_head_lmp_carbon_etch_multi_ion(spec: SimSpec) -> str:
    """Generate head.lmp for carbon-etch + multi-ion (ion_mix) mode."""
    mix          = spec.ion_mix
    has_zbl      = any(SPECIES[c.species]["needs_zbl"]   for c in mix)
    has_molecule = any(SPECIES[c.species]["is_molecule"] for c in mix)
    is_burst     = spec.flux_ratio > 0 and spec.radical_burst
    is_rie       = spec.flux_ratio > 0 and not spec.radical_burst
    has_radicals = spec.flux_ratio > 0

    zbl_sp           = next((SPECIES[c.species] for c in mix if SPECIES[c.species]["needs_zbl"]), None)
    potential        = _potential_block(zbl_sp if has_zbl else SPECIES["O"])
    mass4_var        = zbl_sp["mass_var"] if has_zbl else "M_Ar"
    molecule_decl    = "molecule O2 O2.molecule\n" if has_molecule else ""
    nonargon_regroup = "group nonargon type 1 2 3\n" if has_zbl else ""

    rie_pre_loop = ""
    rie_loop_top = ""
    if has_radicals:
        rie_pre_loop = (
            f"variable    cn_start equal ${{neut_complete}}\n"
            f"variable    cn equal 0\n"
        )
        rie_loop_top = (
            f'if "${{cn_start}} > 0" then "variable cn equal ${{cn_start}}" else "variable cn equal 0"\n'
            f"\n"
        )

    if has_radicals:
        ncarbon_print = f'print "${{c}} 0 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}_0.data nofix nocoeff\n"
    else:
        ncarbon_print = f'print "${{c}} ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}.data nofix nocoeff\n"

    if is_burst:
        radical_loop = _radical_burst_block(spec)
    elif is_rie:
        radical_loop = _radical_loop_block(spec)
    else:
        radical_loop = ""
    loop_counter_line = f"variable    c equal ${{c}}+1\n"

    ion_dump_open, etch_event_block, channeling_block, ion_dump_close = (
        _build_ion_dump_blocks(spec, is_carbon_etch=True)
    )

    return (
        f"# head.lmp - generated by DiamondEtchMD (carbon-etch multi-ion)\n"
        f"# config_file={spec.initial_config_file}\n"
        f"# ions=[{', '.join(f'{c.species}@{c.energy}eV*{c.fraction:.0%}' for c in mix)}]\n"
        f"# T={spec.surface_temperature}K  angle={spec.ion_angle}deg\n"
        f"package kokkos neigh/qeq full neigh half newton on\n"
        f"units\t\treal\n"
        f"include     config.lmp\n"
        f'if "${{pot}} == COMB3" then &\n'
        f'"atom_style charge" &\n'
        f'elif "${{pot}} == REAX" &\n'
        f'"atom_style charge" &\n'
        f"else &\n"
        f'"atom_style atomic"\n'
        f"\n"
        f"boundary\tp p m\n"
        f"\n"
        f"{_carbon_etch_load_block()}"
        f"\n"
        f"variable \tlp equal $(v_end_fluence*v_ML)-${{n_complete}}\n"
        f"variable    a loop ${{lp}}\n"
        f"\n"
        f"{_carbon_etch_anchor_block()}"
        + _expose_zone_def(spec)
        + f"\n"
        f"group \tanchor region anchor\n"
        + _freeze_mask_block(spec)
        + f"group\tinsert empty\n"
        f"group \tmobile subtract all anchor\n"
        f"group   carbon type 1\n"
        f"\n"
        f"variable    n_anchor_2x equal 2*count(anchor)\n"
        f"# Channeling threshold: 2 Å below the lowest anchor atom (avoids false positives)\n"
        f"variable    channeling_z equal bound(anchor,zmin)-2.0\n"
        f"\n"
        f"mass        1 ${{M_C}}\n"
        f"\n"
        f"{potential}"
        f"\n"
        f"variable c equal ${{n_complete}}\n"
        f"variable event_count equal ${{n_events}}\n"
        f"{rie_pre_loop or 'variable    cn equal 0\n'}"
        f"\n"
        f"variable\tnfixed   equal count(anchor)\n"
        f"variable\tnmobile  equal count(mobile)\n"
        f"variable \tninject  equal count(insert)\n"
        f"\n"
        f"compute     ake all ke\n"
        f"compute     ape all pe\n"
        f"compute     ike insert ke\n"
        f"variable    ate equal c_ake+c_ape+ecouple\n"
        f"\n"
        f"variable\tcheckevery equal 1000\n"
        f"compute\t    clusts all cluster/atom 3.0\n"
        f"compute\t    clust_min all reduce min c_clusts\n"
        f"compute\t    clust_max all reduce max c_clusts\n"
        f"variable    one_clust equal \"c_clust_min == c_clust_max\"\n"
        f"fix\t\t    stopclust all halt ${{checkevery}} v_one_clust == 0 error continue message yes\n"
        f"\n"
        f"compute     cclusts carbon cluster/atom 1.8\n"
        f"compute     cc1 carbon chunk/atom c_cclusts compress yes\n"
        f"compute     nclusts carbon reduce max c_cc1\n"
        f"\n"
        f"thermo_style    custom step time v_ninject temp c_ike dt ecouple v_ate v_one_clust c_nclusts\n"
        f"compute     \tmtemp mobile temp\n"
        f"compute_modify  mtemp dynamic/dof yes\n"
        f"thermo_modify   temp mtemp\n"
        f"thermo_modify   lost warn flush yes\n"
        f"\n"
        f'print "${{c}}" file begin.txt\n'
        f"{molecule_decl}"
        f"{_carbon_etch_initial_therm_block(spec.initial_thermalization)}"
        f"\n"
        f"# ========================= Begin Per-Impact Outer Loop =========================\n"
        f"label\t\tloop\n"
        f"region bbox delete\n"
        f"region bbox block EDGE EDGE EDGE EDGE EDGE EDGE\n"
        + _expose_zone_redef(spec)
        + f"{_carbon_etch_slab_check()}"
        f"{rie_loop_top}"
        f"\n"
        f"group       insert clear\n"
        f"group \t    mobile subtract all anchor\n"
        f"{nonargon_regroup}"
        f"\n"
        f"{radical_loop}"
        f"{loop_counter_line}"
        f"{ion_dump_open}"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{c}}))\n"
        f"{_multi_ion_select_block(spec)}"
        f"{_multi_ion_deposit_line(spec)}"
        f"\n"
        f"fix     2 mobile nve\n"
        f"fix     3 insert nve\n"
        f"\n"
        f"timestep 1e-10\n"
        f"run 1 post no\n"
        f"variable starting_nclusts equal $(c_nclusts)\n"
        f"\n"
        f"fix ats all dt/reset 1 NULL 1.00 0.01 units box\n"
        f"thermo      100\n"
        f"\n"
        f"variable    t0 equal $(time)\n"
        f"variable    time_elapsed equal time-${{t0}}\n"
        f"fix         thalt all halt 1 v_time_elapsed > ${{impact_time}} error continue message yes\n"
        f"\n"
        f"run         0 post no\n"
        f"variable    n_channelled equal 0\n"
        f"# =========================== begin inner loop ===========================\n"
        f"label\t\tcontinue_impact\n"
        f"\n"
        f"run         1000000000 pre no post no\n"
        f"run         0\n"
        f"{etch_event_block}"
        f"{channeling_block}"
        f"\n"
        f'if "$(time-v_t0) < ${{impact_time}}" then "jump SELF continue_impact"\n'
        f"unfix thalt\n"
        f"# ============================ end inner loop ============================\n"
        f"\n"
        f"unfix   depo\n"
        f"unfix   ats\n"
        f"\n"
        f"include thermalize.lmp\n"
        f"\n"
        f"unfix       2\n"
        f"unfix       3\n"
        f"\n"
        + (
        f"# Remove implanted ZBL ion\n"
        f"group       incident_ion type 4\n"
        f"delete_atoms group incident_ion\n"
        f"group       incident_ion delete\n"
        f"\n"
        if has_zbl and spec.remove_ar else ""
        ) +
        f"group       carbon type 1\n"
        f"group       hydrogen type 2\n"
        f"group       oxygen type 3\n"
        f"variable    ncarbon equal count(carbon)\n"
        f"variable    nhydrogen equal count(hydrogen)\n"
        f"variable    noxygen equal count(oxygen)\n"
        f"\n"
        f'print "Run ${{c}} complete"\n'
        f'print "C_COUNT: ${{ncarbon}}"\n'
        f"{ncarbon_print}"
        f"\n"
        f"{write_data}"
        f"if '$(v_c%v_ML) == 0' then \"write_dump all custom ML_impacts.dump id type q x y z vx vy vz modify sort id append yes\"\n"
        f"\n"
        f"{ion_dump_close}"
        f"# ========================= End Per-Impact Outer Loop =========================\n"
        f"next\t\ta\n"
        f"jump\t\tSELF loop\n"
        f"{_carbon_etch_end_label()}"
    )
