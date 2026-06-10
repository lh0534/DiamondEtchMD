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
        # Ar: hybrid ReaxFF + ZBL for short-range nuclear repulsion
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
        # O, O2, H: plain ReaxFF — type 4 slot maps to C so all pair coeffs are set
        return (
            f'if "${{pot}} == REAX" then &\n'
            f'"pair_style reaxff NULL mincap 200 safezone 1.5" &\n'
            f'"pair_coeff * * ffield.reax C H O C" &\n'
            f'"fix reax_qeq all qeq/reaxff 1 0.0 6.0 1e-6 reaxff"\n'
        )


def _deposit_line(species: dict) -> str:
    """Return the fix deposit command for the given species."""
    if species["is_molecule"]:
        # O2: inject as molecule; type 0 means types come from molecule file
        return (
            f"fix     depo insert deposit 1 0 10000000000 ${{deposeed}} global "
            f"${{i_above}} ${{i_above}} vx 0.0 0.0 vy ${{vely}} ${{vely}} "
            f"vz -${{velz}} -${{velz}} region bbox units box mol O2\n"
        )
    else:
        # single atom: inject by type index
        return (
            f"fix     depo insert deposit 1 ${{incident_type_index}} 10000000000 ${{deposeed}} global "
            f"${{i_above}} ${{i_above}} vx 0.0 0.0 vy ${{vely}} ${{vely}} "
            f"vz -${{velz}} -${{velz}} region bbox units box\n"
        )


def _radical_loop_block(spec: SimSpec) -> str:
    """Return the LAMMPS radical pre-exposure loop for RIE-etch mode.

    Deposits `flux_ratio` O• radicals (type 3) before each ion impact.
    Handles mid-loop restarts via cn_start (from neut_complete variable).
    After each radical: writes 5-col ncarbon.txt and impact_snaps/${c}_${cn}.data.
    """
    return (
        f"# ========= Begin RIE-etch O• radical deposition loop =========\n"
        f'if "${{cn_start}} > 0 && ${{cn_start}} < ${{flux_ratio}}" then &\n'
        f'"variable neutral_lp loop $(v_flux_ratio-v_cn_start)" &\n'
        f'elif "${{cn_start}} == ${{flux_ratio}}" &\n'
        f'"jump SELF skip_radicals" &\n'
        f"else &\n"
        f'"variable neutral_lp loop ${{flux_ratio}}"\n'
        f"\n"
        f"label       neutral_loop\n"
        f"variable    cn equal ${{cn}}+1\n"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{cn}}))\n"
        f"group       insert clear\n"
        f"group       mobile subtract all anchor\n"
        f"\n"
        f"# Deposit O• radical (always type 3)\n"
        f"variable    vel_chem equal sqrt(2*${{radical_energy}}*6.02214129*1.0e+7/${{M_O}}/6242)/1000\n"
        f"variable    velz_chem equal cos(${{angl}}*PI/180)*${{vel_chem}}\n"
        f"variable    vely_chem equal sin(${{angl}}*PI/180)*${{vel_chem}}\n"
        f"fix         depo insert deposit 1 3 1 ${{deposeed}} global "
        f"${{chemical_i_above}} ${{chemical_i_above}} "
        f"vx 0.0 0.0 vy ${{vely_chem}} ${{vely_chem}} vz -${{velz_chem}} -${{velz_chem}} "
        f"region bbox units box\n"
        f"fix         2 mobile nve\n"
        f"fix         3 insert nve\n"
        f"dump        current_dump_n all custom 100 etch_event_trajs/event_dump_n${{event_count}}.dump "
        f"id type x y z vx vy vz fx fy fz q\n"
        f"\n"
        f"timestep    1e-10\n"
        f"run         1 post no\n"
        f"run         0\n"
        f"fix         ats_n all dt/reset 1 NULL 1.00 0.01 units box\n"
        f"variable    starting_nclusts equal $(c_nclusts)\n"
        f"variable    t0 equal $(time)\n"
        f"variable    time_elapsed equal time-${{t0}}\n"
        f"fix         thalt all halt 1 v_time_elapsed > ${{inter_neutral_time}} error continue message yes\n"
        f"\n"
        f"# ===================== Radical inner loop =====================\n"
        f"label       continue_n_impact\n"
        f"run         500 pre no post no\n"
        f"run         0\n"
        f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
        f'"variable event_count equal ${{event_count}}+1" &\n'
        f'"variable starting_nclusts equal $(c_nclusts)"\n'
        f'if "${{one_clust}} == 0" then "include sweep.lmp"\n'
        f'if "$(time-v_t0) < ${{inter_neutral_time}}" then "jump SELF continue_n_impact"\n'
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
        f"undump      current_dump_n\n"
        f"unfix       depo\n"
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
        f"# Thermalize between radicals and ion\n"
        f"include     thermalize.lmp\n"
        f"\n"
    )


def get_head_lmp(spec: SimSpec) -> str:
    """Generate the contents of head.lmp for the given SimSpec.

    Supports ion-etch (flux_ratio == 0) and RIE-etch (flux_ratio > 0).
    """
    cfg = ORIENT[spec.orientation]
    lattice_cmd = cfg["lattice_cmd"]
    bottom_expr = cfg["bottom_expr"]
    species = SPECIES[spec.species]
    is_rie = spec.flux_ratio > 0

    # O2 molecule declaration (before the loop)
    molecule_decl = ""
    if species["is_molecule"]:
        molecule_decl = f"molecule O2 O2.molecule\n"

    # Ar: re-establish nonargon group at top of each loop iteration
    nonargon_regroup = ""
    if species["needs_zbl"]:
        nonargon_regroup = f"group nonargon type 1 2 3\n"

    # Post-impact removal for inert species (Ar); gated by spec.remove_ar
    removal_block = ""
    if species["remove_after_impact"] and spec.remove_ar:
        removal_block = (
            f"\n"
            f"# Remove {spec.species} (inert, does not participate in chemistry)\n"
            f"group       IonRemove type {species['type_index']}\n"
            f"delete_atoms group IonRemove\n"
            f"group       IonRemove delete\n"
        )

    # RIE-etch: cn (radical counter) restart variables before the loop
    rie_pre_loop = ""
    if is_rie:
        rie_pre_loop = (
            f"variable    cn_start equal ${{neut_complete}}\n"
            f"variable    cn equal 0\n"
        )

    # RIE-etch: cn_start reset at top of loop iteration
    rie_loop_top = ""
    if is_rie:
        rie_loop_top = (
            f'if "${{cn_start}} > 0" then "variable cn equal ${{cn_start}}" else "variable cn equal 0"\n'
            f"\n"
        )

    # ncarbon.txt output format: 5-col for RIE-etch, 4-col for ion-etch
    if is_rie:
        ncarbon_print = f'print "${{c}} 0 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}_0.data nofix nocoeff\n"
    else:
        ncarbon_print = f'print "${{c}} ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}.data nofix nocoeff\n"

    # RIE-etch radical loop block (inserted before the ion deposit section)
    radical_loop = _radical_loop_block(spec) if is_rie else ""

    # In RIE-etch mode, ion counter is NOT incremented at top of loop (radicals run first)
    # In ion-etch mode, increment happens in place
    loop_counter_line = f"variable    c equal ${{c}}+1\n"

    return (
        f"# head.lmp — generated by DiamondEtchMD\n"
        f"# orientation={spec.orientation}  species={spec.species}"
        f"  {spec.energy}eV  {spec.temperature}K  angle={spec.angle}deg\n"
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
        f"variable        sublat equal ${{bottom}}+1/2\n"
        f"region          anchor block INF INF INF INF ${{bottom}} ${{sublat}} units lattice\n"
        f"\n"
        f"# Groups\n"
        f"group \tanchor region anchor\n"
        f"group\tinsert empty\n"
        f"group \tmobile subtract all anchor\n"
        f"group   carbon type 1\n"
        f"\n"
        f"# Particle masses (4 types: C, H, O, Ar)\n"
        f"mass        1 ${{M_C}}\n"
        f"mass        2 ${{M_H}}\n"
        f"mass        3 ${{M_O}}\n"
        f"mass        4 ${{M_Ar}}\n"
        f"\n"
        f"# Potential\n"
        f"{_potential_block(species)}"
        f"\n"
        f"# Pre-run counters\n"
        f"variable c equal ${{n_complete}}\n"
        f"variable event_count equal ${{n_events}}\n"
        f"{rie_pre_loop}"
        f"\n"
        f"# Atom counts\n"
        f"variable\tnfixed   equal count(anchor)\n"
        f"variable\tnmobile  equal count(mobile)\n"
        f"variable \tninject  equal count(insert)\n"
        f"\n"
        f"# Incident particle velocity components\n"
        f"variable \tvel equal sqrt(2*${{energ}}*6.02214129*1.0e+7/${{M_incident}}/6242)/1000\n"
        f"variable \tvelz equal cos(${{angl}}*PI/180)*${{vel}}\n"
        f"variable \tvely equal sin(${{angl}}*PI/180)*${{vel}}\n"
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
        f"{rie_loop_top}"
        f"dump current_dump all custom 100 etch_event_trajs/event_dump_${{event_count}}.dump id type x y z vx vy vz fx fy fz q\n"
        f"\n"
        f"group       insert clear\n"
        f"group \t    mobile subtract all anchor\n"
        f"{nonargon_regroup}"
        f"\n"
        f"{radical_loop}"
        f"{loop_counter_line}"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{c}}))\n"
        f"{_deposit_line(species)}"
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
        f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
        f'"variable event_count equal ${{event_count}}+1" &\n'
        f'"variable starting_nclusts equal $(c_nclusts)"\n'
        f'if "${{one_clust}} == 0" then &\n'
        f'"include sweep.lmp" &\n'
        f'"region channelled block INF INF INF INF INF ${{bottom}} units lattice" &\n'
        f'"group channelled_group region channelled" &\n'
        f'"variable n_channelled equal count(channelled_group)" &\n'
        f'"region channelled delete" &\n'
        f"\n"
        f'if "${{n_channelled}} > 0" then &\n'
        f'"delete_atoms group channelled_group" &\n'
        f'"run 0" &\n'
        f'"group channelled_group delete" &\n'
        f'"variable n_channelled equal 0" &\n'
        f'"include notify_channeled.lmp"\n'
        f"\n"
        f'if "$(time-v_t0) < ${{impact_time}}" then "jump SELF continue_impact"\n'
        f"unfix thalt\n"
        f"# ============================ end inner loop ============================\n"
        f"\n"
        f"unfix   depo\n"
        f"unfix   ats\n"
        f"{removal_block}"
        f"\n"
        f"# Thermalize\n"
        f"include thermalize.lmp\n"
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
        f"undump current_dump\n"
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
        "variable    velz equal cos(${angl}*PI/180)*${vel}\n",
        "variable    vely equal sin(${angl}*PI/180)*${vel}\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "\n",
    ])

    return "".join(lines)


def _multi_ion_deposit_line() -> str:
    """Return a conditional fix deposit that handles both atom and molecule ions."""
    return (
        f'if "${{cur_ion_is_mol}} == 1" then &\n'
        f'"fix     depo insert deposit 1 0 10000000000 ${{deposeed}} global '
        f'${{i_above}} ${{i_above}} vx 0.0 0.0 vy ${{vely}} ${{vely}} '
        f'vz -${{velz}} -${{velz}} region bbox units box mol O2" &\n'
        f"else &\n"
        f'"fix     depo insert deposit 1 ${{incident_type_index}} 10000000000 ${{deposeed}} global '
        f'${{i_above}} ${{i_above}} vx 0.0 0.0 vy ${{vely}} ${{vely}} '
        f'vz -${{velz}} -${{velz}} region bbox units box"\n'
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
    has_zbl      = any(SPECIES[c.species]["needs_zbl"]           for c in mix)
    has_molecule = any(SPECIES[c.species]["is_molecule"]          for c in mix)
    has_removal  = any(SPECIES[c.species]["remove_after_impact"]  for c in mix)
    is_rie       = spec.flux_ratio > 0

    # Potential block — Ar mix needs hybrid ZBL; O/O2 mix uses plain ReaxFF
    potential = _potential_block(SPECIES["Ar"] if has_zbl else SPECIES["O"])

    molecule_decl   = "molecule O2 O2.molecule\n" if has_molecule else ""
    nonargon_regroup = "group nonargon type 1 2 3\n" if has_zbl else ""

    if has_removal and spec.remove_ar:
        removal_block = (
            f"\n"
            f"# Remove any Ar atoms deposited this impact (guarded: may be 0 if non-Ar was selected)\n"
            f"group       IonRemove type 4\n"
            f'if "$(count(IonRemove)) > 0" then "delete_atoms group IonRemove"\n'
            f"group       IonRemove delete\n"
        )
    else:
        removal_block = ""

    rie_pre_loop = ""
    rie_loop_top = ""
    if is_rie:
        rie_pre_loop = (
            f"variable    cn_start equal ${{neut_complete}}\n"
            f"variable    cn equal 0\n"
        )
        rie_loop_top = (
            f'if "${{cn_start}} > 0" then "variable cn equal ${{cn_start}}" else "variable cn equal 0"\n'
            f"\n"
        )

    if is_rie:
        ncarbon_print = f'print "${{c}} 0 ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}_0.data nofix nocoeff\n"
    else:
        ncarbon_print = f'print "${{c}} ${{ncarbon}} ${{nhydrogen}} ${{noxygen}}" append ncarbon.txt\n'
        write_data    = f"write_data impact_snaps/${{c}}.data nofix nocoeff\n"

    radical_loop    = _radical_loop_block(spec) if is_rie else ""
    loop_counter_line = f"variable    c equal ${{c}}+1\n"

    ion_label = "_".join(c.species for c in mix)

    return (
        f"# head.lmp — generated by DiamondEtchMD (multi-ion)\n"
        f"# orientation={spec.orientation}  "
        f"ions=[{', '.join(f'{c.species}@{c.energy}eV×{c.fraction:.0%}' for c in mix)}]\n"
        f"# T={spec.temperature}K  angle={spec.angle}deg\n"
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
        f"variable        sublat equal ${{bottom}}+1/2\n"
        f"region          anchor block INF INF INF INF ${{bottom}} ${{sublat}} units lattice\n"
        f"\n"
        f"group \tanchor region anchor\n"
        f"group\tinsert empty\n"
        f"group \tmobile subtract all anchor\n"
        f"group   carbon type 1\n"
        f"\n"
        f"mass        1 ${{M_C}}\n"
        f"mass        2 ${{M_H}}\n"
        f"mass        3 ${{M_O}}\n"
        f"mass        4 ${{M_Ar}}\n"
        f"\n"
        f"{potential}"
        f"\n"
        f"variable c equal ${{n_complete}}\n"
        f"variable event_count equal ${{n_events}}\n"
        f"{rie_pre_loop}"
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
        f"{rie_loop_top}"
        f"dump current_dump all custom 100 etch_event_trajs/event_dump_${{event_count}}.dump "
        f"id type x y z vx vy vz fx fy fz q\n"
        f"\n"
        f"group       insert clear\n"
        f"group \t    mobile subtract all anchor\n"
        f"{nonargon_regroup}"
        f"\n"
        f"{radical_loop}"
        f"{loop_counter_line}"
        f"variable    deposeed equal floor(random(1,72099+${{seed_adjust}},${{c}}))\n"
        f"{_multi_ion_select_block(spec)}"
        f"{_multi_ion_deposit_line()}"
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
        f'if "$(c_nclusts) > ${{starting_nclusts}}" then &\n'
        f'"variable event_count equal ${{event_count}}+1" &\n'
        f'"variable starting_nclusts equal $(c_nclusts)"\n'
        f'if "${{one_clust}} == 0" then &\n'
        f'"include sweep.lmp" &\n'
        f'"region channelled block INF INF INF INF INF ${{bottom}} units lattice" &\n'
        f'"group channelled_group region channelled" &\n'
        f'"variable n_channelled equal count(channelled_group)" &\n'
        f'"region channelled delete" &\n'
        f"\n"
        f'if "${{n_channelled}} > 0" then &\n'
        f'"delete_atoms group channelled_group" &\n'
        f'"run 0" &\n'
        f'"group channelled_group delete" &\n'
        f'"variable n_channelled equal 0" &\n'
        f'"include notify_channeled.lmp"\n'
        f"\n"
        f'if "$(time-v_t0) < ${{impact_time}}" then "jump SELF continue_impact"\n'
        f"unfix thalt\n"
        f"# ============================ end inner loop ============================\n"
        f"\n"
        f"unfix   depo\n"
        f"unfix   ats\n"
        f"{removal_block}"
        f"\n"
        f"include thermalize.lmp\n"
        f"\n"
        f"unfix       2\n"
        f"unfix       3\n"
        f"\n"
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
        f"undump current_dump\n"
        f"# ========================= End Per-Impact Outer Loop =========================\n"
        f"next\t\ta\n"
        f"jump\t\tSELF loop\n"
    )
