"""
lammps/config.py — generator for config.lmp, the per-condition parameter file.

config.lmp is included at the top of head.lmp and defines all LAMMPS variables
that vary between simulation conditions: box size, energy, temperature, species
type, surface flags (reconstruct, O_terminate, O_ether_terminate), and timing.
"""

from ..orientations import ORIENT
from ..species import SPECIES
from ..spec import SimSpec, CyclePhase


def get_config_lmp(spec: SimSpec) -> str:
    """Generate the contents of config.lmp for the given SimSpec."""
    species = SPECIES[spec.species]
    surf = ORIENT[spec.orientation]["surfaces"][spec.surface]

    recon_flag = "true" if surf["reconstruct"]   else "false"
    o_flag     = "true" if surf["O_terminate"]   else "false"
    o_eth_flag = "true" if surf["O_ether"]       else "false"

    # O2 energy is split across 2 atoms; user specifies total dimer energy
    energy_per_atom = spec.energy / species["energy_divisor"]

    return (
        f"# DiamondEtchMD generated config (single-species)\n"
        f"# orientation={spec.orientation}  surface={spec.surface}\n"
        f"# species={spec.species}  energy={spec.energy}eV  angle={spec.angle}deg"
        f"  T={spec.temperature}K\n"
        f"\n"
        f"variable    ML equal {spec.ml}            # atoms per monolayer\n"
        f"variable    end_fluence equal {spec.fluence}   # in ML\n"
        f"\n"
        f"variable    energ equal {energy_per_atom}     # incident energy per atom (eV)\n"
        f"variable    angl equal {spec.angle}       # incident particle angle (deg from normal)\n"
        f"variable    T equal {spec.temperature}    # substrate temperature (K)\n"
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
        f"variable    i_above equal {species['i_above']}    # Å above surface to inject particle\n"
        f"variable    impact_time equal {spec.impact_time}\n"
        f"variable    thermalization_time equal {spec.thermalization_time}\n"
        f"\n"
        f"variable    M_C equal 12.011\n"
        f"variable    M_H equal 1.00784\n"
        f"variable    M_O equal 16.0\n"
        f"variable    M_Ar equal 39.948\n"
        f"variable    M_incident equal ${{{species['mass_var']}}}\n"
        f"variable    incident_type_index equal {species['type_index']}  # {spec.species}\n"
        f"\n"
        f"variable    seed_adjust equal 0\n"
        f"variable    simdepo equal 1\n"
    )


def get_config_lmp_cycling(spec: SimSpec) -> str:
    """Generate config.lmp for a cycling (multi-phase) SimSpec."""
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
            f"variable    phase_{i}_flux_ratio equal {p.flux_ratio}\n"
            f"variable    phase_{i}_radical_energy equal {p.radical_energy}\n"
        )

    phase_summary = ", ".join(
        f"{p.species}@{p.energy}eV×{p.fluence_ml}ML"
        + (f"+O•R{p.flux_ratio}" if p.flux_ratio > 0 else "")
        for p in spec.phases
    )

    return (
        f"# DiamondEtchMD generated config (cycling mode)\n"
        f"# orientation={spec.orientation}  surface={spec.surface}  T={spec.temperature}K\n"
        f"# phases: {phase_summary}\n"
        f"# {spec.cycles} cycle(s) × {total_phase_ml} ML/cycle = {total_fluence_ml} ML total\n"
        f"\n"
        f"variable    ML equal {spec.ml}            # atoms per monolayer\n"
        f"variable    cycles equal {spec.cycles}\n"
        f"variable    end_fluence equal {total_fluence_ml}  # total ML\n"
        f"\n"
        f"variable    angl equal {spec.angle}       # incident angle (deg from normal)\n"
        f"variable    T equal {spec.temperature}    # substrate temperature (K)\n"
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
        f"variable    inter_neutral_time equal {spec.inter_neutral_time}\n"
        f"\n"
        f"variable    ion_i_above equal 6.0     # Å above surface to inject ion\n"
        f"variable    chemical_i_above equal 6.0 # Å above surface to inject O•\n"
        f"\n"
        f"variable    M_C equal 12.011\n"
        f"variable    M_H equal 1.00784\n"
        f"variable    M_O equal 16.0\n"
        f"variable    M_Ar equal 39.948\n"
        f"\n"
        f"variable    seed_adjust equal 0\n"
        f"\n"
        f"# Per-phase parameters\n"
        f"{phase_vars}"
    )
