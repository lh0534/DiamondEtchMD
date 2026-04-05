"""
lammps/config.py — generator for config.lmp, the per-condition parameter file.

config.lmp is included at the top of head.lmp and defines all LAMMPS variables
that vary between simulation conditions: box size, energy, temperature, species
type, surface flags (reconstruct, O_terminate, O_ether_terminate), and timing.
"""

from ..orientations import ORIENT
from ..species import SPECIES
from ..spec import SimSpec

# Terminations that set O_terminate=true
_O_TERMINATE = {"O", "O_1x1", "O_2x1_single", "O_2x1_pandey"}


def get_config_lmp(spec: SimSpec) -> str:
    """Generate the contents of config.lmp for the given SimSpec."""
    species = SPECIES[spec.species]

    # reconstruct flag: only used by the 100 template (bare_2x1 triggers it).
    # 111 templates apply their own reconstruction unconditionally; 113 has none.
    recon_flag    = "true" if (spec.orientation == "100" and spec.reconstruction == "bare_2x1") else "false"
    o_flag        = "true" if spec.termination in _O_TERMINATE else "false"
    o_eth_flag    = "true" if spec.termination == "O_ether"    else "false"

    # O2 energy is split across 2 atoms; user specifies total dimer energy
    energy_per_atom = spec.energy / species["energy_divisor"]

    return (
        f"# DiamondEtchMD generated config\n"
        f"# orientation={spec.orientation}  reconstruction={spec.reconstruction}"
        f"  termination={spec.termination}\n"
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
