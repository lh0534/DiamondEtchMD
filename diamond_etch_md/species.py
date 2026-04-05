"""
species.py — registry of supported incident species.

The simulation framework uses a C-H-O ReaxFF potential with up to 4 atom types:
  type 1 = C, type 2 = H, type 3 = O, type 4 = Ar

Each species entry contains:
  type_index          : LAMMPS atom type index for the incident particle
  mass_var            : LAMMPS variable name holding the atomic mass (defined in config.lmp)
  i_above             : height in Å above the surface at which the particle is injected
  is_molecule         : whether the species is injected as a LAMMPS molecule (vs single atom)
  molecule_file       : filename of the molecule template (None for single atoms)
  energy_divisor      : spec.energy is divided by this to get per-atom energy in config.lmp
  needs_zbl           : whether a hybrid ReaxFF+ZBL pair style is needed (inert gas ions)
  remove_after_impact : whether to delete these atoms after each impact (inert gas ions)
"""

SPECIES = {
    "O": {
        "type_index": 3,
        "mass_var": "M_O",
        "i_above": 6.0,
        "is_molecule": False,
        "molecule_file": None,
        "energy_divisor": 1,
        "needs_zbl": False,
        "remove_after_impact": False,
    },
    "H": {
        "type_index": 2,
        "mass_var": "M_H",
        "i_above": 6.0,
        "is_molecule": False,
        "molecule_file": None,
        "energy_divisor": 1,
        "needs_zbl": False,
        "remove_after_impact": False,
    },
    "Ar": {
        "type_index": 4,
        "mass_var": "M_Ar",
        "i_above": 6.0,
        "is_molecule": False,
        "molecule_file": None,
        "energy_divisor": 1,
        "needs_zbl": True,
        "remove_after_impact": True,
    },
    "O2": {
        "type_index": 3,
        "mass_var": "M_O",
        "i_above": 6.0,
        "is_molecule": True,
        "molecule_file": "O2.molecule",
        "energy_divisor": 2,
        "needs_zbl": False,
        "remove_after_impact": False,
    },
}
