"""
species.py — registry of supported incident species for the radicals framework.

The radicals framework uses a C-H-O ReaxFF potential with 3 atom types:
  type 1 = C, type 2 = H, type 3 = O

Each species entry contains:
  type_index  : LAMMPS atom type index for the incident particle
  mass_var    : LAMMPS variable name holding the atomic mass (defined in config.lmp)
  i_above     : height in Å above the surface at which the particle is injected
"""

# Species supported by the radicals framework (C-H-O ReaxFF, 3 atom types)
SPECIES = {
    "O": {"type_index": 3, "mass_var": "M_O", "i_above": 6.0},
    "H": {"type_index": 2, "mass_var": "M_H", "i_above": 6.0},
}
