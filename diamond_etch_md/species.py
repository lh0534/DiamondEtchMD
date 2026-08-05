"""
species.py — registry of supported incident species.

The simulation framework uses a C-H-O ReaxFF potential with up to 4 atom types:
  type 1 = C, type 2 = H, type 3 = O, type 4 = inert ion (any ZBL species)

Type-4 ions (all elements except C/H/O) use a hybrid ReaxFF+ZBL pair style:
  pair_coeff * * reaxff ffield.reax C H O NULL   ← NULL = no ReaxFF for type 4
  pair_coeff i 4 zbl Zi Z4                       ← ZBL handles short-range repulsion

ZBL ions do not bond to the substrate and are removed after each impact.
No changes to ffield.reax are needed — the NULL slot covers any element.

Each species entry contains:
  type_index          : LAMMPS atom type index for the incident particle
  mass_var            : LAMMPS variable name holding the atomic mass (defined in config.lmp)
  mass                : atomic mass in amu
  i_above             : height in Å above the surface at which the particle is injected
  is_molecule         : whether the species is injected as a LAMMPS molecule (vs single atom)
  molecule_file       : filename of the molecule template (None for single atoms)
  energy_divisor      : spec.energy is divided by this to get per-atom energy in config.lmp
  needs_zbl           : whether a hybrid ReaxFF+ZBL pair style is needed
  atomic_number       : nuclear charge Z for ZBL pair coefficients (required when needs_zbl)
  remove_after_impact : whether to delete these atoms after each impact
"""

# ── ZBL ion table: all elements except C (6), H (1), O (8) ─────────────────
# Values: symbol → (atomic_number Z, standard_atomic_weight amu)
# Radioactive/synthetic elements use the mass of the most stable isotope.
_ZBL_ELEMENTS = {
    "He": (2,    4.0026),
    "Li": (3,    6.941),
    "Be": (4,    9.0122),
    "B":  (5,   10.811),
    "N":  (7,   14.007),
    "F":  (9,   18.998),
    "Ne": (10,  20.180),
    "Na": (11,  22.990),
    "Mg": (12,  24.305),
    "Al": (13,  26.982),
    "Si": (14,  28.085),
    "P":  (15,  30.974),
    "S":  (16,  32.06),
    "Cl": (17,  35.45),
    "Ar": (18,  39.948),
    "K":  (19,  39.098),
    "Ca": (20,  40.078),
    "Sc": (21,  44.956),
    "Ti": (22,  47.867),
    "V":  (23,  50.942),
    "Cr": (24,  51.996),
    "Mn": (25,  54.938),
    "Fe": (26,  55.845),
    "Co": (27,  58.933),
    "Ni": (28,  58.693),
    "Cu": (29,  63.546),
    "Zn": (30,  65.38),
    "Ga": (31,  69.723),
    "Ge": (32,  72.630),
    "As": (33,  74.922),
    "Se": (34,  78.971),
    "Br": (35,  79.904),
    "Kr": (36,  83.798),
    "Rb": (37,  85.468),
    "Sr": (38,  87.62),
    "Y":  (39,  88.906),
    "Zr": (40,  91.224),
    "Nb": (41,  92.906),
    "Mo": (42,  95.95),
    "Tc": (43,  98.0),
    "Ru": (44, 101.07),
    "Rh": (45, 102.906),
    "Pd": (46, 106.42),
    "Ag": (47, 107.868),
    "Cd": (48, 112.414),
    "In": (49, 114.818),
    "Sn": (50, 118.710),
    "Sb": (51, 121.760),
    "Te": (52, 127.60),
    "I":  (53, 126.904),
    "Xe": (54, 131.293),
    "Cs": (55, 132.905),
    "Ba": (56, 137.327),
    "La": (57, 138.905),
    "Ce": (58, 140.116),
    "Pr": (59, 140.908),
    "Nd": (60, 144.242),
    "Pm": (61, 145.0),
    "Sm": (62, 150.36),
    "Eu": (63, 151.964),
    "Gd": (64, 157.25),
    "Tb": (65, 158.925),
    "Dy": (66, 162.500),
    "Ho": (67, 164.930),
    "Er": (68, 167.259),
    "Tm": (69, 168.934),
    "Yb": (70, 173.045),
    "Lu": (71, 174.967),
    "Hf": (72, 178.49),
    "Ta": (73, 180.948),
    "W":  (74, 183.84),
    "Re": (75, 186.207),
    "Os": (76, 190.23),
    "Ir": (77, 192.217),
    "Pt": (78, 195.084),
    "Au": (79, 196.967),
    "Hg": (80, 200.592),
    "Tl": (81, 204.38),
    "Pb": (82, 207.2),
    "Bi": (83, 208.980),
    "Po": (84, 209.0),
    "At": (85, 210.0),
    "Rn": (86, 222.0),
    "Fr": (87, 223.0),
    "Ra": (88, 226.0),
    "Ac": (89, 227.0),
    "Th": (90, 232.038),
    "Pa": (91, 231.036),
    "U":  (92, 238.029),
    "Np": (93, 237.0),
    "Pu": (94, 244.0),
    "Am": (95, 243.0),
    "Cm": (96, 247.0),
    "Bk": (97, 247.0),
    "Cf": (98, 251.0),
    "Es": (99, 252.0),
    "Fm": (100, 257.0),
    "Md": (101, 258.0),
    "No": (102, 259.0),
    "Lr": (103, 262.0),
    "Rf": (104, 267.0),
    "Db": (105, 270.0),
    "Sg": (106, 271.0),
    "Bh": (107, 270.0),
    "Hs": (108, 277.0),
    "Mt": (109, 276.0),
    "Ds": (110, 281.0),
    "Rg": (111, 282.0),
    "Cn": (112, 285.0),
    "Nh": (113, 286.0),
    "Fl": (114, 289.0),
    "Mc": (115, 290.0),
    "Lv": (116, 293.0),
    "Ts": (117, 294.0),
    "Og": (118, 294.0),
}

# ── Reactive (ReaxFF) species ────────────────────────────────────────────────
SPECIES = {
    "O": {
        "type_index": 3,
        "mass_var": "M_O",
        "mass": 16.0,
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
        "mass": 1.00784,
        "i_above": 6.0,
        "is_molecule": False,
        "molecule_file": None,
        "energy_divisor": 1,
        "needs_zbl": False,
        "remove_after_impact": False,
    },
    "O2": {
        "type_index": 3,
        "mass_var": "M_O",
        "mass": 16.0,
        "i_above": 6.0,
        "is_molecule": True,
        "molecule_file": "O2.molecule",
        "energy_divisor": 2,
        "needs_zbl": False,
        "remove_after_impact": False,
    },
}

# ── Auto-generate ZBL ion entries from the elements table ───────────────────
for _sym, (_Z, _mass) in _ZBL_ELEMENTS.items():
    SPECIES[_sym] = {
        "type_index": 4,
        "mass_var": f"M_{_sym}",
        "mass": _mass,
        "i_above": 6.0,
        "is_molecule": False,
        "molecule_file": None,
        "energy_divisor": 1,
        "needs_zbl": True,
        "atomic_number": _Z,
        "remove_after_impact": True,
    }
