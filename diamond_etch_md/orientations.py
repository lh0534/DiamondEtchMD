"""
orientations.py — registry of supported diamond surface orientations.

Each entry in ORIENT contains:

  lattice_cmd  : LAMMPS lattice command matching the make_surf template
  bottom_expr  : LAMMPS expression to convert zlo (Å) to lattice z-units
  default_box  : (x, y, lat_top) in lattice units — sensible starting points
  ml_factor    : ML = ml_factor * x * y
  surfaces     : maps surface key → {template, reconstruct, O_terminate, O_ether}

The `surface` key is the single user-facing descriptor for the surface state.
It encodes both the geometric reconstruction and the chemical termination.

ML factors (analytically derived, empirically verified):
  100: ml_factor = 1   (verified: 8×8 → 64)
  110: ml_factor = 4   (verified: 4×6 → 96)
  111: ml_factor = 2   (verified: 5×9 → 90)
  113: ml_factor = 4   (verified: 9×3 → 108)

Surface keys:
  100: 1x1, 2x1, 2x1_O, O_ether
  110: (empty string), O
  111: 1x1, 2x1_single, 2x1_pandey, 1x1_O, 2x1_single_O, 2x1_pandey_O
  113: (empty string), O
"""

ORIENT = {
    "100": {
        "lattice_cmd": (
            "lattice    diamond ${lat_a}"
            " orient z 0 0 1 orient x 1 1 0 orient y -1 1 0"
            " spacing $(1/sqrt(2)) $(1/sqrt(2)) 1 origin 0 0 0"
        ),
        "bottom_expr": "$(zlo/(v_lat_a))",
        "default_box": (8, 8, 3),
        "ml_factor":   1,
        "surfaces": {
            "1x1":     {"template": "package:lammps/templates/make_surf_100.lmp",
                        "reconstruct": False, "O_terminate": False, "O_ether": False},
            "2x1":     {"template": "package:lammps/templates/make_surf_100.lmp",
                        "reconstruct": True,  "O_terminate": False, "O_ether": False},
            "2x1_O":   {"template": "package:lammps/templates/make_surf_100.lmp",
                        "reconstruct": True,  "O_terminate": True,  "O_ether": False},
            "O_ether":  {"template": "package:lammps/templates/make_surf_100.lmp",
                        "reconstruct": False, "O_terminate": False, "O_ether": True},
        },
    },
    "110": {
        "lattice_cmd": (
            "lattice    diamond ${lat_a}"
            " orient z 1 1 0 orient x -1 1 0 orient y 0 0 1"
            " spacing $(2/sqrt(2)) 1 $(3/sqrt(2)) origin 0 0 0"
        ),
        "bottom_expr": "$(zlo/(v_lat_a*3/sqrt(2)))",
        "default_box": (4, 6, 5),
        "ml_factor":   4,
        "surfaces": {
            "":  {"template": "package:lammps/templates/make_surf_110.lmp",
                  "reconstruct": False, "O_terminate": False, "O_ether": False},
            "O": {"template": "package:lammps/templates/make_surf_110.lmp",
                  "reconstruct": False, "O_terminate": True,  "O_ether": False},
        },
    },
    "111": {
        "lattice_cmd": (
            "lattice    diamond ${lat_a}"
            " orient z 1 1 1 orient x 2 -1 -1 orient y 0 1 -1"
            " spacing $(3/sqrt(6)) $(1/sqrt(2)) $(sqrt(3))"
            " origin 0 0 $(1*sqrt(3)/4)"
        ),
        "bottom_expr": "$(zlo/(v_lat_a*sqrt(3)))",
        "default_box": (5, 9, 3),
        "ml_factor":   2,
        "surfaces": {
            "1x1":          {"template": "package:lammps/templates/make_surf_111_1x1.lmp",
                             "reconstruct": False, "O_terminate": False, "O_ether": False},
            "2x1_single":   {"template": "package:lammps/templates/make_surf_111_2x1_single.lmp",
                             "reconstruct": False, "O_terminate": False, "O_ether": False},
            "2x1_pandey":   {"template": "package:lammps/templates/make_surf_111_2x1_pandey.lmp",
                             "reconstruct": False, "O_terminate": False, "O_ether": False},
            "1x1_O":        {"template": "package:lammps/templates/make_surf_111_1x1.lmp",
                             "reconstruct": False, "O_terminate": True,  "O_ether": False},
            "2x1_single_O": {"template": "package:lammps/templates/make_surf_111_2x1_single.lmp",
                             "reconstruct": False, "O_terminate": True,  "O_ether": False},
            "2x1_pandey_O": {"template": "package:lammps/templates/make_surf_111_2x1_pandey.lmp",
                             "reconstruct": False, "O_terminate": True,  "O_ether": False},
        },
    },
    "113": {
        "lattice_cmd": (
            "lattice    diamond ${lat_a}"
            " orient z 1 1 3 orient x -1 1 0 orient y -3 -3 2"
            " spacing $(1/sqrt(2)) $(11/sqrt(22)) $(5/sqrt(11)) origin 0 0 0"
        ),
        "bottom_expr": "$(zlo/(v_lat_a*5/sqrt(11)))",
        "default_box": (9, 3, 3),
        "ml_factor":   4,
        "surfaces": {
            "":  {"template": "package:lammps/templates/make_surf_113.lmp",
                  "reconstruct": False, "O_terminate": False, "O_ether": False},
            "O": {"template": "package:lammps/templates/make_surf_113.lmp",
                  "reconstruct": False, "O_terminate": True,  "O_ether": False},
        },
    },
}
