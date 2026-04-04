"""
orientations.py — registry of supported diamond surface orientations.

Each entry in ORIENT contains:

  lattice_cmd  : LAMMPS lattice command matching the make_surf.lmp for this orientation
  bottom_expr  : LAMMPS expression to convert zlo (Å) to lattice z-units for the anchor region
  default_box  : (x, y, lat_top) in lattice units — sensible starting points
  ml_factor    : ML = ml_factor * x * y  (derived from volume ratio and crystallographic plane count)

  Derivation: atoms/LAMMPS-unit-cell = 8 * spacing_x * spacing_y * spacing_z
              diamond planes per z-unit = FCC planes * 2 (diamond splits each FCC plane)
              ml_factor = atoms_per_cell / planes_per_z_unit

  001: 4 atoms/cell, 4 planes/z-unit → ml_factor = 1   (verified: 9×9 box → ML=81)
  111: 12 atoms/cell, 6 planes/z-unit → ml_factor = 2  (verified: 5×9 box → ML=90)
  113: 20 atoms/cell, 5 planes/z-unit → ml_factor = 4  (verified: 9×3 box → ML=108,
       confirmed by O-terminated surface having exactly 108 more atoms than bare)

  make_surf    : maps reconstruction key → source make_surf.lmp path as a STRING relative
                 to the dfiles/ root.  "*" means one file handles all cases via config.lmp
                 variables.  Paths are kept as strings (not Path objects) to avoid coupling
                 to absolute filesystem paths at import time.
"""

ORIENT = {
    "100": {
        "lattice_cmd": (
            "lattice    diamond ${lat_a}"
            " orient z 0 0 1 orient x 1 1 0 orient y -1 1 0"
            " spacing $(1/sqrt(2)) $(1/sqrt(2)) 1 origin 0 0 0"
        ),
        "bottom_expr": "$(zlo/(v_lat_a))",
        "default_box": (9, 9, 3),
        "ml_factor":   1,
        # single file handles all cases; reconstruction (bare/2x1) and termination
        # (bare/H/O/O_ether) are controlled by config.lmp flag variables.
        # 2x1: displaces alternating surface dimer rows along [110]/[-1-10].
        # O_ether: bridges O between adjacent surface C atoms (ether geometry).
        "make_surf": {
            "bare": "package:lammps/templates/make_surf_100.lmp",
            "2x1":  "package:lammps/templates/make_surf_100.lmp",
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
        "make_surf": {
            "bare":       "111/bare_surf/1x1_non-reconstructed/make_surf.lmp",
            "1x1":        "111/bare_surf/1x1_non-reconstructed/make_surf.lmp",
            "2x1_single": "111/bare_surf/2x1_Single_Chains/make_surf.lmp",
            "2x1_pandey": "111/bare_surf/2x1_Pandey_Chains/make_surf.lmp",
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
        "make_surf": {
            "bare": "113/bare_surf/make_surf.lmp",
            "O":    "113/O_terminated/make_surf.lmp",
        },
    },
}
