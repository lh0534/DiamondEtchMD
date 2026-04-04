"""
orientations.py — registry of supported diamond surface orientations.

Each entry in ORIENT contains:

  lattice_cmd       : LAMMPS lattice command matching the make_surf template
  bottom_expr       : LAMMPS expression to convert zlo (Å) to lattice z-units
  default_box       : (x, y, lat_top) in lattice units — sensible starting points
  ml_factor         : ML = ml_factor * x * y
  make_surf         : maps reconstruction key → package-bundled template path
  valid_terminations: maps reconstruction key → set of allowed termination strings

ML factor derivation: atoms/LAMMPS-unit-cell = 8 * spacing_x * spacing_y * spacing_z
                      diamond planes per z-unit = FCC planes * 2
                      ml_factor = atoms_per_cell / planes_per_z_unit

  100: 4 atoms/cell, 4 planes/z-unit → ml_factor = 1   (verified: 9×9 → 81)
  111: 12 atoms/cell, 6 planes/z-unit → ml_factor = 2  (verified: 5×9 → 90)
  113: 20 atoms/cell, 5 planes/z-unit → ml_factor = 4  (verified: 9×3 → 108)

Reconstruction naming convention: <termination>_<periodicity>
  bare_1x1, bare_2x1            (100)
  bare_1x1, bare_2x1_single,
  bare_2x1_pandey               (111)
  bare                          (113)

Termination naming convention:
  bare                          all orientations
  O, O_ether                    100 only
  O_1x1, O_2x1_single,
  O_2x1_pandey                  111 only (reconstruction is implied by termination name)
  O                             113
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
        # single template; reconstruction (bare_1x1/bare_2x1) controlled by
        # the `reconstruct` config flag; termination via O_terminate/O_ether_terminate.
        "make_surf": {
            "bare_1x1": "package:lammps/templates/make_surf_100.lmp",
            "bare_2x1": "package:lammps/templates/make_surf_100.lmp",
        },
        "valid_terminations": {
            "bare_1x1": {"bare", "O", "O_ether"},
            "bare_2x1": {"bare", "O", "O_ether"},
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
        # separate template per reconstruction; O termination handled via
        # O_terminate config flag within each template.
        "make_surf": {
            "bare_1x1":        "package:lammps/templates/make_surf_111_1x1.lmp",
            "bare_2x1_single": "package:lammps/templates/make_surf_111_2x1_single.lmp",
            "bare_2x1_pandey": "package:lammps/templates/make_surf_111_2x1_pandey.lmp",
        },
        "valid_terminations": {
            "bare_1x1":        {"bare", "O_1x1"},
            "bare_2x1_single": {"bare", "O_2x1_single"},
            "bare_2x1_pandey": {"bare", "O_2x1_pandey"},
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
            "bare": "package:lammps/templates/make_surf_113.lmp",
        },
        "valid_terminations": {
            "bare": {"bare", "O"},
        },
    },
}
