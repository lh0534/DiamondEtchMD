"""
orientations.py — registry of supported diamond surface orientations.

Each entry in ORIENT contains:

  lattice_cmd  : LAMMPS lattice command matching the make_surf template
  bottom_expr  : LAMMPS expression for the lattice-unit bottom of the slab: the
                 lowest C atom snapped onto the addfix grid -5/8 - k/2 (make_surf
                 starts at -5/8; addfix steps 1/2).  Using zlo instead breaks on
                 restart when any atom was below the slab as the snapshot was
                 written: bottom goes off-grid and addfix then creates layers
                 overlapping the lattice.  Needs group carbon defined first.
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
        "bottom_expr": "$(-5/8-0.5*ceil(2*(-5/8-bound(carbon,zmin)/(v_lat_a))-1e-9))",
        "z_unit":      "v_lat_a",   # Å per lattice z-unit
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
        "bottom_expr": "$(-5/8-0.5*ceil(2*(-5/8-bound(carbon,zmin)/(v_lat_a*3/sqrt(2)))-1e-9))",
        "z_unit":      "v_lat_a*3/sqrt(2)",   # Å per lattice z-unit
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
        "bottom_expr": "$(-5/8-0.5*ceil(2*(-5/8-bound(carbon,zmin)/(v_lat_a*sqrt(3)))-1e-9))",
        "z_unit":      "v_lat_a*sqrt(3)",   # Å per lattice z-unit
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
        "bottom_expr": "$(-5/8-0.5*ceil(2*(-5/8-bound(carbon,zmin)/(v_lat_a*5/sqrt(11)))-1e-9))",
        "z_unit":      "v_lat_a*5/sqrt(11)",   # Å per lattice z-unit
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


def restart_bottom_lines(orientation: str) -> str:
    """LAMMPS lines that set `bottom` (lattice z-units) on a (re)start.

    bottom_expr snaps the lowest C atom onto the addfix grid; a stray atom just
    below the slab would then put the anchor region under the real anchor
    planes, so move bottom up half a lattice until [bottom, bottom+1/2] holds at
    least half a monolayer of C (strays are a few atoms; an anchor layer is ~2 ML).
    """
    cfg = ORIENT[orientation]
    zu = cfg["z_unit"]
    return (
        "group       carbon type 1\n"
        f"variable    bottom equal {cfg['bottom_expr']}\n"
        "label       bottom_search\n"
        f"region      btest block INF INF INF INF $(v_bottom*{zu}) $((v_bottom+0.5)*{zu}) units box\n"
        "group       btest region btest\n"
        "variable    nbtest equal $(count(btest))\n"
        "group       btest delete\n"
        "region      btest delete\n"
        f'if "${{nbtest}} < $(0.5*v_ML) && $(v_bottom*{zu}) < $(zhi)" then '
        '"variable bottom equal $(v_bottom+0.5)" "jump SELF bottom_search"\n'
    )
