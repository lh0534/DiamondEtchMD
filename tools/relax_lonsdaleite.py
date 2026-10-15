"""Relax bulk lonsdaleite (hexagonal diamond) and cubic diamond with the package ReaxFF.

Prints the zero-pressure lattice parameters this force field gives:
  cubic:       a (compare with templates/lat_a.txt)
  hexagonal:   a, c, c/a and the internal parameter u (vertical bond / c)
and the energy difference per atom.  The hexagonal values go into
orientations.HEX_LATTICE.

Run (one CPU core is plenty; myenv has LAMMPS 23 Jun 2022 with REAXFF):
    python tools/relax_lonsdaleite.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from lammps import lammps

FFIELD = Path(__file__).resolve().parents[1] / "diamond_etch_md/lammps/templates/ffield.reax"
LAT_A = float((FFIELD.parent / "lat_a.txt").read_text())
NX, NY, NZ = 8, 5, 5          # orthohexagonal cells: ~20 x 22 x 21 Å, 1600 atoms


def _common(L):
    for cmd in [
        "units real", "atom_style charge", "boundary p p p",
    ]:
        L.command(cmd)


def _potential(L):
    for cmd in [
        "mass 1 12.011",
        "pair_style reaxff NULL",
        f"pair_coeff * * {FFIELD} C",
        "fix qeq all qeq/reaxff 1 0.0 6.0 1e-6 reaxff",       # as in make_surf
        "thermo_style custom step pe lx ly lz pxx pyy pzz",
        "thermo 50",
    ]:
        L.command(cmd)


def _relax(L, mode):
    L.command(f"fix br all box/relax {mode} 0.0 vmax 0.002")
    L.command("min_style cg")
    L.command("minimize 1e-12 1e-10 5000 50000")
    L.command("unfix br")
    L.command("minimize 1e-12 1e-10 5000 50000")     # relax positions at the final box


def relax_cubic():
    L = lammps(cmdargs=["-log", "relax_cubic.log", "-screen", "none", "-nocite"])
    _common(L)
    L.command(f"lattice diamond {LAT_A}")
    L.command("region b block 0 5 0 5 0 5")
    L.command("create_box 1 b")
    L.command("create_atoms 1 box")
    _potential(L)
    _relax(L, "iso")
    a = L.get_thermo("lx") / 5
    e = L.get_thermo("pe") / L.get_natoms()
    L.close()
    return a, e


def relax_hex():
    a0, c0, u0 = LAT_A / np.sqrt(2), 2 * LAT_A / np.sqrt(3), 3 / 8
    z0 = 0.25 - u0 / 2
    basis = []
    for f1, f2, f3 in [(1/3, 2/3, z0), (2/3, 1/3, z0 + 0.5), (2/3, 1/3, -z0), (1/3, 2/3, 0.5 - z0)]:
        # hexagonal -> orthohexagonal fractions (cell a x sqrt3 a x c), plus the centring
        x, y = f1 - f2 / 2, f2 / 2
        for tx, ty in ((0, 0), (0.5, 0.5)):
            basis.append(((x + tx) % 1, (y + ty) % 1, f3 % 1))
    L = lammps(cmdargs=["-log", "relax_hex.log", "-screen", "none", "-nocite"])
    _common(L)
    L.command("lattice custom 1.0 a1 {} 0 0 a2 0 {} 0 a3 0 0 {} ".format(a0, np.sqrt(3) * a0, c0)
              + " ".join(f"basis {x:.10f} {y:.10f} {z:.10f}" for x, y, z in basis))
    L.command(f"region b block 0 {NX} 0 {NY} 0 {NZ}")
    L.command("create_box 1 b")
    L.command("create_atoms 1 box")
    _potential(L)
    _relax(L, "aniso")
    lx, ly, lz = (L.get_thermo(k) for k in ("lx", "ly", "lz"))
    e = L.get_thermo("pe") / L.get_natoms()
    n = L.get_natoms()
    x = np.array([[L.extract_atom("x")[i][k] for k in range(3)] for i in range(n)])
    L.close()
    a, a_y, c = lx / NX, ly / NY / np.sqrt(3), lz / NZ
    # u = vertical (c-axis) bond length / c: pairs with nearly equal x, y
    box = np.array([lx, ly, lz])
    d = x[:, None, :] - x[None, :, :]
    d -= np.round(d / box) * box
    r = np.linalg.norm(d, axis=-1)
    vert = (r > 1.0) & (r < 1.8) & (np.hypot(d[..., 0], d[..., 1]) < 0.1)
    bonds = (r > 1.0) & (r < 1.8)
    u = np.abs(d[..., 2][vert]).mean() / c
    return dict(a=a, a_from_y=a_y, c=c, c_over_a=c / a, u=u,
                bond_vertical=float(np.abs(d[..., 2][vert]).mean()),
                bond_other=float(r[bonds & ~vert].mean()),
                coordination=sorted(set(bonds.sum(axis=1).tolist())), e_per_atom=e)


if __name__ == "__main__":
    a_cub, e_cub = relax_cubic()
    hexp = relax_hex()
    out = dict(cubic=dict(a=a_cub, lat_a_txt=LAT_A, e_per_atom=e_cub),
               hexagonal=hexp,
               ideal_hexagonal=dict(a=a_cub / np.sqrt(2), c=2 * a_cub / np.sqrt(3), u=0.375,
                                    c_over_a=np.sqrt(8 / 3)),
               dE_hex_minus_cubic_meV_per_atom=(hexp["e_per_atom"] - e_cub) / 23.0605 * 1000)
    print(json.dumps(out, indent=2, default=float))
    Path("relax_lonsdaleite.json").write_text(json.dumps(out, indent=2, default=float))
    sys.exit(0)
