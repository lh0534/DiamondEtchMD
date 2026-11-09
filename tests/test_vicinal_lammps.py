"""
tests/test_vicinal_lammps.py — vicinal (miscut) orientations in real LAMMPS.

  1. The generated `lattice` command makes exactly the lattice points vicinal.py
     builds the stepped slab from (pair_style zero).
  2. addfix replenishment under the Python slab: three half-unit layers bond on,
     nothing overlaps, and the restart bottom snap recovers the grid.
  3. End to end (LAMMPS 23 Jun 2022 with REAXFF): make_surf_vicinal.lmp + four
     Ar+ 40 eV impacts through head.lmp for 119 2x1_DA, 119 2x1_DB, 557 and 775.

Run with:
    /home/lh0534/.conda/envs/myenv/bin/python -m pytest tests/test_vicinal_lammps.py -m lammps -v
"""
import numpy as np
import pytest

from diamond_etch_md.orientations import ORIENT, restart_bottom_lines
from diamond_etch_md.vicinal import LAT_A, VICINAL, build_surface, geometry, lattice_points

lammps_mod = pytest.importorskip("lammps", reason="LAMMPS Python module not available")
pytestmark = pytest.mark.lammps


def _lmp(tmp_path):
    return lammps_mod.lammps(cmdargs=["-log", str(tmp_path / "log.lammps"), "-screen", "none", "-nocite"])


def _setup(L, o, nx, ny, top):
    g = geometry(o)
    for cmd in [
        "units real", "atom_style charge", "boundary p p m",
        f"variable lat_a equal {LAT_A}",
        f'variable vic_lattice string "{g["lattice_args"]}"',
        f"variable vic_zu equal {g['s']:.8f}",
        "variable bottom equal -5/8",
        ORIENT[o]["lattice_cmd"],
        f"region bbox block 0 {nx} 0 {ny} ${{bottom}} {top + 5} units lattice",
        "create_box 3 bbox",
        "mass * 12.011", "pair_style zero 3.0", "pair_coeff * *",
    ]:
        L.command(cmd)
    return g


def _positions(L):
    n = L.get_natoms()
    return np.array([[L.extract_atom("x")[i][k] for k in range(3)] for i in range(n)])


def _wrap(x, box):
    x = x.copy()
    x[:, :2] %= box
    x[x[:, 0] >= box[0] - 1e-6, 0] -= box[0]
    x[x[:, 1] >= box[1] - 1e-6, 1] -= box[1]
    return x


@pytest.mark.parametrize("o", list(VICINAL))
def test_lammps_lattice_matches_python(o, tmp_path):
    nx, ny, top = 1, 3, 2
    L = _lmp(tmp_path)
    g = _setup(L, o, nx, ny, top)
    L.command(f"region lbox block -1e-6 {nx} -1e-6 {ny} ${{bottom}} {top} units lattice")
    L.command("create_atoms 1 region lbox")
    L.command("delete_atoms overlap 0.1 all all")              # x = Lx / y = Ly duplicates
    box = np.array([nx * g["Lx"], ny * g["Ly"]])
    xl = _wrap(_positions(L), box)
    L.close()
    xp, _ = lattice_points(o, nx, ny, -5 / 8 * g["s"], top * g["s"])
    xp = _wrap(xp, box)
    assert len(xl) == len(xp)
    d = xl[:, None, :] - xp[None, :, :]
    d[..., :2] -= np.round(d[..., :2] / box) * box
    assert np.linalg.norm(d, axis=-1).min(axis=1).max() < 1e-5   # every LAMMPS atom is a Python point


@pytest.mark.parametrize("o,surf", [("119", "2x1_DA"), ("119", "2x1_DB"), ("557", "1x1"), ("775", "1x1")])
def test_addfix_under_python_slab(o, surf, tmp_path):
    nx, ny, top = 1, 4, 3
    b = build_surface(o, surf, nx, ny, top)
    L = _lmp(tmp_path)
    g = _setup(L, o, nx, ny, top)
    path = tmp_path / "slab.data"
    with open(path, "w") as fh:
        fh.write(f"slab\n\n{len(b['x'])} atoms\n3 atom types\n\n")
        for k, ax in enumerate("xyz"):
            fh.write(f"{b['box'][k, 0]:.8f} {b['box'][k, 1]:.8f} {ax}lo {ax}hi\n")
        fh.write("\nAtoms # charge\n\n")
        for i, p in enumerate(b["x"], start=1):
            fh.write(f"{i} 1 0.0 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    L.command(f"read_data {path} add merge")
    L.command(f"variable ML equal {round(g['ml_per_cell'] * nx * ny)}")
    n0 = L.get_natoms()
    for _ in range(3):                                          # addfix.lmp, three times
        for cmd in [
            "variable bottom equal ${bottom}-1/2",
            "change_box all z delta -0.5 0 units lattice",
            "variable sublat equal ${bottom}+1/2",
            "region anchor block INF INF INF INF ${bottom} ${sublat} units lattice",
            "create_atoms 1 region anchor",
            "region anchor delete",
        ]:
            L.command(cmd)
    x = _positions(L)
    box = np.array([nx * g["Lx"], ny * g["Ly"]])
    d = x[:, None, :] - x[None, :, :]
    d[..., :2] -= np.round(d[..., :2] / box) * box
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    assert r.min() > 1.4                                        # no overlaps (dimers are 1.6 Å)
    coord = (r < 1.8).sum(axis=1)
    z = x[:, 2]
    bulk = (z > z.min() + 2.0) & (z < -g["s"] / 8)              # replenished layers, away from the new bottom
    assert bulk.sum() > 20 and (coord[bulk] == 4).all()
    added = L.get_natoms() - n0
    assert added == pytest.approx(3 * 0.5 * g["s"] / LAT_A ** 3 * 8 * box[0] * box[1], rel=0.02)
    L.command("variable bottom_now equal ${bottom}")
    snap = tmp_path / "snap.lmp"                               # jump SELF needs a file
    snap.write_text(restart_bottom_lines(o))
    L.file(str(snap))
    assert abs(L.extract_variable("bottom", None, 0) - L.extract_variable("bottom_now", None, 0)) < 1e-9
    L.close()


@pytest.mark.parametrize("o,surf", [("119", "2x1_DA"), ("119", "2x1_DB"), ("557", "1x1"), ("775", "1x1")])
def test_e2e_vicinal_ion_etch(o, surf, tmp_path):
    """make_surf_vicinal.lmp + head.lmp: four Ar+ 40 eV impacts (ReaxFF + ZBL)."""
    from tests.test_burst_sampling import _needs_2022_build, _set_var, _run_sim
    _needs_2022_build()
    from diamond_etch_md import SimSpec, compute_ml, make_sim, validate
    spec = SimSpec(orientation=o, surface=surf, species="Ar", energy=40.0,
                   box_x=1, box_y=4, box_depth=3, ml=compute_ml(o, 1, 4), fluence=1,
                   impact_time=300.0, thermalization_time=100.0, dump_mode="none",
                   name=f"t_vic_{o}_{surf}")
    validate(spec)
    d = tmp_path / f"{o}_{surf}"
    make_sim(spec, d)
    _set_var(d / "config.lmp", "end_fluence", repr(4 / spec.ml))   # four impacts
    (d / "radical_log.txt").write_text("0 0 0 0 0 1\n")           # _run_sim reads it; ion-only run
    nc, _ = _run_sim(d)
    log = (d / "log1.lammps").read_text()
    assert "ERROR" not in log and "BOTTOM -0.625" in log
    assert sorted({int(r[0]) for r in nc}) == [0, 1, 2, 3, 4]   # ion-only: one row per impact
    surf_log = (d / "log_make_surf.lammps").read_text()
    assert "ERROR" not in surf_log
    n_data = int(next(l.split()[0] for l in (d / "vicinal_surface.data").read_text().splitlines()
                      if l.strip().endswith(" atoms")))
    assert int(nc[0][2]) == n_data                               # make_surf kept every slab atom
