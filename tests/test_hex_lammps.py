"""
tests/test_hex_lammps.py — hexagonal diamond lattices in real LAMMPS (pair_style zero).

For each hex orientation, builds the slab with the generated `lattice custom`
command exactly as make_surf_hex.lmp does, then replays addfix.lmp's bottom
replenishment (bottom -= 1/2, change_box down half a lattice unit, create_atoms
in the new anchor slab) three times, and checks:
  - LAMMPS puts atoms where hexagonal.py predicts (lattice units = cell lengths);
  - every replenished layer bonds onto the slab: interior atoms have 4 neighbours
    at the one bond length and no two atoms overlap;
  - the restart bottom snap (orientations.restart_bottom_lines) recovers the grid.

Run with:
    /home/lh0534/.conda/envs/myenv/bin/python -m pytest tests/test_hex_lammps.py -m lammps -v
"""
import numpy as np
import pytest

from diamond_etch_md.hexagonal import HEX_AXES, bond_lengths, lattice_args, oriented_cell
from diamond_etch_md.orientations import ORIENT, restart_bottom_lines

lammps_mod = pytest.importorskip("lammps", reason="LAMMPS Python module not available")
pytestmark = pytest.mark.lammps

BONDS = bond_lengths()             # (vertical, other)


def _positions(L):
    n = L.get_natoms()
    x = np.array([[L.extract_atom("x")[i][k] for k in range(3)] for i in range(n)])
    return x


def _coord(x, box_xy):
    d = x[:, None, :] - x[None, :, :]
    d[..., :2] -= np.round(d[..., :2] / box_xy) * box_xy
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    return ((r < 1.8)).sum(axis=1), r


@pytest.mark.parametrize("o", list(HEX_AXES))
def test_hex_slab_and_addfix_in_lammps(o, tmp_path):
    nx, ny, top = 3, 3, 3
    cfg = oriented_cell(o)
    L = lammps_mod.lammps(cmdargs=["-log", str(tmp_path / "log.lammps"), "-screen", "none", "-nocite"])
    for cmd in [
        "units real", "atom_style charge", "boundary p p m",
        f'variable hex_lattice string "{lattice_args(o)}"',
        f"variable hex_zu equal {cfg['cell'][2]:.8f}",
        "variable bottom equal -5/8",
        ORIENT[o]["lattice_cmd"],
        f"region bbox block 0 {2 * nx} 0 {2 * ny} ${{bottom}} {top + 5} units lattice",
        "create_box 3 bbox",
        f"region lbox block -1e-6 {nx} -1e-6 {ny} ${{bottom}} {top} units lattice",
        "create_atoms 1 region lbox",
        f"change_box all x final 0 {nx} y final 0 {ny} z final ${{bottom}} {top + 5} units lattice",
        "mass * 12.011", "pair_style zero 3.0", "pair_coeff * *",
        "delete_atoms overlap $(sqrt(3)/8) all all",
        "variable ML equal " + str(ORIENT[o]["ml_factor"] * nx * ny),
    ]:
        L.command(cmd)
    Lc = np.array(cfg["cell"])
    box_xy = Lc[:2] * [nx, ny]
    x0 = _positions(L)
    # atoms where hexagonal.py predicts: z/Lz mod 1 on the predicted planes
    zf = (x0[:, 2] / Lc[2]) % 1
    assert np.all(np.min(np.abs(((zf[:, None] - cfg["planes"][None, :]) + 0.5) % 1 - 0.5), axis=1) < 1e-6)
    n_halves = int((top + 5 / 8) * 2)                 # whole half-cells; the rest is gap

    for _ in range(3):                                 # addfix.lmp, three times
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
    added = len(x) - len(x0)
    assert added * n_halves == 3 * len(x0)              # three half-cell layers like the slab's
    n, r = _coord(x, box_xy)
    assert r.min() > min(BONDS) - 1e-4                  # no overlaps
    bonds = r[r < 1.8]
    assert np.all(np.min(np.abs(bonds[:, None] - np.array(BONDS)[None, :]), axis=1) < 1e-4)
    z = x[:, 2]
    interior = (z > z.min() + 1.0) & (z < z.max() - 1.0)
    assert (n[interior] == 4).all()

    # restart: the snapped bottom recovers the current grid value
    L.command("variable bottom_now equal ${bottom}")
    snap = tmp_path / "snap.lmp"                       # jump SELF needs a file
    snap.write_text(restart_bottom_lines(o))
    L.file(str(snap))
    assert abs(L.extract_variable("bottom", None, 0) - L.extract_variable("bottom_now", None, 0)) < 1e-9
    L.close()


# (orientation, box_depth, species, eV): the threshold scan's ~10 Å slabs with Ar+,
# and its thinnest slabs (5-6 atomic planes, bottom half-cell frozen) with O+
E2E_CASES = [("hex0001", 2, "Ar", 40.0), ("hex1120", 4, "Ar", 40.0), ("hex1010", 2, "Ar", 40.0),
             ("hex0001", 1, "O", 10.0), ("hex1120", 2, "O", 10.0), ("hex1010", 1, "O", 10.0)]


@pytest.mark.parametrize("o,depth,species,energy", E2E_CASES)
def test_e2e_hex_ion_etch(o, depth, species, energy, tmp_path):
    """make_surf + head.lmp on a small hex slab: four ion impacts (ReaxFF; + ZBL for Ar).

    Checks the run finishes cleanly, logs every impact, and starts from the
    snapped bottom grid (-5/8)."""
    from tests.test_burst_sampling import _needs_2022_build, _set_var, _run_sim
    _needs_2022_build()
    from diamond_etch_md import SimSpec, compute_ml, make_sim, validate
    bx, by = {"hex0001": (4, 4), "hex1120": (2, 4), "hex1010": (4, 4)}[o]   # ML = 32
    spec = SimSpec(orientation=o, surface="1x1", species=species, energy=energy,
                   box_x=bx, box_y=by, box_depth=depth, ml=compute_ml(o, bx, by), fluence=1,
                   impact_time=300.0, thermalization_time=100.0, dump_mode="none",
                   name=f"t_hex_{o}_{species}")
    validate(spec)
    d = tmp_path / f"{o}_{species}"
    make_sim(spec, d)
    assert spec.ml == 32
    _set_var(d / "config.lmp", "end_fluence", 0.125)                # four impacts (exact)
    nc = _run_sim_ion(d, _run_sim)
    log = (d / "log1.lammps").read_text()
    assert "ERROR" not in log
    assert "BOTTOM -0.625" in log
    assert sorted({int(r[0]) for r in nc}) == [0, 1, 2, 3, 4]   # ion-only: one row per impact


def _run_sim_ion(d, run_sim):
    """_run_sim expects radical_log.txt; ion-only runs have none."""
    (d / "radical_log.txt").write_text("0 0 0 0 0 1\n")
    nc, _ = run_sim(d)
    return nc
