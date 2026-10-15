"""
tests/test_vicinal.py — vicinal (miscut) orientations 119 / 557 / 775, no LAMMPS.

  - lattice: periods, miscut angles, and every addfix grid line (-5/8 - k/2,
    lattice units) between atomic planes;
  - surfaces: no loose atoms, the intended step structure (2x1_DA / 2x1_DB: double
    steps with dimer bonds along x / y; 557 / 775 edge atoms with 2 / 1 dangling bonds);
  - generated inputs: config vic_lattice/vic_zu, head lattice, the slab data file.
"""
import itertools

import numpy as np
import pytest

from diamond_etch_md.orientations import ORIENT
from diamond_etch_md.spec import SimSpec, compute_ml, validate
from diamond_etch_md.vicinal import (LAT_A, VICINAL, build_surface, geometry, lattice_points,
                                     surfaces, _dimer_axis, _BASIS)

VIC = list(VICINAL)


@pytest.mark.parametrize("o,miscut,Lx", [("119", 8.93, 22.97), ("557", 9.45, 25.08), ("775", 8.47, 27.96)])
def test_geometry(o, miscut, Lx):
    g = geometry(o)
    assert g["miscut"] == pytest.approx(miscut, abs=0.01)
    assert g["Lx"] == pytest.approx(Lx, abs=0.01)
    assert g["Ly"] == pytest.approx(LAT_A / np.sqrt(2))
    assert 3.5 < g["s"] < 4.5 and g["margin"] > 0.1
    assert g["per_period"] == (4 if g["dimers"] else 2)


@pytest.mark.parametrize("o", VIC)
def test_grid_lines_fall_between_atomic_planes(o):
    g = geometry(o)
    R = g["R"]
    k = 8
    cells = np.array(list(itertools.product(range(-k, k + 1), repeat=3)), float)
    z = np.unique(np.round(((cells[:, None, :] + (_BASIS + g["origin"])[None]).reshape(-1, 3) @ R[2]) * LAT_A, 6))
    lines = (-5 / 8 - np.arange(0, 40) / 2) * g["s"]
    gap = np.min(np.abs(z[None, :] - lines[:, None]), axis=1)
    assert gap.min() > 0.1                                   # Å to the nearest atomic plane


@pytest.mark.parametrize("o", VIC)
def test_lattice_is_periodic_in_x_and_y(o):
    g = geometry(o)
    x1, _ = lattice_points(o, 1, 2, -5.0, 5.0)
    x2, _ = lattice_points(o, 2, 2, -5.0, 5.0)
    shifted = x1 + [g["Lx"], 0, 0]
    d = np.linalg.norm(x2[None, :, :] - shifted[:, None, :], axis=-1)
    assert np.all(d.min(axis=1) < 1e-6)                      # the next x period repeats the first


def _surface_terraces(o, surf, ny=6):
    """Exposed atoms grouped into terraces along x: [(x range, mean height along the
    terrace normal, n atoms, dimer axis)] in order of x."""
    g = geometry(o)
    b = build_surface(o, surf, 1, ny, 4)
    x, c, dm = b["x"], b["coordination"], b["dimer"]
    box = b["box"][:2, 1]
    top = x[:, 2] > x[:, 2].max() - 4
    m = top & ((c < 4) | (dm > 0))
    h = x @ (g["R"] @ g["terrace"])
    idx = np.where(m)[0][np.argsort(x[m, 0])]
    groups, cur = [], [idx[0]]
    for a, b_ in zip(idx[:-1], idx[1:]):
        if abs(h[b_] - h[a]) > g["layer"] / 2 and abs(x[b_, 0] - x[a, 0]) > 0.05:
            groups.append(cur); cur = [b_]
        else:
            cur.append(b_)
    groups.append(cur)
    # a terrace split by the x boundary: its part at x ~ 0 is the image of the part at
    # x ~ Lx one period (per_period layers) higher
    P = g["per_period"] * g["layer"]
    hh = {i: h[i] for i in idx}
    if len(groups) > 1 and abs(np.mean(h[groups[0]]) - P - np.mean(h[groups[-1]])) < g["layer"] / 2:
        for i in groups[0]:
            hh[i] = h[i] - P
        groups = groups[1:-1] + [groups[-1] + groups[0]]
    out = []
    for gr in groups:
        sub = np.zeros(len(x), int); sub[gr] = dm[gr]
        out.append((len(gr), float(np.mean([hh[i] for i in gr])), _dimer_axis(x, sub, box) if sub.sum() else None))
    return b, out


@pytest.mark.parametrize("o", VIC)
def test_surfaces_have_no_loose_atoms(o):
    g = geometry(o)
    for surf in surfaces(o):
        b = build_surface(o, surf, 1, 4, 3)
        above = b["x"][:, 2] > -g["s"] / 8
        assert (b["coordination"][above] >= 2).all(), surf
        Lx, Ly = b["box"][0, 1], b["box"][1, 1]
        assert (b["x"][:, 0] >= 0).all() and (b["x"][:, 0] < Lx).all()
        assert (b["x"][:, 1] >= 0).all() and (b["x"][:, 1] < Ly).all()


@pytest.mark.parametrize("surf,axis", [("2x1_DA", "x"), ("2x1_DB", "y")])
def test_119_double_steps(surf, axis):
    g = geometry("119")
    b, terr = _surface_terraces("119", surf)
    big = [t for t in terr if t[0] >= 18]
    assert len(big) == 2                                     # two double steps per period
    assert abs(big[1][1] - big[0][1]) == pytest.approx(2 * g["layer"], abs=0.05)
    assert all(t[2] == axis for t in big)
    top = b["x"][:, 2] > b["x"][:, 2].max() - 4
    surf_atoms = top & ((b["coordination"] < 4) | (b["dimer"] > 0))
    assert (b["coordination"][surf_atoms] == 3).all()        # one dangling bond per surface atom


@pytest.mark.parametrize("o,n2", [("557", 2), ("775", 0)])
def test_111_vicinal_step_edges(o, n2):
    ny = 6
    b = build_surface(o, "1x1", 1, ny, 4)
    c = b["coordination"]
    top = b["x"][:, 2] > b["x"][:, 2].max() - 4
    assert int(((c == 2) & top).sum()) == n2 * ny            # per step-edge site: 2 steps x 1 atom
    assert not ((c <= 1) & top).any()


@pytest.mark.parametrize("o", VIC)
def test_generated_inputs(o, tmp_path):
    from diamond_etch_md import make_sim
    from diamond_etch_md.lammps.config import get_config_lmp
    from diamond_etch_md.lammps.head import get_head_lmp
    surf = surfaces(o)[-1]
    spec = SimSpec(orientation=o, surface=surf, box_x=1, box_y=4, box_depth=3,
                   ml=compute_ml(o, 1, 4), species="Ar", energy=50.0)
    validate(spec)
    g = geometry(o)
    cfg = get_config_lmp(spec)
    assert f'variable    vic_lattice string "{g["lattice_args"]}"' in cfg
    assert f"variable    vic_zu equal {g['s']:.8f}" in cfg
    head = get_head_lmp(spec)
    assert "lattice    ${vic_lattice}" in head and "v_vic_zu" in head
    make_sim(spec, tmp_path / o)
    data = (tmp_path / o / "vicinal_surface.data").read_text().splitlines()
    n = int(next(l.split()[0] for l in data if l.strip().endswith(" atoms")))
    assert n == len(build_surface(o, surf, 1, 4, 3)["x"])
    assert "read_data       vicinal_surface.data add merge" in (tmp_path / o / "make_surf.lmp").read_text()
    assert ORIENT[o]["surfaces"][surf]["template"].endswith("make_surf_vicinal.lmp")
