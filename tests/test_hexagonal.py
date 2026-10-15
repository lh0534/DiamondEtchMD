"""
tests/test_hexagonal.py — hexagonal diamond (lonsdaleite) orientations, no LAMMPS.

Builds the slab the way make_surf_hex.lmp does (lattice points between the
addfix grid and lat_top) from hexagonal.py's shifted basis and checks:
  - bulk: every atom has 4 neighbours at one bond length;
  - top surface: each surface atom loses exactly one bond (bulk-terminated 1x1);
  - the cuts (top, grid) sit in gaps, and every addfix layer is identical;
  - CNA labels the slab interior as diamond, and still labels cubic diamond.
"""
import numpy as np
import pytest

from diamond_etch_md.hexagonal import HEX_AXES, bond_lengths, oriented_cell, lattice_args
from diamond_etch_md.orientations import ORIENT
from diamond_etch_md.spec import SimSpec, validate, compute_ml

HEX = list(HEX_AXES)
BONDS = bond_lengths()             # (vertical, other)


def _slab(orientation, nx, ny, bottom, top):
    """Atom positions (Å) of lattice points with bottom <= z/Lz <= top, like create_atoms."""
    cfg = oriented_cell(orientation)
    L = np.array(cfg["cell"]); b = cfg["basis"]
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(int(np.floor(bottom)) - 1, int(np.ceil(top)) + 1):
                f = b + np.array([i, j, k])
                keep = (f[:, 2] >= bottom) & (f[:, 2] <= top)
                pts.append(f[keep] * L)
    return np.vstack(pts), L * np.array([nx, ny, 1])


def _coordination(x, box, cutoff=1.8):
    d = x[:, None, :] - x[None, :, :]
    d[..., :2] -= np.round(d[..., :2] / box[:2]) * box[:2]
    r = np.linalg.norm(d, axis=-1)
    return ((r > 0.1) & (r < cutoff)).sum(axis=1), r


@pytest.mark.parametrize("o", HEX)
def test_bulk_is_fourfold_with_one_bond_length(o):
    x, box = _slab(o, 3, 3, -5 / 8, 3)
    n, r = _coordination(x, box)
    z = x[:, 2]; Lz = oriented_cell(o)["cell"][2]
    interior = (z > z.min() + 2.0) & (z < z.max() - 2.0)
    assert interior.sum() > 20 and (n[interior] == 4).all()
    bonds = r[(r > 0.1) & (r < 1.8)]
    assert np.all(np.min(np.abs(bonds[:, None] - np.array(BONDS)[None, :]), axis=1) < 1e-6)
    # each interior atom: one vertical bond, three others
    ri = r[interior]
    assert (np.isclose(ri, BONDS[0]).sum(axis=1) == 1).all() and (np.isclose(ri, BONDS[1]).sum(axis=1) == 3).all()


@pytest.mark.parametrize("o", HEX)
def test_top_surface_has_one_dangling_bond_per_atom(o):
    x, box = _slab(o, 3, 3, -5 / 8, 3)
    n, _ = _coordination(x, box)
    z = x[:, 2]
    top = z > z.max() - 0.05                       # topmost plane
    assert (n[top] == 3).all()
    # nothing else near the top is under-coordinated (the cut breaks single bonds only)
    near = (z > z.max() - 2.5) & ~top
    assert (n[near] == 4).all()


@pytest.mark.parametrize("o", HEX)
def test_cuts_in_gaps_and_identical_addfix_layers(o):
    cfg = oriented_cell(o)
    assert cfg["top_margin"] > 0.4 and cfg["grid_margin"] > 0.4      # Å to nearest plane
    L = np.array(cfg["cell"])
    layers = []
    for k in range(4):                              # addfix: [bottom, bottom + 1/2], bottom = -5/8 - k/2
        lo = -5 / 8 - k / 2
        x, _ = _slab(o, 2, 2, lo, lo + 0.5)
        x[:, 2] -= lo * L[2]
        layers.append(x[np.lexsort(x.T[::-1])])
    for lay in layers[1:]:
        assert len(lay) == len(layers[0])
        # same heights; lateral positions may be shifted by a lattice translation
        np.testing.assert_allclose(np.sort(lay[:, 2]), np.sort(layers[0][:, 2]), atol=1e-9)
    # one ML per atomic plane (hex1120: one plane per half cell; the others two)
    per_cell = len(layers[0]) / 4
    assert per_cell == len(cfg["planes"]) // 2 * ORIENT[o]["ml_factor"]


@pytest.mark.parametrize("o", HEX)
def test_cna_labels_hexagonal_diamond(o):
    from diamond_etch_md.analysis.cna import diamond_cna, DIAMOND
    x, box = _slab(o, 4, 4, -5 / 8, 4)
    data = {"x": x[:, 0], "y": x[:, 1], "z": x[:, 2], "types": np.ones(len(x), dtype=np.int32),
            "box": np.array([[0, box[0]], [0, box[1]], [x[:, 2].min(), x[:, 2].max()]])}
    lab = diamond_cna(data)
    z = x[:, 2]
    interior = (z > z.min() + 4.0) & (z < z.max() - 4.0)
    assert interior.sum() > 20 and (lab[interior] == DIAMOND).all()


def test_cna_still_labels_cubic_and_rejects_random():
    from diamond_etch_md.analysis.cna import diamond_cna, DIAMOND
    a = 3.5653871687699
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    fcc = np.vstack([base, base + 0.25])
    x = np.vstack([(fcc + [i, j, k]) * a for i in range(4) for j in range(4) for k in range(4)])
    data = {"x": x[:, 0], "y": x[:, 1], "z": x[:, 2], "types": np.ones(len(x), dtype=np.int32),
            "box": np.array([[0, 4 * a], [0, 4 * a], [0, 4 * a]])}
    lab = diamond_cna(data)
    interior = (x[:, 2] > a) & (x[:, 2] < 3 * a)
    assert (lab[interior] == DIAMOND).all()
    rng = np.random.default_rng(0)
    xr = rng.random((400, 3)) * [12, 12, 12]
    data = {"x": xr[:, 0], "y": xr[:, 1], "z": xr[:, 2], "types": np.ones(400, dtype=np.int32),
            "box": np.array([[0, 12], [0, 12], [0, 12]])}
    assert (diamond_cna(data) == DIAMOND).sum() == 0


@pytest.mark.parametrize("o", HEX)
def test_generated_inputs(o, tmp_path):
    from diamond_etch_md.lammps.config import get_config_lmp
    from diamond_etch_md.lammps.head import get_head_lmp
    bx, by, bz = ORIENT[o]["default_box"]
    spec = SimSpec(orientation=o, surface="1x1", box_x=bx, box_y=by, box_depth=bz,
                   ml=compute_ml(o, bx, by), species="Ar", energy=50.0)
    validate(spec)
    cfg = get_config_lmp(spec)
    assert f'variable    hex_lattice string "{lattice_args(o)}"' in cfg
    assert f"variable    hex_zu equal {oriented_cell(o)['cell'][2]:.8f}" in cfg
    head = get_head_lmp(spec)
    assert "lattice    custom 1.0 ${hex_lattice}" in head
    assert "v_hex_zu" in head and "lattice    diamond" not in head
    from diamond_etch_md import make_sim
    make_sim(spec, tmp_path / o)
    surf = (tmp_path / o / "make_surf.lmp").read_text()
    assert "lattice    custom 1.0 ${hex_lattice}" in surf and "lat_az" not in surf


def test_cubic_config_has_no_hex_vars():
    from diamond_etch_md.lammps.config import get_config_lmp
    assert "hex_lattice" not in get_config_lmp(SimSpec(species="Ar", energy=50.0))
