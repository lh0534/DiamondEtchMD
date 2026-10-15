"""
hexagonal.py — hexagonal diamond (lonsdaleite) lattices for the hex* orientations.

LAMMPS has no hexagonal-diamond lattice style, so each orientation is a
`lattice custom` orthohexagonal cell (8 atoms, a x sqrt(3)a x c) with its axes
permuted so the surface normal is z:

  hex0001  basal plane    normal along c          z unit = c
  hex1120  a-plane        normal along a          z unit = a
  hex1010  m-plane        normal along sqrt(3)a   z unit = sqrt(3) a

Like the cubic orientations, the slab is built between the addfix grid
-5/8 - k/2 and the integer lattice top, and the bottom is replenished half a
lattice unit at a time.  The atomic planes repeat every half cell along each
normal, so every replenished layer is identical.  The basis is shifted along z
so that both the top (z = integer) and the grid (z = -1/8 mod 1/2) fall inside
the widest gap between planes: the one that cuts a single bond per surface atom
(bulk-terminated 1x1, one dangling bond per surface carbon).
"""
from typing import Dict, List, Tuple

import numpy as np

# Zero-pressure lonsdaleite with the package ReaxFF (tools/relax_lonsdaleite.py;
# the same procedure gives cubic a = 3.565387 Å = lat_a.txt).  a, c in Å;
# u = vertical (c-axis) bond length / c.  Ideal (from cubic): 2.52111, 4.11695, 0.375.
# Bonds: 1.5455 Å vertical, 1.5471 Å otherwise (cubic 1.5439 Å).  This ReaxFF puts
# lonsdaleite 150 meV/atom above cubic diamond (DFT: ~25-30 meV/atom).
HEX_LATTICE = {"a": 2.524898, "c": 4.127084, "u": 0.374477}

# surface normal and in-plane axes as indices into the orthohexagonal cell (a, sqrt3 a, c)
HEX_AXES = {
    "hex0001": (0, 1, 2),
    "hex1120": (1, 2, 0),
    "hex1010": (2, 0, 1),
}


def bond_lengths(a=None, c=None, u=None):
    """(vertical, other) C-C bond lengths of lonsdaleite with these parameters."""
    a = HEX_LATTICE["a"] if a is None else a
    c = HEX_LATTICE["c"] if c is None else c
    u = HEX_LATTICE["u"] if u is None else u
    return u * c, float(np.hypot(a / np.sqrt(3), (0.5 - u) * c))


def ortho_basis(a: float, c: float, u: float) -> Tuple[np.ndarray, np.ndarray]:
    """(fractional basis (8, 3), cell lengths (3,)) of the orthohexagonal cell."""
    z0 = 0.25 - u / 2
    frac = []
    for f1, f2, f3 in [(1/3, 2/3, z0), (2/3, 1/3, z0 + 0.5), (2/3, 1/3, -z0), (1/3, 2/3, 0.5 - z0)]:
        x, y = f1 - f2 / 2, f2 / 2                      # hexagonal -> orthohexagonal
        for tx, ty in ((0.0, 0.0), (0.5, 0.5)):         # C-centring of the orthohexagonal cell
            frac.append(((x + tx) % 1, (y + ty) % 1, f3 % 1))
    return np.array(frac), np.array([a, np.sqrt(3) * a, c])


def _planes(z: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    zs = np.sort(z % 1)
    keep = np.r_[True, np.diff(zs) > tol]
    return zs[keep]


def _bonds_cut(frac: np.ndarray, L: np.ndarray, zcut: float, cutoff: float = 1.8) -> int:
    """Bonds per cell crossing the plane z = zcut (fractional, mod 1)."""
    n = 0
    for i in range(len(frac)):
        for j in range(len(frac)):
            for s in np.ndindex(3, 3, 3):
                s = np.array(s) - 1
                d = (frac[j] + s - frac[i]) * L
                r = np.linalg.norm(d)
                if 1e-6 < r < cutoff:
                    zi, zj = frac[i, 2], frac[j, 2] + s[2]
                    for m in (zcut - 1, zcut, zcut + 1):
                        if (zi - m) * (zj - m) < 0:
                            n += 1
    return n // 2


def oriented_cell(orientation: str, a: float = None, c: float = None,
                  u: float = None) -> Dict:
    """Shifted basis and cell for one hex orientation.

    Returns dict(cell=(Lx, Ly, Lz) Å, basis=(8, 3) fractions, shift=z shift,
    ml_factor=atoms per atomic plane per cell, top_margin/grid_margin=Å from the
    top cut / grid cut to the nearest plane, planes=fractional plane heights).
    """
    a = HEX_LATTICE["a"] if a is None else a
    c = HEX_LATTICE["c"] if c is None else c
    u = HEX_LATTICE["u"] if u is None else u
    frac0, L0 = ortho_basis(a, c, u)
    ax = list(HEX_AXES[orientation])
    frac, L = frac0[:, ax], L0[ax]

    planes = _planes(frac[:, 2])
    # widest gap that cuts the fewest bonds; both cuts (top at 0, grid at -1/8)
    # go symmetrically inside it
    ext = np.r_[planes, planes[0] + 1]
    gaps = [(ext[k], ext[k + 1]) for k in range(len(planes))]
    best = min(gaps, key=lambda g: (_bonds_cut(frac, L, (g[0] + g[1]) / 2), -(g[1] - g[0])))
    lo, hi = best
    # place the gap centre at -1/16 (midway between the top cut 0 and the grid cut -1/8)
    shift = (-1 / 16 - (lo + hi) / 2) % 1
    frac = frac.copy()
    frac[:, 2] = (frac[:, 2] + shift) % 1
    planes = _planes(frac[:, 2])

    def margin(zc):
        d = np.abs(((planes - zc) + 0.5) % 1 - 0.5)
        return float(d.min() * L[2])

    counts = [int(np.sum(np.abs(((frac[:, 2] - p) + 0.5) % 1 - 0.5) < 1e-6)) for p in planes]
    return dict(cell=tuple(float(x) for x in L), basis=frac, shift=float(shift),
                ml_factor=int(min(counts)), planes=planes,
                top_margin=margin(0.0), grid_margin=margin(-1 / 8),
                bonds_cut_top=_bonds_cut(frac, L, 0.0), plane_counts=counts)


def lattice_args(orientation: str, **kw) -> str:
    """Arguments after `lattice custom 1.0` for this orientation (Å cell vectors)."""
    cfg = oriented_cell(orientation, **kw)
    Lx, Ly, Lz = cfg["cell"]
    return (f"a1 {Lx:.8f} 0 0 a2 0 {Ly:.8f} 0 a3 0 0 {Lz:.8f} "
            + " ".join(f"basis {x:.10f} {y:.10f} {z:.10f}" for x, y, z in cfg["basis"]))
