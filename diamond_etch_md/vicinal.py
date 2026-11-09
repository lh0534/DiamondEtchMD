"""
vicinal.py — miscut (vicinal) diamond surfaces: the 119 / 557 / 775 orientations.

The cubic lattice is rotated so z is the vicinal normal, x the downhill direction
across the steps and y along the step edges:

  119  8.93° off (001) toward [110]: double-layer steps (a/2), 2x1-reconstructed terraces
  557  9.45° off (111) toward [-1-12]: bilayer steps, edge atoms with 2 dangling bonds
  775  8.47° off (111) toward [11-2]: bilayer steps, edge atoms with 1 dangling bond

box_x counts periods along x (each drops 4 (001) layers / 2 (111) bilayers, so the
steps connect through the x boundary), box_y counts a/sqrt2 cells along the edges.

The LAMMPS lattice (`lattice diamond ... orient ... spacing ... origin ...`) is used
for the anchor and addfix regions.  Vicinal crystals have no wide gaps between
atomic planes, so the z lattice unit is 2q FCC plane spacings (~4 Å, half = one
replenished/anchor layer ~2 Å) and the origin is shifted along the normal so every
boundary of the addfix grid (-5/8 - k/2) falls in the widest gap.

The stepped top is built here, from the same lattice points LAMMPS generates, and
make_sim writes it to vicinal_surface.data for make_surf_vicinal.lmp to read:

  1x1     557/775: flat cut (bilayer steps), bulk-terminated
  2x1_DA  119: double-layer steps, every terrace's dimer rows parallel to the
          step edge (Chadi D_A); the odd row's unpaired atom sits at each step's foot
  2x1_DB  119: double-layer steps, dimer rows perpendicular to the edge (D_B)

Dimers are pre-formed (pairs moved to 1.6 Å); make_surf minimizes.
"""
import itertools
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_TEMPLATES = Path(__file__).parent / "lammps" / "templates"
LAT_A = float((_TEMPLATES / "lat_a.txt").read_text())

_BASIS = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], float)
_BASIS = np.vstack([_BASIS, _BASIS + .25])
_BONDS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / 4   # sublattice A

# normal, x (downhill), y (along the step edges), terrace normal, half-unit in FCC plane spacings
VICINAL = {
    "119": dict(normal=(1, 1, 9), xdir=(9, 9, -2), ydir=(-1, 1, 0), terrace=(0, 0, 1), q=5),
    "557": dict(normal=(5, 5, 7), xdir=(-7, -7, 10), ydir=(1, -1, 0), terrace=(1, 1, 1), q=6),
    "775": dict(normal=(7, 7, 5), xdir=(5, 5, -14), ydir=(-1, 1, 0), terrace=(1, 1, 1), q=6),
}
STEP_OFFSET = 1.0       # Å; double steps sit between atom rows, so the steps in a box are equivalent
DIMER_LENGTH = 1.6      # Å; pre-formed dimers (relaxed by make_surf's minimization)


def _period(v) -> float:
    """Shortest diamond-lattice translation along integer direction v, in units of a."""
    v = np.array(v, float) / np.gcd.reduce(np.abs(np.array(v)))
    return np.linalg.norm(v) / 2 if int(round(v.sum())) % 2 == 0 else np.linalg.norm(v)


def geometry(o: str) -> Dict:
    """Rotation, cell lengths (Å), z unit, origin and the LAMMPS lattice arguments."""
    c = VICINAL[o]
    R = np.vstack([np.array(v, float) / np.linalg.norm(v) for v in (c["xdir"], c["ydir"], c["normal"])])
    ez = R[2]
    k = 6
    cells = np.array(list(itertools.product(range(-k, k + 1), repeat=3)), float)
    fcc = np.unique(np.round(((cells[:, None, :] + _BASIS[:4][None]).reshape(-1, 3)) @ ez, 9))
    d = float(np.min(np.diff(fcc)))                         # FCC plane spacing along the normal (a)
    dia = ((cells[:, None, :] + _BASIS[None]).reshape(-1, 3)) @ ez
    ph = np.unique(np.round((dia % d) / d, 6))
    gaps = np.diff(np.r_[ph, ph[0] + 1])
    j = int(np.argmax(gaps))
    centre = (ph[j] + gaps[j] / 2) % 1
    q = c["q"]
    s = 2 * q * d                                           # z lattice unit (a)
    bphase = (-(5 / 4) * q) % 1                             # phase of the -5/8 s grid line
    shift = ((bphase - centre) % 1) * d                     # move the lattice up along the normal (a):
    # the gap centre (phase centre) then lands on the grid lines (phase bphase)
    origin = shift * ez                                     # the same shift in crystal coordinates (a)
    # LAMMPS adds `origin` after the rotation, in units of the spacings (x a1 spacing,
    # y a2 spacing, z a3 spacing of the rotated box), so the shift along the normal is
    # origin 0 0 shift/s
    lmp_origin_z = shift / s
    assert 0 <= lmp_origin_z < 1
    Lx, Ly = _period(c["xdir"]), _period(c["ydir"])
    ter = np.array(c["terrace"], float) / np.linalg.norm(c["terrace"])
    miscut = float(np.degrees(np.arccos(ez @ ter)))
    dimers = c["terrace"] == (0, 0, 1)
    spacing = 0.25 if dimers else 1 / np.sqrt(3)            # terrace layer spacing (a)
    density = 2.0 if dimers else 4 / np.sqrt(3)             # atoms per a^2 of one terrace plane
    args = (f"diamond ${{lat_a}} orient z {' '.join(map(str, c['normal']))} "
            f"orient x {' '.join(map(str, c['xdir']))} orient y {' '.join(map(str, c['ydir']))} "
            f"spacing {Lx:.10f} {Ly:.10f} {s:.10f} origin 0 0 {lmp_origin_z:.10f}")
    return dict(R=R, Lx=Lx * LAT_A, Ly=Ly * LAT_A, s=s * LAT_A, s_a=s, origin=origin,
                terrace=ter, miscut=miscut, layer=spacing * LAT_A, dimers=dimers,
                ml_per_cell=density * Lx * Ly, margin=gaps.max() * d * LAT_A / 2,
                per_period=int(round(Lx * np.sin(np.radians(miscut)) / spacing)),
                lattice_args=args)


def lattice_points(o: str, nx: int, ny: int, zlo: float, zhi: float) -> Tuple[np.ndarray, np.ndarray]:
    """(positions Å, crystal coordinates in a) of the lattice points LAMMPS creates in
    [0, nx Lx) x [0, ny Ly) x [zlo, zhi] Å for this orientation."""
    g = geometry(o)
    R = g["R"]
    span = max(nx * g["Lx"], ny * g["Ly"], zhi - zlo, abs(zlo), abs(zhi)) / LAT_A
    n = int(np.ceil(np.sqrt(3) * span)) + 3
    rng = range(-n, n + 1)
    cells = np.array(list(itertools.product(rng, rng, rng)), float)
    crys = (cells[:, None, :] + (_BASIS + g["origin"])[None]).reshape(-1, 3)
    r = crys @ R.T * LAT_A
    Lx, Ly = nx * g["Lx"], ny * g["Ly"]
    keep = ((r[:, 0] >= -1e-6) & (r[:, 0] < Lx - 1e-6) & (r[:, 1] >= -1e-6) & (r[:, 1] < Ly - 1e-6)
            & (r[:, 2] >= zlo) & (r[:, 2] <= zhi))
    return r[keep], crys[keep]


def _neighbours(x, box):
    d = x[:, None, :] - x[None, :, :]
    d[..., :2] -= np.round(d[..., :2] / box) * box
    rr = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(rr, np.inf)
    return (rr < 1.8).sum(axis=1), d, rr


def build_surface(o: str, surface: str, nx: int, ny: int, depth: int) -> Dict:
    """Atoms of the stepped slab: bottom at the addfix grid line z = -5/8 (lattice
    units), top a stepped surface near z = depth (lattice units).

    Returns dict(x (N,3) Å, box (3,2) Å, dimer (N,) 0/1, coordination (N,))."""
    if surface not in surfaces(o):
        raise ValueError(f"unknown surface {surface!r} for vicinal orientation {o}")
    g = geometry(o)
    R, s = g["R"], g["s"]
    Lx, Ly = nx * g["Lx"], ny * g["Ly"]
    box = np.array([Lx, Ly])
    zbot, zanchor = -5 / 8 * s, -1 / 8 * s                  # slab bottom and top of the anchor
    ztop = depth * s
    r, crys = lattice_points(o, nx, ny, zbot, ztop + 3 * s)
    h = crys @ g["terrace"] * LAT_A                          # height along the terrace normal
    dimers = double = surface in ("2x1_DA", "2x1_DB")

    if not double:
        # flat cut (bilayer steps): scan one step rise; keep the cuts leaving the fewest
        # over-cut atoms (more dangling bonds than a terrace atom), then (below) the one
        # with the fewest dangling bonds after stripping loose atoms
        terrace_db = 2 if g["dimers"] else 1
        scored = []
        for zt in np.linspace(ztop, ztop + 2.5, 101):
            band = (r[:, 2] <= zt) & (r[:, 2] > zt - 6.0)    # judge the top 4 Å; their neighbours lie within 6 Å
            c, _, _ = _neighbours(r[band], box)
            near = r[band, 2] > zt - 4.0
            scored.append((int(((4 - c[near]) > terrace_db).sum()), zt))
        fewest = min(b for b, _ in scored)
        candidates, seen = [], set()
        for b, zt in scored:
            sel = r[:, 2] <= zt
            key = sel.tobytes()
            if b == fewest and key not in seen:
                seen.add(key)
                candidates.append(sel)
    else:
        # layer staircase: layer n_top on [0, w), n_top - 2 on [w, 2w), ... (w = Lx/2 per
        # period); one period over the lattice is 4 layers lower, so the steps connect.
        # Both parities are tried; the dimer direction then picks D_A or D_B.
        nl = np.round(h / g["layer"]).astype(int)
        s_up = np.sign(g["terrace"] @ R[0])                  # -1: layers drop along +x
        w = g["Lx"] / 2
        k = np.floor((r[:, 0] - STEP_OFFSET) / w).astype(int)
        n_base = int(np.round(np.percentile(nl[r[:, 2] <= ztop + 0.5 * s], 99)))
        candidates = []
        for parity in (0, 1):
            ntop = n_base + parity + 2 * s_up * k
            candidates.append((nl <= ntop) | (r[:, 2] <= zanchor))

    built = []
    for sel in candidates:
        x, cr = r[sel].copy(), crys[sel]
        coord, d, rr = _neighbours(x, box)
        while True:                                          # strip loose atoms above the anchor
            loose = (coord <= 1) & (x[:, 2] > zanchor)
            if not loose.any():
                break
            x, cr = x[~loose], cr[~loose]
            coord, d, rr = _neighbours(x, box)
        dimer = np.zeros(len(x), int)
        if dimers:
            dimer, x = _dimerize(x, cr, coord, d, rr, box, R, g)
            coord, _, _ = _neighbours(x, box)
        if double:
            axis = _dimer_axis(x, dimer, box)
            if (axis == "x") != (surface == "2x1_DA"):
                continue                                     # the other parity is this type
        for kk, L in ((0, Lx), (1, Ly)):                    # fold -1e-6 .. 0 and L into [0, L)
            x[:, kk] %= L
            x[x[:, kk] >= L - 1e-9, kk] -= L
        top = x[:, 2] > zanchor
        dangling = int((4 - coord[top]).sum())
        built.append((dangling, len(built), dict(
            x=x, box=np.array([[0, Lx], [0, Ly], [zbot - 1e-6, x[:, 2].max() + 20.0]]),
            dimer=dimer, coordination=coord)))
    if not built:
        raise RuntimeError(f"could not build {o} {surface}")
    return min(built, key=lambda t: t[:2])[2]


def _dimerize(x, crys, coord, d, rr, box, R, g):
    """Pair (001) terrace atoms into 2x1 dimers along their dangling bonds."""
    nref = R @ np.array([0, 0, 1.0])
    bonds = R @ (_BONDS.T * LAT_A)
    top = x[:, 2] > x[:, 2].max() - 4.0
    surf = np.where((coord < 4) & top)[0]
    miss = {}
    for i in surf:
        f = (crys[i] - g["origin"]) * 2                      # sublattice A: half-integer cell coords
        sign = 1 if np.allclose(f, np.round(f), atol=1e-6) else -1
        m = []
        for b in (sign * bonds).T:
            t = x - (x[i] + b)
            t[:, :2] -= np.round(t[:, :2] / box) * box
            if np.min(np.linalg.norm(t, axis=1)) > 0.3:
                m.append(b - (b @ nref) * nref)              # in-plane part of a missing bond
        miss[i] = m
    adj = {i: [] for i in surf}
    for i, j in itertools.combinations(surf, 2):
        v = d[j, i]
        # same terrace layer (no rise along [001]; works across the periodic x boundary),
        # a/sqrt2 apart, each with a dangling bond pointing at the other
        if abs(rr[i, j] - LAT_A / np.sqrt(2)) < 0.05 and abs(v @ nref) < 0.3 \
                and any(b @ v > 0.3 for b in miss[i]) and any(b @ -v > 0.3 for b in miss[j]):
            adj[i].append(j)
            adj[j].append(i)
    dimer = np.zeros(len(x), int)
    used = set()
    for start in sorted(surf, key=lambda i: len(adj[i])):    # row ends first
        if start in used or not adj[start]:
            continue
        chain, prev, cur = [start], None, start
        while True:
            nxt = [k for k in adj[cur] if k != prev and k not in chain]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
        chain = [k for k in chain if k not in used]
        # odd row: leave unpaired the end with fewer dangling bonds (the atom at a step's
        # foot, already bonded into the riser), not a doubly dangling one
        if len(chain) % 2 and coord[chain[0]] > coord[chain[-1]]:
            chain = chain[1:]
        for i, j in zip(chain[0::2], chain[1::2]):
            v = d[j, i]
            shift = (np.linalg.norm(v) - DIMER_LENGTH) / 2 * v / np.linalg.norm(v)
            x[i] += shift
            x[j] -= shift
            dimer[i] = dimer[j] = 1
            used.update((i, j))
    for k, L in ((0, box[0]), (1, box[1])):                  # dimers may straddle the x/y boundary
        x[:, k] %= L
        x[x[:, k] >= L - 1e-9, k] -= L
    return dimer, x


def _dimer_axis(x, dimer, box) -> str:
    idx = np.where(dimer > 0)[0]
    d = x[idx][:, None, :] - x[idx][None, :, :]
    d[..., :2] -= np.round(d[..., :2] / box) * box
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    v = d[np.arange(len(idx)), np.argmin(r, axis=1)]
    return "x" if np.sum(np.abs(v[:, 0]) > np.abs(v[:, 1])) > len(idx) / 2 else "y"


def surfaces(o: str):
    return ("2x1_DA", "2x1_DB") if VICINAL[o]["terrace"] == (0, 0, 1) else ("1x1",)


def write_surface_data(spec, path) -> int:
    """Write the stepped slab for spec (orientation/surface/box) as a LAMMPS data file
    (atom_style charge, type 1 = C).  Returns the number of atoms."""
    s = build_surface(spec.orientation, spec.surface, spec.box_x, spec.box_y, spec.box_depth)
    x, box = s["x"], s["box"]
    g = geometry(spec.orientation)
    with open(path, "w") as fh:
        fh.write(f"C({spec.orientation}) {spec.surface} vicinal slab, {g['miscut']:.2f} deg miscut, "
                 f"{spec.box_x}x{spec.box_y}x{spec.box_depth}, unrelaxed (diamond_etch_md.vicinal)\n\n")
        fh.write(f"{len(x)} atoms\n3 atom types\n\n")
        for k, ax in enumerate("xyz"):
            fh.write(f"{box[k, 0]:.8f} {box[k, 1]:.8f} {ax}lo {ax}hi\n")
        fh.write("\nMasses\n\n1 12.011\n2 1.00784\n3 16.0\n\nAtoms # charge\n\n")
        for i, p in enumerate(x, start=1):
            fh.write(f"{i} 1 0.0 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    return len(x)
