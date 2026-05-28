"""
analysis/cna.py — Post-hoc diamond structure analysis from impact_snaps/.

Reads LAMMPS data files written by `write_data` after each impact and computes:
  - adaptive CNA classification (same algorithm as identify_diamond.py)
  - 1-D z-density profile for carbon
  - amorphous layer thickness via the 10%–90% density criterion

CNA categories (per atom):
  0 = OTHER          — amorphous / sp2 carbon
  1 = DIAMOND        — cubic diamond bulk
  2 = DIAMOND_NEIGH1 — 1st neighbor of a diamond atom (surface/interface)
  3 = DIAMOND_NEIGH2 — 2nd neighbor of a diamond atom

The CNA runs on ALL atom types so that H and O neighbours are counted
correctly when classifying surface carbon atoms.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

try:
    from scipy.ndimage import gaussian_filter1d as _gauss
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── z-density constants (matching cycle_plots.ipynb) ─────────────────────────
_LATTICE_A  = 3.5653871687699               # diamond lattice parameter, Å
_ML_THICK_A = _LATTICE_A / 4               # (100) interlayer spacing, Å/ML
_SIGMA_KDE  = _LATTICE_A / 8               # Gaussian KDE bandwidth, Å
_RHO_REF    = 3.52 * 6.02214076e23 / 12.011 / 1e24  # diamond density, atoms/Å³
_FRAC_LO    = 0.10                          # surface threshold
_FRAC_HI    = 0.90                          # bulk-top threshold
_DZ         = 0.05                          # z-grid spacing, Å
_Z_MAX      = 120.0                         # grid upper bound, Å
_Z_EDGES    = np.arange(-_DZ / 2, _Z_MAX + _DZ, _DZ)
_Z_CENTERS  = _Z_EDGES[:-1] + _DZ / 2
_SIGMA_BINS = _SIGMA_KDE / _DZ             # KDE σ in grid bins

OTHER, DIAMOND, NEIGH1, NEIGH2 = 0, 1, 2, 3


def read_lammps_data(path) -> Dict:
    """Parse a LAMMPS charge-style data file written by write_data.

    Returns
    -------
    dict with keys:
        box   : (3, 2) float array  [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
        types : (N,) int array
        x, y, z : (N,) float arrays
    """
    lines = Path(path).read_text().splitlines()

    n_atoms = 0
    box = np.zeros((3, 2))
    in_atoms = False
    type_list, x_list, y_list, z_list = [], [], [], []

    for raw in lines:
        l = raw.strip()
        if not l:
            continue  # blank lines are separators; do not exit atom-reading mode

        if re.match(r'\d+\s+atoms\b', l):
            n_atoms = int(l.split()[0])
        elif m := re.match(r'([-\d.eE+]+)\s+([-\d.eE+]+)\s+xlo xhi', l):
            box[0] = float(m.group(1)), float(m.group(2))
        elif m := re.match(r'([-\d.eE+]+)\s+([-\d.eE+]+)\s+ylo yhi', l):
            box[1] = float(m.group(1)), float(m.group(2))
        elif m := re.match(r'([-\d.eE+]+)\s+([-\d.eE+]+)\s+zlo zhi', l):
            box[2] = float(m.group(1)), float(m.group(2))
        elif l.startswith('Atoms'):
            in_atoms = True
        elif l.split()[0] in ('Velocities', 'Bonds', 'Angles', 'Masses', 'Pair'):
            in_atoms = False
        elif in_atoms:
            parts = l.split()
            if len(parts) >= 6:
                # id type charge x y z [ix iy iz ...]
                type_list.append(int(parts[1]))
                x_list.append(float(parts[3]))
                y_list.append(float(parts[4]))
                z_list.append(float(parts[5]))

    return {
        'box':   box,
        'types': np.array(type_list, dtype=np.int32),
        'x':     np.array(x_list),
        'y':     np.array(y_list),
        'z':     np.array(z_list),
    }


# ── JAX kernels (identical to identify_diamond.py) ────────────────────────────

if _HAS_JAX:
    @jax.jit
    def _jax_find_nn4(pos, Lx, Ly):
        N = pos.shape[0]
        diff = pos[:, None, :] - pos[None, :, :]
        dx = diff[:, :, 0] - Lx * jnp.round(diff[:, :, 0] / Lx)
        dy = diff[:, :, 1] - Ly * jnp.round(diff[:, :, 1] / Ly)
        dz = diff[:, :, 2]
        dsq = dx*dx + dy*dy + dz*dz
        dsq = dsq.at[jnp.arange(N), jnp.arange(N)].set(jnp.inf)
        return jnp.argsort(dsq, axis=1, stable=True)[:, :4].astype(jnp.int32)

    @jax.jit
    def _jax_find_nn12(nn4):
        N = nn4.shape[0]
        k_flat = nn4[nn4].reshape(N, 16)
        is_self = (k_flat == jnp.arange(N)[:, None])
        n_self = jnp.sum(is_self, axis=1)
        order = jnp.argsort(is_self.astype(jnp.int32), axis=1, stable=True)
        nn12 = jnp.take_along_axis(k_flat, order, axis=1)[:, :12].astype(jnp.int32)
        return nn12, (n_self == 4)

    @jax.jit
    def _jax_cna_classify(pos, nn12, valid, Lx, Ly):
        pos_nn12 = pos[nn12]
        d = pos_nn12 - pos[:, None, :]
        d = d.at[:, :, 0].set(d[:, :, 0] - Lx * jnp.round(d[:, :, 0] / Lx))
        d = d.at[:, :, 1].set(d[:, :, 1] - Ly * jnp.round(d[:, :, 1] / Ly))
        r_accum = jnp.sum(jnp.sqrt(jnp.sum(d**2, axis=-1)), axis=1)
        cutoff_sq = (r_accum * 0.10059223) ** 2

        diff12 = pos_nn12[:, :, None, :] - pos_nn12[:, None, :, :]
        dx12 = diff12[:, :, :, 0] - Lx * jnp.round(diff12[:, :, :, 0] / Lx)
        dy12 = diff12[:, :, :, 1] - Ly * jnp.round(diff12[:, :, :, 1] / Ly)
        dz12 = diff12[:, :, :, 2]
        dsq12 = dx12**2 + dy12**2 + dz12**2

        eye12 = jnp.eye(12, dtype=bool)[None, :, :]
        bond = (dsq12 < cutoff_sq[:, None, None]) & ~eye12
        bi = bond.astype(jnp.int32)

        cna_4 = jnp.all(jnp.sum(bi, axis=2) == 4, axis=1)
        deg = jnp.matmul(bi, bi)
        num_bonds = jnp.sum(deg * bi, axis=-1) // 2
        cna_2 = jnp.all(num_bonds == 2, axis=1)
        cna_1 = jnp.all(jnp.where(bond, deg, 1) == 1, axis=(1, 2))

        is_diamond = valid & cna_4 & cna_2 & cna_1
        return jnp.where(is_diamond, DIAMOND, OTHER)


def _np_find_nn4(pos, Lx, Ly):
    N = len(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    diff[:, :, 0] -= Lx * np.round(diff[:, :, 0] / Lx)
    diff[:, :, 1] -= Ly * np.round(diff[:, :, 1] / Ly)
    dsq = (diff ** 2).sum(axis=2)
    np.fill_diagonal(dsq, np.inf)
    return np.argsort(dsq, axis=1)[:, :4].astype(np.int32)


def _np_find_nn12(nn4):
    N = len(nn4)
    k_flat = nn4[nn4].reshape(N, 16)
    is_self = (k_flat == np.arange(N)[:, None])
    n_self = is_self.sum(axis=1)
    order = np.argsort(is_self.astype(np.int32), axis=1, kind='stable')
    nn12 = k_flat[np.arange(N)[:, None], order][:, :12].astype(np.int32)
    return nn12, (n_self == 4)


def _np_cna_classify(pos, nn12, valid, Lx, Ly):
    pos_nn12 = pos[nn12]
    d = pos_nn12 - pos[:, None, :]
    d[:, :, 0] -= Lx * np.round(d[:, :, 0] / Lx)
    d[:, :, 1] -= Ly * np.round(d[:, :, 1] / Ly)
    r_accum = np.sqrt((d ** 2).sum(axis=2)).sum(axis=1)
    cutoff_sq = (r_accum * 0.10059223) ** 2

    diff12 = pos_nn12[:, :, None, :] - pos_nn12[:, None, :, :]
    diff12[:, :, :, 0] -= Lx * np.round(diff12[:, :, :, 0] / Lx)
    diff12[:, :, :, 1] -= Ly * np.round(diff12[:, :, :, 1] / Ly)
    dsq12 = (diff12 ** 2).sum(axis=3)

    eye12 = np.eye(12, dtype=bool)
    bond = (dsq12 < cutoff_sq[:, None, None]) & ~eye12
    bi = bond.astype(np.int32)

    cna_4 = (bi.sum(axis=2) == 4).all(axis=1)
    deg = np.matmul(bi, bi)
    num_bonds = (deg * bi).sum(axis=2) // 2
    cna_2 = (num_bonds == 2).all(axis=1)
    cna_1 = (np.where(bond, deg, 1) == 1).all(axis=(1, 2))

    return np.where(valid & cna_4 & cna_2 & cna_1, DIAMOND, OTHER).astype(np.int32)


def _label_neighbors(pat, nn4, nn12):
    """Promote OTHER → NEIGH2/NEIGH1 for atoms adjacent to diamond atoms."""
    for i in np.where(pat == DIAMOND)[0]:
        for k in nn12[i]:
            if pat[k] == OTHER:
                pat[k] = NEIGH2
        for k in nn4[i]:
            if pat[k] in (OTHER, NEIGH2):
                pat[k] = NEIGH1
    return pat


def diamond_cna(data: Dict, c_type: int = 1) -> np.ndarray:
    """Adaptive CNA classification; returns int32 labels for carbon atoms only.

    Labels: OTHER=0, DIAMOND=1, NEIGH1=2, NEIGH2=3.
    Runs on ALL atom types (H, O, Ar included) for correct neighbour search.
    Uses JAX JIT kernels if available, otherwise pure numpy.
    PBC applied in x and y only (slab geometry).
    """
    pos = np.column_stack([data['x'], data['y'], data['z']])
    N = len(pos)
    if N == 0:
        return np.array([], dtype=np.int32)

    box = data['box']
    Lx = float(box[0, 1] - box[0, 0])
    Ly = float(box[1, 1] - box[1, 0])

    if _HAS_JAX:
        pos_j = jnp.array(pos)
        nn4  = _jax_find_nn4(pos_j, Lx, Ly)
        nn12, valid = _jax_find_nn12(nn4)
        pat  = _jax_cna_classify(pos_j, nn12, valid, Lx, Ly)
        pat  = np.array(pat,  dtype=np.int32)
        nn4  = np.array(nn4,  dtype=np.int32)
        nn12 = np.array(nn12, dtype=np.int32)
    else:
        nn4  = _np_find_nn4(pos, Lx, Ly)
        nn12, valid = _np_find_nn12(nn4)
        pat  = _np_cna_classify(pos, nn12, valid, Lx, Ly)

    pat = _label_neighbors(pat, nn4, nn12)
    return pat[data['types'] == c_type]


def zdensity_profile(
    data: Dict,
    c_type: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-D Gaussian KDE density profile for carbon atoms (matching cycle_plots.ipynb).

    Returns
    -------
    z_centers : fixed grid centres in Å (relative to zlo)
    density   : Gaussian-smoothed C atoms per Å³ per bin
    z_c_rel   : C atom z-positions relative to zlo
    """
    mask = data['types'] == c_type
    z_c = data['z'][mask]
    box = data['box']
    zlo = float(box[2, 0])
    area = (box[0, 1] - box[0, 0]) * (box[1, 1] - box[1, 0])

    z_c_rel = z_c - zlo
    hist, _ = np.histogram(z_c_rel, bins=_Z_EDGES)
    density = hist / (area * _DZ)

    if _HAS_SCIPY:
        density = _gauss(density.astype(float), sigma=_SIGMA_BINS)

    return _Z_CENTERS.copy(), density, z_c_rel


def amorphous_metrics(
    density: np.ndarray,
    z_c_rel: np.ndarray,
) -> Tuple[float, int]:
    """Return (thickness_A, n_in_zone) using a fixed bulk reference density.

    z_upper = highest z where density > FRAC_LO × RHO_REF  (surface top)
    z_lower = highest z where density > FRAC_HI × RHO_REF  (top of bulk crystal)
    thickness = z_upper − z_lower
    n_in_zone = C atoms with z_c_rel in [z_lower, z_upper]
    """
    idx_lo = np.where(density > _FRAC_LO * _RHO_REF)[0]
    idx_hi = np.where(density > _FRAC_HI * _RHO_REF)[0]
    if not len(idx_lo) or not len(idx_hi):
        return 0.0, 0
    z_upper = float(_Z_CENTERS[idx_lo[-1]])
    z_lower = float(_Z_CENTERS[idx_hi[-1]])
    if z_lower >= z_upper:
        return 0.0, 0
    thickness = z_upper - z_lower
    n_in_zone = int(np.sum((z_c_rel >= z_lower) & (z_c_rel <= z_upper)))
    return thickness, n_in_zone


def analyze_impact(path, c_type: int = 1) -> Dict:
    """Compute CNA metrics for one impact_snaps/*.data file.

    Returns
    -------
    dict with keys:
        n_diamond            — carbon atoms classified as cubic diamond (label 1)
        n_crystalline        — diamond + 1st/2nd neighbours (labels 1-3)
        n_amorphous          — OTHER carbon atoms (label 0)
        n_amorphous_zone     — C atoms in the amorphous zone (z-density method)
        diamond_fraction     — n_diamond / n_C
        amorphous_thickness_A — amorphous layer thickness in Å (z-density profile)
    """
    data = read_lammps_data(path)
    n_C = int((data['types'] == c_type).sum())

    labels = diamond_cna(data, c_type=c_type)
    n_diamond     = int((labels == DIAMOND).sum())
    n_crystalline = int((labels >= DIAMOND).sum())
    n_amorphous   = int((labels == OTHER).sum())

    _, density, z_c_rel = zdensity_profile(data, c_type=c_type)
    thickness, n_amorphous_zone = amorphous_metrics(density, z_c_rel)

    return {
        'n_diamond':             n_diamond,
        'n_crystalline':         n_crystalline,
        'n_amorphous':           n_amorphous,
        'n_amorphous_zone':      n_amorphous_zone,
        'diamond_fraction':      n_diamond / n_C if n_C > 0 else 0.0,
        'amorphous_thickness_A': thickness,
    }


def load_cna_series(
    data_dir,
    stride: int = 1,
    c_type: int = 1,
    verbose: bool = False,
) -> List[Dict]:
    """Compute CNA metrics for all (or strided) impact_snaps/ entries.

    Parameters
    ----------
    data_dir : path to impact_snaps/ directory
    stride   : analyze every stride-th impact (1 = every impact)
    verbose  : print progress to stdout

    Returns
    -------
    list of dict, sorted by impact number.  Each dict has keys:
        impact, n_diamond, n_crystalline, n_amorphous, n_amorphous_zone,
        diamond_fraction, amorphous_thickness_A
    """
    # Match post-ion snapshots: {n}_0.data (new cycling/RIE) or {n}.data (old format).
    # Radical-sweep snapshots ({n}_{k>0}.data) are excluded.
    # When both {n}.data and {n}_0.data exist, prefer {n}_0.data (post-impact state).
    _snap_re = re.compile(r'^(\d+)(?:_0)?$')

    data_dir = Path(data_dir)
    def _impact_num(f):
        m = _snap_re.match(f.stem)
        return int(m.group(1)) if m else None

    def _is_suffixed(f):
        return f.stem.endswith('_0')

    # Deduplicate: for each impact number, prefer {n}_0.data over {n}.data
    by_impact: Dict[int, Path] = {}
    for f in data_dir.glob('*.data'):
        n = _impact_num(f)
        if n is None:
            continue
        existing = by_impact.get(n)
        if existing is None or _is_suffixed(f):
            by_impact[n] = f

    files = [by_impact[n] for n in sorted(by_impact)]
    files = files[::stride]

    if verbose and _HAS_JAX:
        print("  (using JAX kernels)", flush=True)

    records = []
    for i, f in enumerate(files):
        if verbose and i % 100 == 0:
            print(f"  CNA: {i}/{len(files)}  ({f.name})", flush=True)
        m = analyze_impact(f, c_type=c_type)
        m['impact'] = _impact_num(f)
        records.append(m)
    return records
