"""
analysis/cna.py — Post-hoc sp3 / amorphous carbon analysis from impact_snaps/.

Reads LAMMPS data files written by `write_data` after each impact and computes:
  - sp3 carbon count (exactly 4 C neighbours within 1.85 Å, PBC in x/y)
  - 1-D z-density profile for carbon
  - amorphous layer thickness via the 10%–90% density criterion
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
            in_atoms = False
            continue

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


def sp3_mask(data: Dict, c_type: int = 1, cutoff: float = 1.85) -> np.ndarray:
    """Return a bool mask (length = N_C) that is True for sp3 carbon atoms.

    An sp3 carbon has exactly 4 C neighbours within `cutoff` Å.
    Minimum-image PBC is applied in x and y only (boundary = p p m).
    """
    mask = data['types'] == c_type
    pos = np.column_stack([data['x'][mask], data['y'][mask], data['z'][mask]])
    if len(pos) == 0:
        return np.array([], dtype=bool)

    box = data['box']
    lx = box[0, 1] - box[0, 0]
    ly = box[1, 1] - box[1, 0]

    # Pairwise displacement vectors (N×N×3), PBC in x and y
    delta = pos[None, :, :] - pos[:, None, :]
    delta[:, :, 0] -= lx * np.round(delta[:, :, 0] / lx)
    delta[:, :, 1] -= ly * np.round(delta[:, :, 1] / ly)

    r2 = (delta ** 2).sum(axis=2)
    np.fill_diagonal(r2, np.inf)

    return (r2 < cutoff ** 2).sum(axis=1) == 4


def zdensity_profile(
    data: Dict,
    c_type: int = 1,
    bin_width: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """1-D z-density profile for carbon, in atoms Å⁻³.

    Returns
    -------
    z_centers : bin centres in Å
    density   : C atoms per Å³ per bin
    """
    mask = data['types'] == c_type
    z_c = data['z'][mask]
    box = data['box']
    zlo, zhi = float(box[2, 0]), float(box[2, 1])
    area = (box[0, 1] - box[0, 0]) * (box[1, 1] - box[1, 0])

    edges = np.arange(zlo, zhi + bin_width, bin_width)
    counts, _ = np.histogram(z_c, bins=edges)
    z_centers = (edges[:-1] + edges[1:]) / 2
    density = counts / (area * bin_width)
    return z_centers, density


def _bulk_density_ref(density: np.ndarray, bottom_frac: float = 0.3) -> float:
    """Estimate bulk diamond density from the deepest `bottom_frac` of z-bins."""
    n = max(1, int(len(density) * bottom_frac))
    return float(np.mean(density[:n]))


def amorphous_thickness_angstrom(
    z_centers: np.ndarray,
    density: np.ndarray,
    threshold_lo: float = 0.10,
    threshold_hi: float = 0.90,
) -> float:
    """Return the amorphous layer thickness in Å.

    Scans from the vacuum side (high z) downward.  The outer edge is where
    density first reaches `threshold_lo` × bulk; the inner edge is where
    density first reaches `threshold_hi` × bulk.  Returns the distance
    between those two points (0 if the transition cannot be resolved).
    """
    bulk = _bulk_density_ref(density)
    if bulk == 0:
        return 0.0

    d_norm = density / bulk
    z_lo = z_hi = None

    for j in range(len(z_centers) - 1, -1, -1):
        if z_lo is None and d_norm[j] >= threshold_lo:
            z_lo = z_centers[j]
        if z_hi is None and d_norm[j] >= threshold_hi:
            z_hi = z_centers[j]
            break

    if z_lo is None or z_hi is None:
        return 0.0
    return max(0.0, float(z_lo - z_hi))


def analyze_impact(path, c_type: int = 1, cutoff: float = 1.85, bin_width: float = 0.5) -> Dict:
    """Compute CNA metrics for one impact_snaps/*.data file.

    Returns
    -------
    dict: n_sp3, n_amorphous, sp3_fraction, amorphous_thickness_A, bulk_density
    """
    data = read_lammps_data(path)
    n_C = int((data['types'] == c_type).sum())

    sp3 = sp3_mask(data, c_type=c_type, cutoff=cutoff)
    n_sp3 = int(sp3.sum())

    z_c, dens = zdensity_profile(data, c_type=c_type, bin_width=bin_width)
    thickness = amorphous_thickness_angstrom(z_c, dens)
    bulk_d = _bulk_density_ref(dens)

    return {
        'n_sp3':                n_sp3,
        'n_amorphous':          n_C - n_sp3,
        'sp3_fraction':         n_sp3 / n_C if n_C > 0 else 0.0,
        'amorphous_thickness_A': thickness,
        'bulk_density':         bulk_d,
    }


def load_cna_series(
    data_dir,
    stride: int = 1,
    c_type: int = 1,
    cutoff: float = 1.85,
    bin_width: float = 0.5,
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
        impact, n_sp3, n_amorphous, sp3_fraction,
        amorphous_thickness_A, bulk_density
    """
    data_dir = Path(data_dir)
    files = sorted(
        (f for f in data_dir.glob('*.data') if f.stem.isdigit()),
        key=lambda f: int(f.stem),
    )
    files = files[::stride]

    records = []
    for i, f in enumerate(files):
        if verbose and i % 100 == 0:
            print(f"  CNA: {i}/{len(files)}  ({f.name})", flush=True)
        m = analyze_impact(f, c_type=c_type, cutoff=cutoff, bin_width=bin_width)
        m['impact'] = int(f.stem)
        records.append(m)
    return records
