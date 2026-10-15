"""Shared helpers used by multiple analysis modules."""


def _phase_of_impact(impact: int, spec, ml: int) -> int:
    """Return 0-based phase index for a given impact number in a cycling simulation."""
    total_cycle = sum(p.fluence_ml for p in spec.phases) * ml
    pos = (impact - 1) % total_cycle
    cum = 0
    for pi, p in enumerate(spec.phases):
        cum += p.fluence_ml * ml
        if pos < cum:
            return pi
    return len(spec.phases) - 1


DIAMOND_LAT_A = 3.5653871687699   # Å, default when lat_a.txt is unavailable

# Depth-profile KDE bandwidth as a fraction of the lattice constant.  (100)
# layers are a/4 apart, so a/8 already flattens the bulk (~1% ripple).  (111)
# stacks layers in pairs (a√3/12 apart) repeating every a/√3 = 2.06 Å, which
# a/8 leaves as ±56% spikes; a/4 brings that to ~±4%.  (110) layers are
# a√2/4 = 1.26 Å apart (a/8 → ~±17%, a/6 → ~±2%).  Hexagonal diamond: hex0001
# bilayers like (111) (a/4 → ±4%), hex1120 planes 1.26 Å apart (a/5 → ±2%),
# hex1010 plane pairs every 2.18 Å (a/4 → ±5%).
KDE_SIGMA_FRAC = {"100": 1 / 8, "110": 1 / 6, "111": 1 / 4, "113": 1 / 8,
                  "hex0001": 1 / 4, "hex1120": 1 / 5, "hex1010": 1 / 4,
                  # vicinal: atomic planes only ~0.3 Å apart along the normal
                  "119": 1 / 8, "557": 1 / 8, "775": 1 / 8}


def kde_sigma(lat_a, orientation=None):
    """KDE bandwidth (Å) for depth profiles of the given surface orientation."""
    return lat_a * KDE_SIGMA_FRAC.get(str(orientation), 1 / 8)


def _bottom_period(h, max_period=4.0, gap=0.15, tol=0.1):
    """(z0, period, cut) of the crystalline layers at the bottom of h, or None.

    Layers are clusters of heights separated by more than `gap` Å (layers
    under a quarter of the fullest one, i.e. strays, are ignored).  The period
    is the smallest shift mapping every layer in [z0, z0 + 2P) onto a layer
    with a matching atom count one period up (two periods, so a lone bottom
    layer can't match the first gap).  `cut` splits the gap
    just below the layer at z0 + P, so [z0 - gap, cut) holds one period.
    """
    import numpy as np
    h = np.sort(np.asarray(h, dtype=float))
    if len(h) < 4:
        return None
    h = h[h < h[0] + 3 * max_period]
    br = np.where(np.diff(h) > gap)[0]
    starts, ends = np.r_[0, br + 1], np.r_[br + 1, len(h)]
    pos = np.array([h[a:b].mean() for a, b in zip(starts, ends)])
    n = ends - starts
    keep = n >= 0.25 * n.max()
    pos, n = pos[keep], n[keep]
    for k in range(1, len(pos)):
        period = pos[k] - pos[0]
        if period > max_period:
            break
        two = np.where(pos < pos[0] + 2 * period - tol)[0]
        if all(np.any((np.abs(pos - pos[i] - period) < tol)
                      & (np.abs(n - n[i]) <= 0.2 * n[i])) for i in two):
            return pos[0], period, (pos[k] + pos[pos < pos[k] - tol].max()) / 2
    return None


def kde_density(h, area_A2, lat_a, top, sigma=None, dz=0.05, orientation=None,
                continue_lattice=True):
    """Gaussian-KDE depth profile of heights h (Å above the lattice bottom),
    normalized to bulk diamond (8 / a^3), on a dz grid from 0 to `top`.

    sigma defaults to kde_sigma(lat_a, orientation) (same as cna.py), which
    keeps the bulk nearly flat for each orientation.  With continue_lattice
    (for C), the bottom layers' repeat (_bottom_period) is copied below the
    slab so the density neither halves nor overshoots at the bottom, whatever
    the orientation or which layer of a (111) pair is lowest; if no clean
    repeat is found, heights are mirrored about h = -a/8 (right for (100)).
    Returns (grid, rho).
    """
    import numpy as np
    sigma = kde_sigma(lat_a, orientation) if sigma is None else sigma
    grid = np.arange(0.0, top + dz, dz)
    pad = int(np.ceil(4 * sigma / dz))
    edges = np.arange(-(pad + 0.5) * dz, top + (pad + 1) * dz, dz)
    h = np.asarray(h, dtype=float)
    below = np.empty(0)
    if continue_lattice and len(h):
        bp = _bottom_period(h)
        if bp is None:
            below = -lat_a / 4.0 - h
        else:
            z0, period, cut = bp
            seg = h[(h >= z0 - 0.15) & (h < cut)]
            below = np.concatenate([seg - k * period
                                    for k in range(1, int(np.ceil(4 * sigma / period)) + 1)])
    counts, _ = np.histogram(np.concatenate([h, below]), bins=edges)
    x = np.arange(-pad, pad + 1) * dz
    kern = np.exp(-0.5 * (x / sigma) ** 2)
    kern /= kern.sum()
    rho = np.convolve(counts / (dz * area_A2), kern, mode='same')[pad:pad + len(grid)]
    return grid, rho / (8.0 / lat_a ** 3)


KB_EV = 8.617333262e-5   # Boltzmann constant, eV/K


def read_radical_log(path):
    """Parse radical_log.txt: impact cn energy_eV polar_deg azimuth_deg [simulated].

    Returns a dict of numpy arrays (impact, cn, energy, polar, simulated); files
    written without the 6th column (no energy threshold) are all simulated.
    """
    import numpy as np
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                vals = [float(x) for x in parts[:6]]
            except ValueError:
                continue
            rows.append(vals if len(vals) == 6 else vals + [1.0])
    # A restart re-draws the radicals of the interrupted impact; keep the last
    # copy of each (impact, cn) so the log matches what was finally simulated.
    last = {}
    for k, r in enumerate(rows):
        last[(int(r[0]), int(r[1]))] = k
    a = np.array([rows[k] for k in sorted(last.values())], dtype=float).reshape(-1, 6)
    return {"impact": a[:, 0].astype(int), "cn": a[:, 1].astype(int), "energy": a[:, 2],
            "polar": a[:, 3], "simulated": a[:, 5] > 0.5}


def skipped_radicals(sim_dir):
    """Set of (impact, cn) radical draws skipped below the energy threshold.

    Empty for burst runs: there cn in radical_log.txt is the atom's index within
    the burst, not a per-radical window in ncarbon.txt / etch_products.txt.
    """
    import json
    from pathlib import Path
    path = Path(sim_dir) / "radical_log.txt"
    if not path.exists():
        return set()
    spec_path = Path(sim_dir) / "spec.json"
    if spec_path.exists():
        d = json.loads(spec_path.read_text())
        if d.get("radical_burst") or any(ph.get("radical_burst") for ph in d.get("phases") or []):
            return set()
    log = read_radical_log(path)
    m = ~log["simulated"]
    return set(zip(log["impact"][m].tolist(), log["cn"][m].tolist()))


def radical_sampling_params(spec):
    """Boltzmann radical settings of a sim: dict(T, dist, threshold, on, cosine),
    or None when radicals are not Boltzmann-sampled or cycling phases differ."""
    if spec is None:
        return None
    objs = ([p for p in spec.phases if p.flux_ratio > 0] if spec.phases
            else ([spec] if getattr(spec, "flux_ratio", 0) > 0 else []))
    params = {(o.radical_temperature, o.radical_energy_dist, o.radical_energy_threshold,
               o.radical_threshold_on, o.radical_angle_distribution)
              for o in objs if o.radical_temperature is not None}
    if len(params) != 1:
        return None
    T, dist, thr, on, cos = params.pop()
    return {"T": T, "dist": dist, "threshold": thr, "on": on, "cosine": cos}


def expected_skip_fraction(params, n=400000):
    """Monte Carlo fraction of radical draws below the threshold, using the same
    sampling formulas as the LAMMPS scripts (None if no threshold)."""
    import numpy as np
    if not params or params["threshold"] is None:
        return None
    rng = np.random.default_rng(12345)
    kT = KB_EV * params["T"]
    if params["dist"] == "flux":
        E = -kT * np.log(rng.random(n) * rng.random(n))
    else:
        E = kT * (rng.standard_normal((n, 3)) ** 2).sum(axis=1) / 2.0
    if params["on"] == "normal":
        cos2 = rng.random(n) if params["cosine"] else 1.0   # Lambert: cos^2(theta) ~ U(0,1)
        E = E * cos2
    return float(np.mean(E < params["threshold"]))
