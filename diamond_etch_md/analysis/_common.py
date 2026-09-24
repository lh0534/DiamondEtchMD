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


def kde_density(h, area_A2, lat_a, top, sigma=None, dz=0.05):
    """Gaussian-KDE depth profile of heights h (Å above the lattice bottom),
    normalized to bulk diamond (8 / a^3), on a dz grid from 0 to `top`.

    sigma defaults to a/8 (same as cna.py): (100) layers a/4 apart then give a
    nearly flat bulk (~1% ripple).  Heights are mirrored about h = -a/8 (half
    a layer spacing below the bottom layer) so the mirror image continues the
    lattice and the density neither halves nor double-counts at the bottom.
    Returns (grid, rho).
    """
    import numpy as np
    sigma = lat_a / 8.0 if sigma is None else sigma
    grid = np.arange(0.0, top + dz, dz)
    pad = int(np.ceil(4 * sigma / dz))
    edges = np.arange(-(pad + 0.5) * dz, top + (pad + 1) * dz, dz)
    h = np.asarray(h, dtype=float)
    counts, _ = np.histogram(np.concatenate([h, -lat_a / 4.0 - h]), bins=edges)
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
    a = np.array(rows, dtype=float).reshape(-1, 6)
    return {"impact": a[:, 0].astype(int), "cn": a[:, 1].astype(int), "energy": a[:, 2],
            "polar": a[:, 3], "simulated": a[:, 5] > 0.5}


def skipped_radicals(sim_dir):
    """Set of (impact, cn) radical draws skipped below the energy threshold."""
    from pathlib import Path
    path = Path(sim_dir) / "radical_log.txt"
    if not path.exists():
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
