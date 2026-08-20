"""
analysis/sticking.py — radical reflection rate checker for RIE simulations.

A radical "reflects" when its etch_products.txt entry has nC=0, nH=0, nO=1, nAr=0
(the O• atom bounced off without sticking).  Reflection rates are computed per ML
bin (impacts_per_ml = spec.ml) and reported via a RADS_REFLECTING warning file.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def check_radical_reflecting(
    sim_dir,
    ml: int = 0,
    threshold: float = 0.95,
) -> dict:
    """Check per-ML radical reflection rate and write RADS_REFLECTING if needed.

    Parameters
    ----------
    sim_dir : path-like
        Simulation directory containing ncarbon.txt and etch_products.txt.
    ml : int
        Atoms per monolayer.  0 = load from spec.json in sim_dir.
    threshold : float
        Reflection fraction that triggers the warning file (default 0.95 = 95%).

    Returns
    -------
    dict with keys:
        ml_stats   : list of per-bin dicts (bin_idx, n_total, n_reflect, reflect_frac)
        bad_bins   : subset where reflect_frac >= threshold
        wrote_warning : bool
    """
    from .ncarbon import parse_ncarbon
    from .etch_products import parse_etch_products

    sim_dir = Path(sim_dir)
    warning_file = sim_dir / "RADS_REFLECTING"

    # Resolve ml
    if ml <= 0:
        spec_path = sim_dir / "spec.json"
        if spec_path.exists():
            with open(spec_path) as f:
                spec_data = json.load(f)
            ml = spec_data.get("ml", 0)
    if ml <= 0:
        raise ValueError(
            f"ml must be > 0; got {ml}. Pass ml= explicitly or ensure spec.json exists."
        )

    nc_path = sim_dir / "ncarbon.txt"
    ep_path = sim_dir / "etch_products.txt"

    if not nc_path.exists():
        return {"ml_stats": [], "bad_bins": [], "wrote_warning": False}

    nc_records = parse_ncarbon(nc_path)
    ep_records = parse_etch_products(ep_path) if ep_path.exists() else []

    # Radical rows: cn > 0
    rad_rows = [r for r in nc_records if r.get("cn", 0) > 0]

    # Reflect set: (impact, cn) pairs where the product was a lone O atom
    reflect_set = {
        (r["impact"], r["cn"])
        for r in ep_records
        if r.get("cn", 0) > 0
        and r["n_C"] == 0
        and r["n_H"] == 0
        and r["n_O"] == 1
        and r["n_Ar"] == 0
    }

    if not rad_rows:
        return {"ml_stats": [], "bad_bins": [], "wrote_warning": False}

    # Group into ML bins: bin_idx = (impact - 1) // ml
    max_impact = max(r["impact"] for r in rad_rows)
    n_complete_bins = max_impact // ml  # only bins where all impacts have been seen

    bins: Dict[int, List] = {}
    for r in rad_rows:
        b = (r["impact"] - 1) // ml
        bins.setdefault(b, []).append((r["impact"], r["cn"]))

    ml_stats = []
    for b in sorted(bins):
        if b >= n_complete_bins:
            continue  # skip the partial final bin
        events = bins[b]
        n_total = len(events)
        n_reflect = sum(1 for key in events if key in reflect_set)
        reflect_frac = n_reflect / n_total if n_total > 0 else None
        ml_stats.append({
            "bin_idx": b,
            "ml_number": b + 1,
            "n_total": n_total,
            "n_reflect": n_reflect,
            "reflect_frac": reflect_frac,
        })

    bad_bins = [s for s in ml_stats if s["reflect_frac"] is not None and s["reflect_frac"] >= threshold]

    wrote_warning = False
    if bad_bins:
        lines = [
            f"RIE radical reflection warning: some monolayers have ≥{threshold*100:.0f}% reflection.",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        for s in bad_bins:
            pct = s["reflect_frac"] * 100
            lines.append(
                f"ML {s['ml_number']}: {pct:.1f}% reflecting "
                f"({s['n_reflect']}/{s['n_total']} radicals reflected)"
            )
        warning_file.write_text("\n".join(lines) + "\n")
        wrote_warning = True
    elif warning_file.exists():
        warning_file.unlink()

    return {"ml_stats": ml_stats, "bad_bins": bad_bins, "wrote_warning": wrote_warning}
