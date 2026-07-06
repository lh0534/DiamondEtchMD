#!/usr/bin/env python3
"""Analyze Ar ion penetration depth from per-impact z-trajectory files.

Each file in penetration/impact_N.txt has two columns:
    time(fs)  z_Ar(Å)

The ion is deposited i_above Å above the highest atom in the structure, so:
    surface_z        = z_initial - i_above   (where z_initial = first z in file)
    penetration_depth = surface_z - z_min

Positive depth = ion penetrated below the surface. Zero = reflected without penetrating.

Usage:
    python analyze_penetration.py [sim_dir]
"""
import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _read_config_var(sim_dir: Path, varname: str) -> float:
    cfg = (sim_dir / "config.lmp").read_text()
    m = re.search(rf"variable\s+{varname}\s+equal\s+([\d.]+)", cfg)
    if not m:
        raise ValueError(f"{varname} not found in config.lmp")
    return float(m.group(1))


def analyze_penetration(sim_dir: Path) -> dict:
    pen_dir = sim_dir / "penetration"
    if not pen_dir.exists():
        print("No penetration/ directory found.")
        return {}

    files = sorted(pen_dir.glob("impact_*.txt"),
                   key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)))
    if not files:
        print("No impact files in penetration/.")
        return {}

    i_above = _read_config_var(sim_dir, "i_above")

    impacts, depths = [], []
    for f in files:
        idx = int(re.search(r"(\d+)", f.stem).group(1))
        try:
            data = np.loadtxt(f)
        except Exception:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.size == 0:
            continue

        z = data[:, 1]
        z_initial   = z[0]
        surface_z   = z_initial - i_above
        depth       = max(surface_z - z.min(), 0.0)

        impacts.append(idx)
        depths.append(depth)

    if not impacts:
        print("No readable penetration files.")
        return {}

    impacts = np.array(impacts)
    depths  = np.array(depths)
    n_total     = len(impacts)
    n_embedded  = int((depths > 0).sum())
    embed_rate  = 100.0 * n_embedded / n_total

    print(f"Penetration analysis: {n_total} impacts  (i_above={i_above} Å)")
    print(f"  Penetrated (depth > 0): {n_embedded} / {n_total} = {embed_rate:.1f}%")
    if n_embedded:
        print(f"  Mean depth (penetrated): {depths[depths > 0].mean():.2f} Å")
    print(f"  Max depth:               {depths.max():.2f} Å")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(impacts, depths, s=8, alpha=0.6, color="steelblue")
    ax.set_xlabel("Impact #")
    ax.set_ylabel("Penetration depth (Å)")
    ax.set_title("Ar penetration depth per impact")

    ax = axes[1]
    pos = depths[depths > 0]
    if len(pos):
        ax.hist(pos, bins=20, color="steelblue", edgecolor="white")
        ax.set_xlabel("Penetration depth (Å)")
        ax.set_ylabel("Count")
        ax.set_title(f"Depth distribution  ({embed_rate:.0f}% penetrated)")
    else:
        ax.text(0.5, 0.5, "No penetration", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    out = sim_dir / "penetration_depth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    return {
        "n_impacts": n_total,
        "n_penetrated": n_embedded,
        "penetration_rate": embed_rate,
        "mean_depth": float(depths[depths > 0].mean()) if n_embedded else 0.0,
        "max_depth": float(depths.max()),
    }


if __name__ == "__main__":
    sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    analyze_penetration(sim_dir)
