"""
analysis/single_impact.py — post-processing for single-impact statistics runs.

Two sources of penetration depth:
  impact_stats.txt  : "final" depth = surf_z_before - ion_z_after (resting position)
  event_dump_N.dump : "max" depth   = surf_z_before - min(z_Ar)  (deepest point reached)

The trajectory-based max depth is the physically meaningful quantity for comparing
with SRIM/TRIM range predictions.  The final-position depth is useful for counting
what fraction of ions actually embedded.

CLI usage (from the simulation directory or with a path argument):
    python -m diamond_etch_md.analysis.single_impact [sim_dir]
    python -m diamond_etch_md.analysis.single_impact balls_Ar_50eV_single

Outputs:
    penetration_analysis.png   — 4-panel figure
    penetration_analysis.txt   — summary statistics
"""

import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


# ── LAMMPS dump streaming ─────────────────────────────────────────────────────

def _iter_dump_frames(path: Path) -> Iterator[dict]:
    """Yield one dict per frame from a LAMMPS custom dump file.

    dict keys: timestep (int), n_atoms (int), cols (list[str]), data (ndarray shape [n,ncols])
    Memory-efficient: yields one frame at a time.
    """
    with path.open() as fh:
        while True:
            # --- header ---
            line = fh.readline()
            if not line:
                return
            if "TIMESTEP" not in line:
                continue
            timestep = int(fh.readline().strip())

            fh.readline()  # ITEM: NUMBER OF ATOMS
            n_atoms = int(fh.readline().strip())

            fh.readline()  # ITEM: BOX BOUNDS
            fh.readline(); fh.readline(); fh.readline()  # 3 bound lines

            header = fh.readline()  # ITEM: ATOMS ...
            cols = header.split()[2:]  # drop "ITEM:" and "ATOMS"

            # --- atom data ---
            rows = []
            for _ in range(n_atoms):
                rows.append(fh.readline().split())
            if not rows:
                continue
            data = np.array(rows, dtype=float)
            yield {"timestep": timestep, "n_atoms": n_atoms, "cols": cols, "data": data}


def max_penetration_from_dump(dump_path: Path, surf_z_before: float) -> float | None:
    """Return surf_z_before - min(z_Ar) across all frames; None if no Ar found.

    Only reads type-4 atoms (Ar) from each frame — ignores all others.
    Positive value = ion penetrated below the surface at some point.
    """
    z_min_ar = None
    for frame in _iter_dump_frames(dump_path):
        data = frame["data"]
        cols = frame["cols"]
        try:
            type_col = cols.index("type")
            z_col    = cols.index("z")
        except ValueError:
            continue
        ar_mask = data[:, type_col] == 4
        if not ar_mask.any():
            continue
        z_ar = data[ar_mask, z_col].min()
        if z_min_ar is None or z_ar < z_min_ar:
            z_min_ar = z_ar
    if z_min_ar is None:
        return None
    return float(surf_z_before - z_min_ar)


# ── impact_stats.txt parsing ──────────────────────────────────────────────────

def parse_impact_stats(sim_dir: Path) -> dict[int, dict]:
    """Parse impact_stats.txt → {trial: {surf_z_before, ion_z_after, ion_in_box, pen_depth}}.

    Skips comment lines (starting with #).
    """
    path = sim_dir / "impact_stats.txt"
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        trial        = int(parts[0])
        surf_z_before = float(parts[1])
        ion_z_after  = float(parts[2])
        ion_in_box   = int(parts[3])
        pen_depth    = float(parts[4])
        records[trial] = {
            "surf_z_before": surf_z_before,
            "ion_z_after":   ion_z_after,
            "ion_in_box":    ion_in_box,
            "pen_depth":     pen_depth,
        }
    return records


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze_single_impact(sim_dir: Path) -> dict:
    """Analyze single-impact statistics run in sim_dir.

    Reads impact_stats.txt for all trials, then augments with trajectory-based
    max penetration depth for any trial that has a dump in etch_event_trajs/.

    Returns a summary dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = parse_impact_stats(sim_dir)
    if not stats:
        print(f"No impact_stats.txt found in {sim_dir}")
        return {}

    dump_dir = sim_dir / "etch_event_trajs"
    dump_files = {}
    if dump_dir.exists():
        for p in dump_dir.glob("event_dump_*.dump"):
            m = re.search(r"event_dump_(\d+)\.dump", p.name)
            if m:
                dump_files[int(m.group(1))] = p

    n_trials      = len(stats)
    n_with_dumps  = len(dump_files)
    print(f"Trials in impact_stats.txt : {n_trials}")
    print(f"Dump files in etch_event_trajs/: {n_with_dumps}")

    # ── Augment with trajectory max depth ─────────────────────────────────────
    max_depths = {}    # trial → max penetration from trajectory (may be > final depth)
    for trial, dump_path in sorted(dump_files.items()):
        rec = stats.get(trial)
        if rec is None:
            continue
        d = max_penetration_from_dump(dump_path, rec["surf_z_before"])
        max_depths[trial] = d
        if (len(max_depths) % 50) == 0:
            print(f"  Parsed {len(max_depths)}/{n_with_dumps} dumps...")

    # ── Outcome classification from impact_stats ───────────────────────────────
    # ion_in_box == 1 → embedded (Ar still in surface after thermalization)
    # ion_in_box == 0 + pen_depth > 0 → sputtered / left box after penetrating
    # pen_depth ≈ 0 (ion_z_after == 9999) → reflected above surface
    REFLECT_SENTINEL = 9000.0   # ion_z_after = 9999 when not in box + never penetrated
    embedded   = []
    reflected  = []
    sputtered  = []
    for trial, rec in sorted(stats.items()):
        if rec["ion_in_box"] == 1:
            embedded.append(trial)
        elif rec["ion_z_after"] > REFLECT_SENTINEL:
            reflected.append(trial)
        else:
            sputtered.append(trial)

    n_embedded  = len(embedded)
    n_reflected = len(reflected)
    n_sputtered = len(sputtered)

    print(f"\nIon outcome (from impact_stats.txt):")
    print(f"  Embedded  : {n_embedded:4d} / {n_trials}  ({100*n_embedded/n_trials:.1f}%)")
    print(f"  Reflected : {n_reflected:4d} / {n_trials}  ({100*n_reflected/n_trials:.1f}%)")
    print(f"  Sputtered : {n_sputtered:4d} / {n_trials}  ({100*n_sputtered/n_trials:.1f}%)")

    # ── Max depth stats (from dumps) ──────────────────────────────────────────
    valid_max = {t: d for t, d in max_depths.items() if d is not None}
    final_depths = np.array([stats[t]["pen_depth"] for t in stats])
    # cap sentinel values: trials where ion was never in box → depth = 0 or negative
    final_depths = np.maximum(final_depths, 0.0)

    if valid_max:
        max_depth_arr = np.array(list(valid_max.values()))
        max_depth_arr = np.maximum(max_depth_arr, 0.0)
        print(f"\nTrajectory-based max penetration depth ({len(valid_max)} trials with dumps):")
        print(f"  Mean : {max_depth_arr.mean():.2f} Å")
        print(f"  Std  : {max_depth_arr.std():.2f} Å")
        print(f"  Max  : {max_depth_arr.max():.2f} Å")
        pen_trials = max_depth_arr[max_depth_arr > 0.5]
        if len(pen_trials):
            print(f"  Mean (penetrating only, d>0.5Å): {pen_trials.mean():.2f} Å  "
                  f"({len(pen_trials)} trials)")
    else:
        print("No dump files found — only final-position depths available.")
        max_depth_arr = None

    print(f"\nFinal-position depth (all {n_trials} trials):")
    pos_final = final_depths[final_depths > 0.5]
    if len(pos_final):
        print(f"  Mean : {final_depths.mean():.2f} Å  (incl. zeros)")
        print(f"  Mean (penetrating only): {pos_final.mean():.2f} Å  ({len(pos_final)} trials)")
        print(f"  Max  : {final_depths.max():.2f} Å")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Single-impact statistics  —  {sim_dir.name}\n"
        f"({n_trials} trials, {n_embedded} embedded, {n_reflected} reflected, {n_sputtered} sputtered)",
        fontsize=11,
    )

    # Panel 1: outcome pie
    ax = axes[0, 0]
    labels  = ["Embedded", "Reflected", "Sputtered"]
    counts  = [n_embedded, n_reflected, n_sputtered]
    colors  = ["#4c72b0", "#dd8452", "#55a868"]
    non_zero = [(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0]
    if non_zero:
        ax.pie([c for _, c, _ in non_zero],
               labels=[f"{l} ({c})" for l, c, _ in non_zero],
               colors=[col for _, _, col in non_zero],
               autopct="%1.0f%%", startangle=90)
    ax.set_title("Ion outcome")

    # Panel 2: final penetration depth distribution (all trials)
    ax = axes[0, 1]
    depth_trimmed = final_depths[final_depths < np.percentile(final_depths, 99.5)]
    if depth_trimmed.max() > 0.1:
        ax.hist(depth_trimmed[depth_trimmed > 0], bins=30,
                color="#4c72b0", edgecolor="white", alpha=0.85,
                label=f"n={int((final_depths>0).sum())}")
        ax.set_xlabel("Final resting depth (Å)")
        ax.set_ylabel("Count")
        ax.set_title("Final-position penetration depth")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No penetration recorded", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Final-position penetration depth")

    # Panel 3: trajectory max depth (dump-based)
    ax = axes[1, 0]
    if max_depth_arr is not None and max_depth_arr.max() > 0.1:
        d_trim = max_depth_arr[max_depth_arr < np.percentile(max_depth_arr, 99.5)]
        ax.hist(d_trim[d_trim > 0], bins=30,
                color="#55a868", edgecolor="white", alpha=0.85,
                label=f"n={int((max_depth_arr>0).sum())}")
        ax.set_xlabel("Max trajectory depth (Å)")
        ax.set_ylabel("Count")
        ax.set_title("Trajectory max penetration depth")
        ax.legend()
    elif max_depth_arr is not None:
        ax.text(0.5, 0.5, "No penetration in dumps", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Trajectory max penetration depth")
    else:
        ax.text(0.5, 0.5, "No dump files\n(dump_mode='none' or 'etch_only'?)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Trajectory max penetration depth")

    # Panel 4: final depth vs max depth scatter (for trials with both)
    ax = axes[1, 1]
    common_trials = sorted(set(valid_max) & set(stats))
    if common_trials:
        fd = np.array([max(0, stats[t]["pen_depth"]) for t in common_trials])
        md = np.maximum(np.array([valid_max[t] for t in common_trials]), 0.0)
        ax.scatter(md, fd, s=10, alpha=0.5, color="#c44e52")
        lim = max(md.max(), fd.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
        ax.set_xlabel("Max trajectory depth (Å)")
        ax.set_ylabel("Final resting depth (Å)")
        ax.set_title("Final vs max depth")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(fontsize=8)
        ax.text(0.05, 0.92, f"n = {len(common_trials)}", transform=ax.transAxes, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No trials with both\nimpact_stats + dump",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Final vs max depth")

    fig.tight_layout()
    out_png = sim_dir / "penetration_analysis.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_png}")

    # ── Text summary ──────────────────────────────────────────────────────────
    out_txt = sim_dir / "penetration_analysis.txt"
    with out_txt.open("w") as f:
        f.write(f"Single-impact penetration analysis: {sim_dir.name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total trials:  {n_trials}\n")
        f.write(f"  Embedded:    {n_embedded} ({100*n_embedded/n_trials:.1f}%)\n")
        f.write(f"  Reflected:   {n_reflected} ({100*n_reflected/n_trials:.1f}%)\n")
        f.write(f"  Sputtered:   {n_sputtered} ({100*n_sputtered/n_trials:.1f}%)\n")
        f.write(f"\nFinal-position depth (all trials, Å):\n")
        f.write(f"  Mean (all):          {final_depths.mean():.3f}\n")
        if len(pos_final):
            f.write(f"  Mean (d>0.5Å):       {pos_final.mean():.3f}  (n={len(pos_final)})\n")
            f.write(f"  Std  (d>0.5Å):       {pos_final.std():.3f}\n")
            f.write(f"  Max:                 {final_depths.max():.3f}\n")
        if valid_max:
            f.write(f"\nTrajectory max depth ({len(valid_max)} trials with dumps, Å):\n")
            f.write(f"  Mean (all):          {max_depth_arr.mean():.3f}\n")
            if len(pen_trials := max_depth_arr[max_depth_arr > 0.5]):
                f.write(f"  Mean (d>0.5Å):       {pen_trials.mean():.3f}  (n={len(pen_trials)})\n")
                f.write(f"  Std  (d>0.5Å):       {pen_trials.std():.3f}\n")
                f.write(f"  Max:                 {max_depth_arr.max():.3f}\n")
    print(f"Saved: {out_txt}")

    return {
        "n_trials":           n_trials,
        "n_embedded":         n_embedded,
        "n_reflected":        n_reflected,
        "n_sputtered":        n_sputtered,
        "mean_final_depth":   float(final_depths.mean()),
        "mean_max_depth":     float(max_depth_arr.mean()) if max_depth_arr is not None else None,
        "max_max_depth":      float(max_depth_arr.max()) if max_depth_arr is not None else None,
    }


if __name__ == "__main__":
    sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    analyze_single_impact(sim_dir)
