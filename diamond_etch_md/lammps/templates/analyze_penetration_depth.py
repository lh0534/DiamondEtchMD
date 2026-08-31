#!/usr/bin/env python3
"""Analyze incident ZBL ion penetration depth from etch_event_trajs/event_dump_N.dump files.

Works for any ZBL ion species (Ar, Er, Kr, Xe, ...) — all are deposited as
LAMMPS atom type 4 and identified by type in the dump files.

An ion counts as penetrated only if it is still present in the final frame of
its dump (i.e. it did not reflect or get swept mid-simulation).  Depth is then:

    depth = max(surf_z - deepest_ion_z, 0)

where deepest_ion_z = min(ion z over all frames).  Outcome categories and
penetration_scatter/histogram y-axis shows max depth reached; outcome
categories (colours) are classified by final resting depth (last frame).
penetration_final_z also plots final resting depth on the y-axis.

Four outcome categories (based on surf_z − final_ion_z):
    reflected      — ion absent from final frame, OR final depth < REFLECT_THRESH
    cage           — in final frame, REFLECT_THRESH ≤ final depth < CAGE_THRESH
                     (ion trapped inside fullerene cage without deep penetration)
    1st interlayer — in final frame, CAGE_THRESH ≤ final depth < CHANNEL_THRESH
    channeled      — in final frame, final depth ≥ CHANNEL_THRESH

surf_z per trial is read from impact_stats.txt (surf_z_before_A column) when
present; otherwise computed as first-frame ion z minus i_above from config.lmp.

Writes:
  ion_z_trajectories.txt    — trial  timestep  min_ion_z_A   (one row per frame)
  ion_z_summary.txt         — trial  n_frames  deepest_ion_z_A  surf_z_A  pen_depth_A  outcome
  penetration_scatter.png   — per-trial max depth coloured by outcome
  penetration_histogram.png — final depth distribution coloured by outcome
  penetration_traces.png    — per-trial depth-vs-frame traces coloured by outcome
  penetration_final_z.png   — per-trial final resting depth coloured by outcome
  penetration_summary.png   — 3-panel: z-traces | pen+final depth per trial | final depth histogram

Usage:
    python analyze_penetration_depth.py [sim_dir]   (default: current directory)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

plt.style.use("tableau-colorblind10")

# Tableau10 colorblind palette entries
_C_REFLECTED  = "C3"
_C_CAGE       = "C0"
_C_INTERLAYER = "C7"
_C_2ND_INT    = "C1"

REFLECT_THRESH = 1.0   # Å — below this → reflected even if in final frame
CAGE_THRESH    = 4.0   # Å — cage zone 1 upper bound / 1st_interlayer lower bound
CAGE2_THRESH   = 11.9   # Å — 1st_interlayer upper bound / cage zone 2 lower bound
CHANNEL_THRESH = 12.0  # Å — cage zone 2 upper bound / 2nd_interlayer lower bound

# gradient bands: reflected | cage1 | 1st_int | cage2 | 2nd_int
# cage zones share the same colour so the gradient visually groups them
_THRESHOLDS = [REFLECT_THRESH, CAGE_THRESH, CAGE2_THRESH, CHANNEL_THRESH]
_COLORS     = [_C_REFLECTED, _C_CAGE, _C_INTERLAYER, _C_CAGE, _C_2ND_INT]


_OUTCOME_ALIASES = {
    "trapped":        "cage",
    "penetrated":     "1st_interlayer",
    "1st interlayer": "1st_interlayer",
    "1st":            "1st_interlayer",
    "2nd interlayer": "2nd_interlayer",
    "2nd":            "2nd_interlayer",
    "channeled":      "2nd_interlayer",  # old token → now 2nd_interlayer
}

def _normalize_outcome(s: str) -> str:
    return _OUTCOME_ALIASES.get(s, s)


def _outcome(depth: float, in_final: bool) -> str:
    if not in_final or depth < REFLECT_THRESH:
        return "reflected"
    if depth < CAGE_THRESH:
        return "cage"           # cage zone 1: 1–4 Å
    if depth < CAGE2_THRESH:
        return "1st_interlayer" # 4–9.5 Å
    if depth < CHANNEL_THRESH:
        return "cage"           # cage zone 2: 9.5–12 Å (counted with cage)
    return "2nd_interlayer"     # ≥12 Å


def _color(outcome: str) -> str:
    return {
        "reflected":      _C_REFLECTED,
        "cage":           _C_CAGE,
        "1st_interlayer": _C_INTERLAYER,
        "2nd_interlayer": _C_2ND_INT,
    }[outcome]


# ── dump parsing ──────────────────────────────────────────────────────────────

def _load_incident_type(sim_dir: Path) -> int:
    """Return incident_type_index from config.lmp (default 4 if not found)."""
    p = sim_dir / "config.lmp"
    if not p.exists():
        return 4
    m = re.search(r"incident_type_index\s+equal\s+(\d+)", p.read_text())
    return int(m.group(1)) if m else 4


def _substrate_atom_count(sim_dir: Path) -> int:
    """Return number of atoms in thermalized.data (0 if not found)."""
    p = sim_dir / "thermalized.data"
    if not p.exists():
        return 0
    with open(p) as fh:
        for line in fh:
            mm = re.match(r"^\s*(\d+)\s+atoms\s*$", line)
            if mm:
                return int(mm.group(1))
    return 0


def _parse_dump(path: Path, incident_type: int = 4, substrate_n: int = 0):
    """Yield (timestep, min_ion_z) per frame; value is None if no ion present.

    Detection strategy:
      incident_type == 4  → identify by atom type (ZBL species: Ar, Er, Kr, ...)
      incident_type != 4  → identify by atom ID > substrate_n (O, O2, H: type 3/2
                             is ambiguous with surface atoms; deposited ion always
                             has the highest ID(s) after read_data + fix deposit)
    """
    if incident_type != 4 and substrate_n == 0:
        # Can't reliably detect non-type-4 ion without substrate count — skip
        return

    z_col = type_col = id_col = None
    timestep = None
    in_atoms = False
    ar_zvals = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line == "ITEM: TIMESTEP":
                if timestep is not None:
                    yield timestep, (min(ar_zvals) if ar_zvals else None)
                timestep = None
                in_atoms = False
                ar_zvals = []
            elif timestep is None and not line.startswith("ITEM:"):
                try:
                    timestep = int(line)
                except ValueError:
                    pass
            elif line.startswith("ITEM: ATOMS"):
                cols     = line.split()[2:]
                z_col    = cols.index("z")
                type_col = cols.index("type")
                id_col   = cols.index("id") if "id" in cols else None
                in_atoms = True
            elif in_atoms and line:
                parts = line.split()
                if incident_type == 4:
                    is_ion = parts[type_col] == "4"
                else:
                    is_ion = (id_col is not None and int(parts[id_col]) > substrate_n)
                if is_ion:
                    ar_zvals.append(float(parts[z_col]))

    if timestep is not None:
        yield timestep, (min(ar_zvals) if ar_zvals else None)


def _load_surf_z(sim_dir: Path) -> dict:
    """Return {trial: surf_z_before_A} from impact_stats.txt."""
    result = {}
    p = sim_dir / "impact_stats.txt"
    if not p.exists():
        return result
    with open(p) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            try:
                result[int(parts[0])] = float(parts[1])
            except (IndexError, ValueError):
                pass
    return result


def _load_spec(sim_dir: Path) -> dict:
    """Load spec.json; return {} if not found."""
    p = sim_dir / "spec.json"
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def _plot_title(spec: dict, n_total: int) -> str:
    """Build a one-line plot title from spec fields."""
    ion    = spec.get("species", "?")
    energy = spec.get("energy", "?")
    angle  = spec.get("ion_angle", 0.0)
    if isinstance(angle, (list, tuple)):
        angle = angle[0]
    angle_str = f"{angle:.0f}°"
    return f"{ion}$^+$ Implantation:  {energy} eV  {angle_str}  (n={n_total})"


def _config_i_above(sim_dir: Path) -> float:
    p = sim_dir / "config.lmp"
    if not p.exists():
        return 0.0
    m = re.search(r"variable\s+i_above\s+equal\s+([\d.]+)", p.read_text())
    return float(m.group(1)) if m else 0.0


# ── background gradient ───────────────────────────────────────────────────────

def _bg_gradient(ax, thresholds, colors, axis='y', alpha=0.18, blend=0.75, n=300):
    """Fill axis background with colored regions and smooth gradient transitions.

    thresholds: boundary values between color regions (N-1 values for N colors)
    colors:     one color per region, ordered from low to high
    blend:      half-width of the gradient transition zone in data units
    """
    lo, hi = ax.get_ylim() if axis == 'y' else ax.get_xlim()
    vals = np.linspace(lo, hi, n + 1)
    mids = 0.5 * (vals[:-1] + vals[1:])
    c_rgba = [np.array(mcolors.to_rgba(c)) for c in colors]

    for j, v in enumerate(mids):
        reg = sum(v >= t for t in thresholds)
        c = c_rgba[min(reg, len(c_rgba) - 1)].copy()

        for b_idx, b in enumerate(thresholds):
            if abs(v - b) < blend:
                t = (v - b + blend) / (2 * blend)
                c = (1 - t) * c_rgba[b_idx] + t * c_rgba[b_idx + 1]
                break

        kw = dict(color=c[:3], alpha=alpha * float(c[3]), lw=0, zorder=0)
        (ax.axhspan if axis == 'y' else ax.axvspan)(vals[j], vals[j + 1], **kw)

    # restore limits — axhspan/axvspan can nudge them
    if axis == 'y':
        ax.set_ylim(lo, hi)
    else:
        ax.set_xlim(lo, hi)


# ── plotting ──────────────────────────────────────────────────────────────────

def _legend_labels(outcomes, n_total):
    """Return {outcome: "Label (N%)"} for each category present."""
    if not n_total:
        pct = lambda n: "0%"
    else:
        pct = lambda n: f"{100*n/n_total:.0f}%"
    n_ref  = outcomes.count("reflected")
    n_cage = outcomes.count("cage")
    n_int  = outcomes.count("1st_interlayer")
    n_int2 = outcomes.count("2nd_interlayer")
    return {
        "reflected":      f"Reflected ({pct(n_ref)})",
        "cage":           f"Cage ({pct(n_cage)})",
        "1st_interlayer": f"1st Interlayer ({pct(n_int)})",
        "2nd_interlayer": f"2nd Interlayer ({pct(n_int2)})",
    }


def _scatter(trials, depths, outcomes, sim_dir: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = _legend_labels(outcomes, len(outcomes))

    for outcome, color in [
        ("reflected",      _C_REFLECTED),
        ("cage",           _C_CAGE),
        ("1st_interlayer", _C_INTERLAYER),
        ("2nd_interlayer", _C_2ND_INT),
    ]:
        mask = np.array([o == outcome for o in outcomes])
        if mask.any():
            ax.scatter(trials[mask], depths[mask], s=20, alpha=0.7, zorder=2,
                       color=color, label=labels[outcome])
        else:
            ax.scatter([], [], s=20, alpha=0.7, color=color, label=labels[outcome])

    ymax = max(depths.max() * 1.1, CHANNEL_THRESH * 1.15) if len(depths) else CHANNEL_THRESH * 1.2
    ax.set_ylim(ymax, 0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='x', color='white', linewidth=0.6, alpha=0.6, zorder=1)
    ax.grid(which='minor', axis='x', color='white', linewidth=0.3, alpha=0.35, zorder=1)
    # _bg_gradient(ax, _THRESHOLDS, _COLORS, axis='y')

    ax.set_xlabel("Trial #")
    ax.set_ylabel("Depth (Å)")
    ax.set_title(title, fontsize=12, y=1.05)
    ax.legend(fontsize=7, frameon=False, ncol=4, labelspacing=0.2, columnspacing=1.0,
              handlelength=1.2, handletextpad=0.2, bbox_to_anchor=(0.5, 0.99
              ), loc="lower center")
    fig.tight_layout()
    out = sim_dir / "penetration_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _histogram(depths, outcomes, sim_dir: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = _legend_labels(outcomes, len(outcomes))

    depths_arr   = np.array(depths)
    outcomes_arr = np.array(outcomes)

    if len(depths_arr) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ymax = CHANNEL_THRESH * 1.2
        bins = np.linspace(0, ymax, 25)
    else:
        ymax = max(depths_arr.max() * 1.05, CHANNEL_THRESH * 1.15)
        bins = np.linspace(0, ymax, 25)

    for outcome, color in [
        ("reflected",      _C_REFLECTED),
        ("cage",           _C_CAGE),
        ("1st_interlayer", _C_INTERLAYER),
        ("2nd_interlayer", _C_2ND_INT),
    ]:
        mask = outcomes_arr == outcome if len(outcomes_arr) else np.array([], dtype=bool)
        data = depths_arr[mask] if mask.any() else np.array([])
        ax.hist(data, bins=bins, color=color, edgecolor="white",
                linewidth=0.4, alpha=0.7, zorder=2, label=labels[outcome],
                orientation="horizontal")

    ax.set_ylim(ymax, 0)
    # _bg_gradient(ax, _THRESHOLDS, _COLORS, axis='y')

    ax.set_xlabel("Count")
    ax.set_ylabel("Depth (Å)")
    ax.set_title(title, fontsize=12, y=1.05)
    ax.legend(fontsize=7, frameon=False, ncol=4, labelspacing=0.2, columnspacing=1.0,
              handlelength=1.2, handletextpad=0.2, bbox_to_anchor=(0.5, 0.99), loc="lower center")
    fig.tight_layout()
    out = sim_dir / "penetration_histogram.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")



def _ztrace(sim_dir: Path, spec: dict, outcomes: list, title: str = "") -> None:
    """Plot per-trial ion depth traces coloured by outcome; save penetration_traces.png."""
    traj_path = sim_dir / "ion_z_trajectories.txt"
    summ_path = sim_dir / "ion_z_summary.txt"
    if not traj_path.exists() or not summ_path.exists():
        print("ztrace: missing ion_z_trajectories.txt or ion_z_summary.txt — skipping")
        return

    trajs = _load_trajectories(sim_dir)
    surf_z_map, outcome_map = _read_surf_z_outcome_map(summ_path)

    if not trajs:
        print("ztrace: no trajectory data — skipping")
        return

    n_total = len(outcomes)
    labels = _legend_labels(outcomes, n_total)

    fig, ax = plt.subplots(figsize=(5.5, 2.5))

    # Plot traces; track which legend entries have been drawn
    _drawn = set()
    for trial, frames in sorted(trajs.items()):
        outcome = outcome_map.get(trial, "reflected")
        surf_z  = surf_z_map.get(trial, np.nan)
        color   = _color(outcome)

        xs = list(range(len(frames)))
        ys = [max(surf_z - z, 0.0) if np.isfinite(surf_z) else 0.0
              for _, z in frames]

        lbl = labels[outcome] if outcome not in _drawn else None
        ax.plot(xs, ys, color=color, alpha=0.5, linewidth=0.8, label=lbl, zorder=2)
        _drawn.add(outcome)

    # Dummy lines for any outcome categories not present in the data
    for outcome, color in [
        ("reflected",      _C_REFLECTED),
        # ("cage",           _C_CAGE),
        ("1st_interlayer", _C_INTERLAYER),
        ("2nd_interlayer", _C_2ND_INT),
    ]:
        if outcome not in _drawn:
            ax.plot([], [], color=color, alpha=0.7, linewidth=1.0,
                    label=labels[outcome])

    # y-axis: depth increasing downward
    all_depths = [
        max(surf_z_map.get(tr, np.nan) - z, 0.0)
        for tr, frs in trajs.items()
        for _, z in frs
        if np.isfinite(surf_z_map.get(tr, np.nan))
    ]
    ymax = max(max(all_depths) * 1.1, CHANNEL_THRESH * 1.15) if all_depths else CHANNEL_THRESH * 1.2
    ax.set_ylim(ymax, 0)
    # _bg_gradient(ax, _THRESHOLDS, _COLORS, axis='y')

    ax.text(0.02, 0.05, f"{spec.get("energy", "?")} eV {spec.get("species", "?")}$^+$", fontsize=12, color="black", transform=ax.transAxes)
    ax.set_xlabel("Frame (adaptive timestep)")
    ax.set_ylabel("Depth (Å)")
    ax.set_xlim(0, 70)
    ax.set_ylim(18, 0)
    # ax.set_title(title, fontsize=12, y=1.05)
    ax.legend(fontsize=7, frameon=False, ncol=4, labelspacing=0.2, columnspacing=1.0, markerscale=2.0,
              handlelength=1.2, handletextpad=0.2, bbox_to_anchor=(0.5, 0.99), loc="lower center")
    fig.tight_layout()
    out = sim_dir / "penetration_traces.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _scatter_final_z(trials, depths, outcomes, sim_dir: Path, title: str = "") -> None:
    """Plot final resting depth (surf_z - last-frame ion z) per trial; save penetration_final_z.png."""
    traj_path = sim_dir / "ion_z_trajectories.txt"
    summ_path = sim_dir / "ion_z_summary.txt"
    if not traj_path.exists() or not summ_path.exists():
        print("scatter_final_z: missing trajectory/summary files — skipping")
        return

    trajs = _load_trajectories(sim_dir)
    surf_z_map, outcome_map = _read_surf_z_outcome_map(summ_path)

    if not trajs:
        print("scatter_final_z: no trajectory data — skipping")
        return

    n_total = len(outcomes)
    labels = _legend_labels(outcomes, n_total)

    trial_nums = sorted(trajs.keys())
    # Final depth = surf_z - last observed ion z (0 if ion absent or surf_z unknown)
    final_depths = {}
    for trial, frames in trajs.items():
        if not frames:
            continue
        last_z = frames[-1][1]
        sz = surf_z_map.get(trial, np.nan)
        if np.isfinite(sz):
            final_depths[trial] = max(sz - last_z, 0.0)

    fig, ax = plt.subplots(figsize=(4, 3))

    for outcome, color in [
        ("reflected",      _C_REFLECTED),
        ("cage",           _C_CAGE),
        ("1st_interlayer", _C_INTERLAYER),
        ("2nd_interlayer", _C_2ND_INT),
    ]:
        t_vals = [t for t in trial_nums if outcome_map.get(t) == outcome and t in final_depths]
        d_vals = [final_depths[t] for t in t_vals]
        if t_vals:
            ax.scatter(t_vals, d_vals, s=20, alpha=0.7, zorder=2,
                       color=color, label=labels[outcome])
        else:
            ax.scatter([], [], s=20, alpha=0.7, color=color, label=labels[outcome])

    all_fd = list(final_depths.values())
    ymax = max(max(all_fd) * 1.1, CHANNEL_THRESH * 1.15) if all_fd else CHANNEL_THRESH * 1.2
    ax.set_ylim(ymax, 0)
    # _bg_gradient(ax, _THRESHOLDS, _COLORS, axis='y')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='x', color='white', linewidth=0.6, alpha=0.6, zorder=1)
    ax.grid(which='minor', axis='x', color='white', linewidth=0.3, alpha=0.35, zorder=1)

    ax.set_xlabel("Trial #")
    ax.set_ylabel("Depth (Å)")
    ax.set_title(title, fontsize=12, y=1.05)
    ax.legend(fontsize=7, frameon=False, ncol=4, labelspacing=0.2, columnspacing=1.0,
              handlelength=1.2, handletextpad=0.2, bbox_to_anchor=(0.5, 0.99), loc="lower center")
    fig.tight_layout()
    out = sim_dir / "penetration_final_z.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _get_final_depths(sim_dir: Path, trials_l: list) -> np.ndarray:
    """Return array of final resting depths (surf_z - last_frame_z) per trial, aligned to trials_l."""
    trajs = _load_trajectories(sim_dir)
    summ_path = sim_dir / "ion_z_summary.txt"
    if not trajs or not summ_path.exists():
        return np.zeros(len(trials_l))
    surf_z_map, _ = _read_surf_z_outcome_map(summ_path)
    out = np.zeros(len(trials_l))
    for i, trial in enumerate(trials_l):
        frames = trajs.get(trial, [])
        if frames:
            last_z = frames[-1][1]
            sz = surf_z_map.get(trial, np.nan)
            if np.isfinite(sz):
                out[i] = max(sz - last_z, 0.0)
    return out


def _summary(sim_dir: Path, trials, pen_depths, final_depths, outcomes, title: str = "") -> None:
    """2-panel summary figure: z-traces | pen+final depth per trial."""
    traj_path = sim_dir / "ion_z_trajectories.txt"
    summ_path = sim_dir / "ion_z_summary.txt"
    if not traj_path.exists() or not summ_path.exists():
        print("summary: missing trajectory/summary files — skipping")
        return

    trajs = _load_trajectories(sim_dir)
    surf_z_map, outcome_map = _read_surf_z_outcome_map(summ_path)

    n_total = len(outcomes)
    labels  = _legend_labels(outcomes, n_total)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True,
                                    gridspec_kw={"wspace": 0, "width_ratios": [1, 1.2]})
    fig.suptitle(title, fontsize=18, y=.92)

    # ── shared y-axis range ────────────────────────────────────────────────
    trials_arr   = np.array(trials)
    pd_arr       = np.array(pen_depths)
    fd_arr       = np.array(final_depths)
    outcomes_arr = np.array(outcomes)

    all_tr_depths = [
        max(surf_z_map.get(tr, np.nan) - z, 0.0)
        for tr, frs in trajs.items()
        for _, z in frs
        if np.isfinite(surf_z_map.get(tr, np.nan))
    ]
    ymax = max(
        max(all_tr_depths) * 1.1 if all_tr_depths else 0,
        pd_arr.max() * 1.1 if len(pd_arr) else 0,
        fd_arr.max() * 1.05 if len(fd_arr) else 0,
        CHANNEL_THRESH * 1.15,
    )

    # ── Panel 1: z-traces (hosts the shared outcome legend) ───────────────
    _drawn = set()
    for trial, frames in sorted(trajs.items()):
        outcome = outcome_map.get(trial, "reflected")
        surf_z  = surf_z_map.get(trial, np.nan)
        color   = _color(outcome)
        xs = list(range(len(frames)))
        ys = [max(surf_z - z, 0.0) if np.isfinite(surf_z) else 0.0 for _, z in frames]
        lbl = labels[outcome] if outcome not in _drawn else None
        ax1.plot(xs, ys, color=color, alpha=0.5, linewidth=0.8, label=lbl, zorder=2)
        _drawn.add(outcome)
    for outcome, color in [
        ("reflected",      _C_REFLECTED),
        # ("cage",           _C_CAGE),
        ("1st_interlayer", _C_INTERLAYER),
        ("2nd_interlayer", _C_2ND_INT),
    ]:
        if outcome not in _drawn:
            ax1.plot([], [], color=color, alpha=0.7, linewidth=1.0, label=labels[outcome])
    ax1.set_ylim(ymax, 0)
    # _bg_gradient(ax1, _THRESHOLDS, _COLORS, axis='y')
    ax1.set_xlabel("Frame (adaptive timestep)")
    ax1.set_ylabel("Depth (Å)")
    ax1.legend(fontsize=10, frameon=False, ncol=4, labelspacing=0.1,
               markerscale=5.0, handlelength=1.0, handletextpad=0.3,
               loc="upper center", bbox_to_anchor=(1, 1.13))

    # ── Panel 2: pen depth + final depth per trial, connected by line ──────
    for outcome, color in [
        ("reflected",      _C_REFLECTED),
        # ("cage",           _C_CAGE),
        ("1st_interlayer", _C_INTERLAYER),
        ("2nd_interlayer", _C_2ND_INT),
    ]:
        mask = outcomes_arr == outcome
        if not mask.any():
            continue
        t_vals  = trials_arr[mask]
        pd_vals = pd_arr[mask]
        fd_vals = fd_arr[mask]
        ax2.vlines(t_vals, fd_vals, pd_vals, color=color, alpha=0.35, linewidth=0.9, zorder=2)
        ax2.scatter(t_vals, pd_vals, s=14, color=color, alpha=0.85, zorder=3, marker='o')
        ax2.scatter(t_vals, fd_vals, s=14, facecolors='none', edgecolors=color,
                    alpha=0.85, linewidths=0.8, zorder=3, marker='o')

    ax2.set_ylim(ymax, 0)
    # _bg_gradient(ax2, _THRESHOLDS, _COLORS, axis='y')
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: str(int(x)) if int(round(x)) % 10 == 0 else "")
    )
    ax2.grid(which='major', axis='x', color='white', linewidth=0.6, alpha=0.6, zorder=1)
    ax2.grid(which='minor', axis='x', color='white', linewidth=0.3, alpha=0.3, zorder=1)
    ax2.tick_params(labelleft=False)
    ax2.set_xlabel("Trial #")
    # marker-type legend (filled vs. open)
    ax2.scatter([], [], s=14, color='grey', marker='o', alpha=0.7, label='Max depth')
    ax2.scatter([], [], s=14, facecolors='none', edgecolors='grey',
                linewidths=0.8, marker='o', alpha=0.7, label='Final depth')
    ax2.legend(fontsize=10, frameon=False, ncol=2, labelspacing=0.1,
               handlelength=1.0, handletextpad=0.3, markerscale=2.0)

    fig.tight_layout()
    out = sim_dir / "penetration_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── summary I/O ───────────────────────────────────────────────────────────────

def _load_trajectories(sim_dir: Path) -> dict:
    """Return {trial: [(timestep, ion_z), ...]} from ion_z_trajectories.txt."""
    p = sim_dir / "ion_z_trajectories.txt"
    if not p.exists():
        return {}
    trajs = {}
    with open(p) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            trial = int(parts[0])
            trajs.setdefault(trial, []).append((int(parts[1]), float(parts[2])))
    return trajs


def _read_surf_z_outcome_map(path: Path) -> tuple:
    """Return ({trial: surf_z}, {trial: outcome}) from ion_z_summary.txt."""
    surf_z_map, outcome_map = {}, {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            trial = int(parts[0])
            surf_z_map[trial]  = float(parts[3])
            outcome_map[trial] = _normalize_outcome(" ".join(parts[5:]))
    return surf_z_map, outcome_map


def _read_summary(path: Path):
    """Load trials/depths/outcomes from an existing ion_z_summary.txt."""
    trials, depths, outcomes = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            trials.append(int(parts[0]))
            depths.append(float(parts[4]))
            outcomes.append(_normalize_outcome(" ".join(parts[5:])))
    return trials, depths, outcomes


def _write_summary(sim_dir, files, surf_z_map, i_above, incident_type=4, substrate_n=0):
    """Parse dumps, write ion_z_trajectories.txt + ion_z_summary.txt, return lists."""
    traj_path = sim_dir / "ion_z_trajectories.txt"
    summ_path = sim_dir / "ion_z_summary.txt"
    trials_l, depths_l, outcomes_l = [], [], []

    with open(traj_path, "w") as traj, open(summ_path, "w") as summ:
        traj.write("# trial  timestep  min_ion_z_A\n")
        summ.write("# trial  n_frames  deepest_ion_z_A  surf_z_A  pen_depth_A  outcome\n")

        for f in files:
            trial  = int(re.search(r"(\d+)", f.stem).group(1))
            frames = list(_parse_dump(f, incident_type=incident_type, substrate_n=substrate_n))

            for ts, z in frames:
                if z is not None:
                    traj.write(f"{trial}  {ts}  {z:.6f}\n")

            valid_z  = [z for _, z in frames if z is not None]
            deepest  = min(valid_z) if valid_z else np.nan
            final_z  = valid_z[-1] if valid_z else np.nan
            in_final = frames[-1][1] is not None if frames else False

            if trial in surf_z_map:
                surf_z = surf_z_map[trial]
            elif valid_z:
                surf_z = valid_z[0] - i_above
            else:
                surf_z = np.nan

            # pen_depth: max depth reached (y-axis of scatter/histogram)
            if np.isfinite(deepest) and np.isfinite(surf_z):
                depth = max(surf_z - deepest, 0.0)
            else:
                depth = 0.0

            # Classify by final resting depth, not max depth
            if in_final and np.isfinite(final_z) and np.isfinite(surf_z):
                final_depth = max(surf_z - final_z, 0.0)
            else:
                final_depth = 0.0
            outcome = _outcome(final_depth, in_final)
            summ.write(
                f"{trial}  {len(frames)}  "
                f"{deepest:.6f}  {surf_z:.6f}  {depth:.6f}  {outcome}\n"
            )
            trials_l.append(trial)
            depths_l.append(depth)
            outcomes_l.append(outcome)

    print(f"Wrote {traj_path}")
    print(f"Wrote {summ_path}")
    return trials_l, depths_l, outcomes_l


# ── main ──────────────────────────────────────────────────────────────────────

def analyze_ar_z(sim_dir: Path, recompute: bool = False) -> dict:
    traj_path = sim_dir / "ion_z_trajectories.txt"
    summ_path = sim_dir / "ion_z_summary.txt"

    incident_type = _load_incident_type(sim_dir)
    substrate_n   = _substrate_atom_count(sim_dir) if incident_type != 4 else 0

    if not recompute and traj_path.exists() and summ_path.exists():
        print(f"Loading existing {summ_path.name}  (use --recompute to regenerate)")
        trials_l, depths_l, outcomes_l = _read_summary(summ_path)
        i_above = _config_i_above(sim_dir)
    else:
        dump_dir = sim_dir / "etch_event_trajs"
        if not dump_dir.exists():
            print("No etch_event_trajs/ directory found.")
            return {}
        files = sorted(
            dump_dir.glob("event_dump_*.dump"),
            key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)),
        )
        if not files:
            print("No event_dump_*.dump files found.")
            return {}
        surf_z_map = _load_surf_z(sim_dir)
        i_above    = _config_i_above(sim_dir)
        trials_l, depths_l, outcomes_l = _write_summary(
            sim_dir, files, surf_z_map, i_above,
            incident_type=incident_type, substrate_n=substrate_n,
        )

    trials   = np.array(trials_l)
    depths   = np.array(depths_l)
    outcomes = outcomes_l

    n_total          = len(trials)
    n_reflected      = outcomes.count("reflected")
    n_cage           = outcomes.count("cage")
    n_1st_interlayer = outcomes.count("1st_interlayer")
    n_2nd_interlayer = outcomes.count("2nd_interlayer")
    embed_rate       = 100.0 * (n_cage + n_1st_interlayer + n_2nd_interlayer) / n_total if n_total else 0.0

    det = f"type 4" if incident_type == 4 else f"type {incident_type}, ID>{substrate_n}"
    print(f"Ion penetration analysis: {n_total} trials  (i_above={i_above} Å, detect={det})")
    print(f"  Reflected      (<{REFLECT_THRESH} Å final or not in final frame): {n_reflected}")
    print(f"  Cage           ({REFLECT_THRESH}–{CAGE_THRESH} Å + {CAGE2_THRESH}–{CHANNEL_THRESH} Å): {n_cage}")
    print(f"  1st Interlayer ({CAGE_THRESH}–{CAGE2_THRESH} Å): {n_1st_interlayer}")
    print(f"  2nd Interlayer (≥{CHANNEL_THRESH} Å): {n_2nd_interlayer}")
    print(f"  Embedded total: {n_cage+n_1st_interlayer+n_2nd_interlayer} / {n_total} = {embed_rate:.1f}%")
    if (depths > 0).any():
        pos = depths[depths > 0]
        print(f"  Mean depth (depth>0): {pos.mean():.2f} Å")
        print(f"  Max depth:            {pos.max():.2f} Å")

    spec_data = _load_spec(sim_dir)
    title = _plot_title(spec_data, n_total) if spec_data else sim_dir.name

    final_depths_arr = _get_final_depths(sim_dir, trials_l)

    _scatter(trials, depths, outcomes, sim_dir, title=title)
    _histogram(final_depths_arr, outcomes, sim_dir, title=title)
    _ztrace(sim_dir, spec_data, outcomes, title=title)
    _scatter_final_z(trials, depths, outcomes, sim_dir, title=title)
    _summary(sim_dir, trials, depths, final_depths_arr, outcomes, title=title)

    return {
        "n_trials":          n_total,
        "n_reflected":       n_reflected,
        "n_cage":            n_cage,
        "n_1st_interlayer":  n_1st_interlayer,
        "n_2nd_interlayer":  n_2nd_interlayer,
        "penetration_rate":  embed_rate,
        "mean_depth":        float(depths[depths > 0].mean()) if (depths > 0).any() else 0.0,
        "max_depth":         float(depths.max()),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sim_dir", nargs="?", default=".", help="simulation directory (default: .)")
    parser.add_argument("--recompute", action="store_true",
                        help="re-parse dumps and overwrite ion_z_trajectories.txt / ion_z_summary.txt")
    args = parser.parse_args()
    analyze_ar_z(Path(args.sim_dir), recompute=args.recompute)
