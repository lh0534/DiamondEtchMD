"""
analysis/plot.py — analysis plots for DiamondEtchMD simulations.

Main entry point: make_plots(sim_dir, spec=None)

Produces PNGs in sim_dir:
  etch.png               — etch depth (ML) vs ion dose (ML)
  o_uptake.png           — surface O (ML) vs dose
  product_grid.png       — 2-D count heatmap: n_C vs n_O
  product_trajectory.png — cumulative yield per product species vs dose
  amorphous.png          — amorphous C (ML) and sp3 fraction vs dose
  amorphous_thickness.png — amorphous layer thickness (Å) vs dose
  etch_per_cycle.png     — etch per cycle (cycling only)

Requires matplotlib and numpy (install with pip install 'DiamondEtchMD[analysis]').
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _need_mpl():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install 'DiamondEtchMD[analysis]'"
        )


# ── shared helpers ────────────────────────────────────────────────────────────

def _product_label(nc, nh, no):
    parts = []
    if nc:
        parts.append('C' + (str(nc) if nc > 1 else ''))
    if nh:
        parts.append('H' + (str(nh) if nh > 1 else ''))
    if no:
        parts.append('O' + (str(no) if no > 1 else ''))
    return ''.join(parts) or '?'


def _parse_nc(path):
    """Return (records, is_cycling) reading ncarbon.txt directly."""
    from .ncarbon import parse_ncarbon
    recs = parse_ncarbon(path)
    return recs, any(r['cn'] > 0 for r in recs)


def _parse_ep(path):
    from .etch_products import parse_etch_products
    return parse_etch_products(path) if Path(path).exists() else []


def _load_spec(spec_path):
    from ..spec import SimSpec, CyclePhase
    data = json.loads(Path(spec_path).read_text())
    phases_data = data.pop('phases', None)
    data['phases'] = [CyclePhase(**p) for p in phases_data] if phases_data else None
    # drop keys not in SimSpec to be forward-compatible
    import dataclasses
    valid = {f.name for f in dataclasses.fields(SimSpec)}
    data = {k: v for k, v in data.items() if k in valid}
    return SimSpec(**data)


def _draw_phase_lines(ax, spec, ml):
    """Draw vertical dotted lines at cycling phase boundaries."""
    if spec is None or spec.phases is None:
        return
    cmap = plt.get_cmap('Set1')
    c = 0
    drawn = set()
    for cycle in range(spec.cycles):
        for pi, p in enumerate(spec.phases):
            c += p.fluence_ml * ml
            x = c / ml
            lbl = p.species if (pi not in drawn and cycle == 0) else None
            ax.axvline(x=x, ls=':', lw=0.7, color=cmap(pi % 9), alpha=0.6, label=lbl)
            drawn.add(pi)


# ── individual plot functions ─────────────────────────────────────────────────

def plot_etch(nc_records, ml, spec=None, ax=None):
    """Etch depth (ML) vs ion dose (ML); adds phase lines for cycling."""
    _need_mpl()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))

    # Use only ion-impact rows for x-axis; all rows for depth
    impacts = np.array([r['impact'] for r in nc_records]) / ml
    n0 = nc_records[0]['n_carbon']
    depth = np.array([(n0 - r['n_carbon']) / ml for r in nc_records])

    ax.plot(impacts, depth, lw=1.2, color='C0')
    if spec is not None and spec.phases is not None:
        _draw_phase_lines(ax, spec, ml)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, fontsize='x-small', title='Phase end',
                      loc='upper left')

    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('Etch depth (ML)')
    ax.set_title('Etch depth vs dose')
    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_o_uptake(nc_records, ml, spec=None, ax=None):
    """Surface O content (ML) vs dose (ML)."""
    _need_mpl()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))

    impacts = np.array([r['impact'] for r in nc_records]) / ml
    o_ml = np.array([r['n_oxygen'] / ml for r in nc_records])

    ax.plot(impacts, o_ml, lw=1.2, color='C3')
    if spec is not None and spec.phases is not None:
        _draw_phase_lines(ax, spec, ml)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, fontsize='x-small', title='Phase end',
                      loc='upper left')

    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('Surface O (ML)')
    ax.set_title('Oxygen uptake vs dose')
    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_product_grid(ep_records, ax=None):
    """2-D count heatmap: n_C (x-axis) vs n_O (y-axis)."""
    _need_mpl()
    if not ep_records:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 5))

    max_C = max(r['n_C'] for r in ep_records)
    max_O = max(r['n_O'] for r in ep_records)
    grid = np.zeros((max_O + 1, max_C + 1))
    for r in ep_records:
        grid[r['n_O'], r['n_C']] += 1

    im = ax.imshow(
        grid, origin='lower', aspect='auto', cmap='viridis',
        extent=[-0.5, max_C + 0.5, -0.5, max_O + 0.5],
    )
    plt.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('C atoms in product')
    ax.set_ylabel('O atoms in product')
    ax.set_title('Product composition distribution')
    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_product_trajectory(ep_records, ml, ax=None):
    """Cumulative yield (ML) per unique product species vs dose (ML).

    Each product species (identified by its n_C, n_H, n_O composition) gets
    one line showing its running cumulative count divided by ml.  Legend is
    placed below the plot.
    """
    _need_mpl()
    if not ep_records:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    by_product = defaultdict(list)
    for r in ep_records:
        by_product[(r['n_C'], r['n_H'], r['n_O'])].append(r['impact'])

    # Sort by total count descending so the most common products are drawn first
    products = sorted(by_product.items(), key=lambda kv: -len(kv[1]))
    max_impact = max(r['impact'] for r in ep_records)
    x = np.arange(1, max_impact + 1) / ml

    cmap = plt.get_cmap('tab20')
    for idx, ((nc, nh, no), imp_list) in enumerate(products):
        counts = np.zeros(max_impact)
        for imp in imp_list:
            if 1 <= imp <= max_impact:
                counts[imp - 1] += 1
        cumulative = np.cumsum(counts) / ml
        ax.plot(x, cumulative, lw=1.0, label=_product_label(nc, nh, no),
                color=cmap(idx % 20))

    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('Cumulative yield (ML)')
    ax.set_title('Etch product trajectory')

    n_cols = max(1, min(len(products), 8))
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.18),
        ncol=n_cols, fontsize='x-small', frameon=True,
    )
    if own:
        plt.subplots_adjust(bottom=0.28)
        return ax.figure
    return ax


def plot_amorphous(cna_records, ml, ax=None):
    """Amorphous C (ML, left axis) and sp3 fraction (right axis) vs dose (ML)."""
    _need_mpl()
    if not cna_records:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))

    impacts = np.array([r['impact'] for r in cna_records]) / ml
    am_ml = np.array([r['n_amorphous'] / ml for r in cna_records])
    sp3 = np.array([r['sp3_fraction'] for r in cna_records])

    ax.plot(impacts, am_ml, lw=1.2, color='C1', label='Amorphous C (ML)')
    ax2 = ax.twinx()
    ax2.plot(impacts, sp3, lw=1.0, ls='--', color='C2', label='sp3 fraction')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('sp3 fraction')

    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('Amorphous C (ML)')
    ax.set_title('Amorphous carbon vs dose (CNA)')

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize='small', loc='best')
    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_amorphous_thickness(cna_records, ml, ax=None):
    """Amorphous layer thickness (Å) vs dose (ML)."""
    _need_mpl()
    if not cna_records:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))

    impacts = np.array([r['impact'] for r in cna_records]) / ml
    thickness = np.array([r['amorphous_thickness_A'] for r in cna_records])

    ax.plot(impacts, thickness, lw=1.2, color='C4')
    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('Amorphous layer thickness (Å)')
    ax.set_title('Surface disorder depth (10%–90% density criterion)')
    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_etch_per_cycle(nc_records, spec, ml, ax=None):
    """Etch per cycle (ML) vs cycle number (cycling mode only)."""
    _need_mpl()
    if spec is None or spec.phases is None:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))

    total_cycle_ml = sum(p.fluence_ml for p in spec.phases) * ml
    ion_recs = [r for r in nc_records if r['cn'] == 0]
    n0 = nc_records[0]['n_carbon'] if nc_records else 0

    cycle_etch = []
    prev = n0
    for cyc in range(spec.cycles):
        start, end = cyc * total_cycle_ml, (cyc + 1) * total_cycle_ml
        recs = [r for r in ion_recs if start < r['impact'] <= end]
        if recs:
            last = recs[-1]['n_carbon']
            cycle_etch.append((prev - last) / ml)
            prev = last
        else:
            cycle_etch.append(0.0)

    ax.bar(range(1, len(cycle_etch) + 1), cycle_etch, color='C0', width=0.7)
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Etch per cycle (ML)')
    ax.set_title('Etch per cycle vs cycle number')
    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_per_phase_yield(nc_records, spec, ml, ax=None):
    """Per-phase average etch yield (bar chart, cycle-averaged, cycling only)."""
    _need_mpl()
    if spec is None or spec.phases is None:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 4))

    from .ncarbon import parse_ncarbon_cycling
    # parse_ncarbon_cycling needs a path — but we already have records.
    # Re-compute from stats dict instead.
    phase_yields = defaultdict(list)
    phase_names = {i: p.species for i, p in enumerate(spec.phases)}

    total_cycle_ml = sum(p.fluence_ml for p in spec.phases) * ml
    ion_recs = [r for r in nc_records if r['cn'] == 0]

    prev_pre_nc = nc_records[0]['n_carbon'] if nc_records else 0
    pre_nc = prev_pre_nc
    for r in nc_records:
        if r['cn'] == 0:
            impact_in_cycle = (r['impact'] - 1) % total_cycle_ml
            cum = 0
            for pi, p in enumerate(spec.phases):
                cum += p.fluence_ml * ml
                if impact_in_cycle < cum:
                    phase_yields[pi].append(pre_nc - r['n_carbon'])
                    break
            pre_nc = r['n_carbon']
        else:
            pre_nc = r['n_carbon']

    labels = [f"{phase_names.get(pi, pi)}" for pi in sorted(phase_yields)]
    means = [np.mean(phase_yields[pi]) if phase_yields[pi] else 0.0
             for pi in sorted(phase_yields)]

    cmap = plt.get_cmap('Set2')
    bars = ax.bar(labels, means, color=[cmap(i) for i in range(len(labels))])
    ax.set_ylabel('Avg etch yield (C/ion)')
    ax.set_title('Per-phase etch yield (cycle-averaged)')
    ax.bar_label(bars, fmt='%.3g', fontsize='small')
    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_o_per_cycle(nc_records, spec, ml, ax=None):
    """Per-phase O loading/unloading per cycle (cycling only)."""
    _need_mpl()
    if spec is None or spec.phases is None:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))

    total_cycle_ml = sum(p.fluence_ml for p in spec.phases) * ml
    ion_recs = [r for r in nc_records if r['cn'] == 0]

    cmap = plt.get_cmap('Set1')
    for pi, p in enumerate(spec.phases):
        phase_start_ml = sum(spec.phases[j].fluence_ml for j in range(pi)) * ml
        phase_end_ml = phase_start_ml + p.fluence_ml * ml
        o_end = []
        for cyc in range(spec.cycles):
            lo = cyc * total_cycle_ml + phase_start_ml
            hi = cyc * total_cycle_ml + phase_end_ml
            recs = [r for r in ion_recs if lo < r['impact'] <= hi]
            if recs:
                o_end.append(recs[-1]['n_oxygen'] / ml)
        if o_end:
            ax.plot(range(1, len(o_end) + 1), o_end,
                    marker='o', ms=4, lw=1.0,
                    color=cmap(pi % 9), label=f"After {p.species}")

    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Surface O at phase end (ML)')
    ax.set_title('O loading/unloading per cycle')
    ax.legend(fontsize='small')
    if own:
        plt.tight_layout()
        return ax.figure
    return ax


# ── main entry point ─────────────────────────────────────────────────────────

def make_plots(
    sim_dir,
    spec=None,
    ml: int = 0,
    cna_records=None,
    save: bool = True,
) -> Dict:
    """Generate all analysis plots for a simulation directory.

    Saves PNGs to sim_dir (if save=True) and returns a dict of {name: Figure}.

    Parameters
    ----------
    sim_dir     : path to simulation directory
    spec        : SimSpec object; auto-loaded from spec.json if None
    ml          : atoms per monolayer; inferred from spec if 0
    cna_records : pre-computed list from load_cna_series(), or None to skip CNA
    save        : write PNG files to sim_dir

    Returns
    -------
    dict of {name: matplotlib.figure.Figure}
    """
    _need_mpl()
    sim_dir = Path(sim_dir)

    if spec is None:
        spec_path = sim_dir / 'spec.json'
        if spec_path.exists():
            spec = _load_spec(spec_path)

    ml = ml or (spec.ml if spec else 0)

    nc_path = sim_dir / 'ncarbon.txt'
    ep_path = sim_dir / 'etch_products.txt'
    if not nc_path.exists():
        raise FileNotFoundError(f"ncarbon.txt not found in {sim_dir}")
    if ml <= 0:
        raise ValueError("ml must be > 0 (provide via spec or ml= keyword)")

    nc, is_cyc = _parse_nc(nc_path)
    ep = _parse_ep(ep_path)
    plot_spec = spec if is_cyc else None

    figs = {}

    def _save(fig, name):
        figs[name] = fig
        if save:
            fig.savefig(sim_dir / f'{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── common plots ──────────────────────────────────────────────────────────
    fig = plot_etch(nc, ml, spec=plot_spec)
    if fig:
        _save(fig, 'etch')

    fig = plot_o_uptake(nc, ml, spec=plot_spec)
    if fig:
        _save(fig, 'o_uptake')

    if ep:
        fig = plot_product_grid(ep)
        if fig:
            _save(fig, 'product_grid')

        fig = plot_product_trajectory(ep, ml)
        if fig:
            _save(fig, 'product_trajectory')

    if cna_records:
        fig = plot_amorphous(cna_records, ml)
        if fig:
            _save(fig, 'amorphous')

        fig = plot_amorphous_thickness(cna_records, ml)
        if fig:
            _save(fig, 'amorphous_thickness')

    # ── cycling-only plots ────────────────────────────────────────────────────
    if is_cyc and spec and spec.phases:
        fig = plot_etch_per_cycle(nc, spec, ml)
        if fig:
            _save(fig, 'etch_per_cycle')

        fig = plot_per_phase_yield(nc, spec, ml)
        if fig:
            _save(fig, 'per_phase_yield')

        fig = plot_o_per_cycle(nc, spec, ml)
        if fig:
            _save(fig, 'o_per_cycle')

    return figs
