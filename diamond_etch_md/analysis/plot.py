"""
analysis/plot.py — analysis plots for DiamondEtchMD simulations.

Main entry point: make_plots(sim_dir, spec=None)

Produces PNGs in sim_dir:
  etch_trajectory.png    — etched C + O uptake vs ion dose; amorphous C lines when CNA
  product_grid.png       — bubble chart: n_C vs n_O per ejected cluster
  product_trajectory.png — cumulative yield per product species vs dose, phase-shaded
  etch_per_cycle.png     — etch per cycle (cycling only)
  per_phase_yield.png    — per-phase etch yield with error bars (cycling only)
  o_per_cycle.png        — O loading/unloading per cycle (cycling only)

Requires matplotlib and numpy (install with pip install 'DiamondEtchMD[analysis]').
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _need_mpl():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install 'DiamondEtchMD[analysis]'"
        )


# ── phase colours (matches reference notebook) ───────────────────────────────

_SPECIES_COLOR = {
    "Ar": "#4e9be6",
    "O":  "#e87034",
    "O2": "#e87034",
}
_FALLBACK_COLORS = ["#3aaa3a", "#d62728", "#9467bd", "#8c564b"]


def _phase_color(species, fallback_idx=0):
    return _SPECIES_COLOR.get(
        species, _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
    )


def _species_ion_label(sp):
    """Return a LaTeX label like 'Ar$^+$' or 'O$_{2}^+$' for a species string."""
    m = re.match(r'^([A-Za-z]+)(\d*)$', sp)
    if m:
        name, num = m.group(1), m.group(2)
        if num:
            return f'{name}$_{{{num}}}^+$'
        return f'{name}$^+$'
    return sp + '$^+$'


def _phase_spans(spec):
    """Return [(x_start_ML, x_end_ML, species), ...] for every phase in every cycle."""
    spans = []
    x = 0
    for _ in range(spec.cycles):
        for p in spec.phases:
            spans.append((x, x + p.fluence_ml, p.species))
            x += p.fluence_ml
    return spans


# ── shared helpers ────────────────────────────────────────────────────────────

def _product_label(nc, nh, no):
    parts = []
    if nc:
        parts.append('C' + (f'$_{{{nc}}}$' if nc > 1 else ''))
    if nh:
        parts.append('H' + (f'$_{{{nh}}}$' if nh > 1 else ''))
    if no:
        parts.append('O' + (f'$_{{{no}}}$' if no > 1 else ''))
    return ''.join(parts) or '?'


def _parse_nc(path):
    from .ncarbon import parse_ncarbon
    recs = parse_ncarbon(path)
    return recs, any(r['cn'] > 0 for r in recs)


def _parse_ep(path):
    from .etch_products import parse_etch_products
    return parse_etch_products(path) if Path(path).exists() else []


def _load_spec(spec_path):
    from ..spec import SimSpec, CyclePhase
    import dataclasses
    data = json.loads(Path(spec_path).read_text())
    phases_data = data.pop('phases', None)
    data['phases'] = [CyclePhase(**p) for p in phases_data] if phases_data else None
    valid = {f.name for f in dataclasses.fields(SimSpec)}
    data = {k: v for k, v in data.items() if k in valid}
    return SimSpec(**data)


def _apply_style():
    try:
        plt.style.use("tableau-colorblind10")
    except OSError:
        pass
    plt.rcParams.update({
        "font.size":         13,
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _add_phase_shading(ax, spec):
    """Fill background with per-species colour bands; return legend patches."""
    if spec is None or spec.phases is None:
        return []
    seen = {}
    for fi, (x0, x1, sp) in enumerate(_phase_spans(spec)):
        col = _phase_color(sp, fi)
        ax.axvspan(x0, x1, color=col, alpha=0.10, lw=0, zorder=0)
        if sp not in seen:
            seen[sp] = mpatches.Patch(color=col, alpha=0.5,
                                       label=f"{_species_ion_label(sp)} phase")
    return list(seen.values())


def _cycle_xlim(max_x_ml, spec):
    """Return (0, right) snapped to an integer number of cycles.

    Always shows at least one full cycle so the phase structure is visible
    even for in-progress runs.  When multiple cycles are complete, snaps to
    the last completed one.
    """
    if spec and spec.phases:
        cycle_ml = sum(p.fluence_ml for p in spec.phases)
        n_complete = int(max_x_ml // cycle_ml)
        return 0, max(1, n_complete) * cycle_ml
    return 0, None  # None → let matplotlib autoscale the right edge


# ── individual plot functions ─────────────────────────────────────────────────

def plot_etch(nc_records, ml, spec=None, ep_records=None, cna_records=None,
             lat_a=None, ax=None):
    """Etched C (ML) and O uptake (ML) vs ion dose.

    When cna_records are provided, also plots amorphous C (z-density, dashed)
    and amorphous C (CNA OTHER, dash-dot) on the same left axis.
    When lat_a is provided, a secondary right axis shows thickness in Å
    (labeled "Amorphous layer thickness" with CNA, "Etch depth" without).
    """
    _need_mpl()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 5))

    # O uptake from ncarbon (ion-impact rows only)
    ion_recs = [r for r in nc_records if r['cn'] == 0]
    nc_x = np.array([r['impact'] for r in ion_recs]) / ml
    o_ml = np.array([r['n_oxygen'] / ml for r in ion_recs])

    # Etched C: cumulative C from etch products
    carbon_ep = [r for r in (ep_records or []) if r['n_C'] > 0]
    if carbon_ep:
        max_imp = max(r['impact'] for r in carbon_ep)
        cumC    = np.zeros(max_imp + 1)
        for r in carbon_ep:
            cumC[r['impact']] += r['n_C']
        depth_y = np.cumsum(cumC) / ml
        depth_x = np.arange(len(depth_y)) / ml
    else:
        depth_x = nc_x
        depth_y = np.zeros(len(nc_x))

    phase_patches = _add_phase_shading(ax, spec)

    ax.step(depth_x, depth_y, where='post', lw=2, color='k', label='Etched C')
    ax.plot(nc_x, o_ml, lw=1.5, color=_SPECIES_COLOR["O"], alpha=0.85,
            label='O uptake')

    if cna_records:
        cna_x = np.array([r['impact'] for r in cna_records]) / ml

        # Z-density amorphous C count (dashed, blue)
        if all('n_amorphous_zone' in r for r in cna_records):
            zdense_ml = np.array([r['n_amorphous_zone'] / ml for r in cna_records])
            ax.plot(cna_x, zdense_ml, lw=1.5, ls='--', color='C0', alpha=0.85,
                    label='Amorphous layer thickness')

        # CNA OTHER C (dash-dot, purple)
        cna_ml = np.array([r['n_amorphous'] / ml for r in cna_records])
        ax.plot(cna_x, cna_ml, lw=1.5, ls='-.', color='C4', alpha=0.85,
                label='Amorphous C')

    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('ML')
    max_x = max(
        depth_x[-1] if len(depth_x) else 0,
        nc_x[-1]    if len(nc_x)    else 0,
    )
    xleft, xright = _cycle_xlim(max_x, spec)
    ax.set_xlim(xleft, xright)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color='0.88', lw=0.5, zorder=0)

    if lat_a:
        ax.spines['right'].set_visible(True)
        scale = lat_a / 4
        sec = ax.secondary_yaxis(
            'right',
            functions=(lambda y, s=scale: y * s, lambda a, s=scale: a / s),
        )
        sec.set_ylabel('Amorphous layer thickness (Å)')

    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(handles=h1 + phase_patches, frameon=False, fontsize=11, loc='best')

    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def _bubble_panel(ax, records, color, title):
    """Draw one bubble-chart panel on ax for the given C-containing records."""
    MAX_C = max((max(r['n_C'] for r in records) if records else 0), 4)
    MAX_O = max((max(r['n_O'] for r in records) if records else 0), 3)

    grid = np.zeros((MAX_O + 1, MAX_C + 1))
    for r in records:
        if r['n_C'] <= MAX_C and r['n_O'] <= MAX_O:
            grid[r['n_O'], r['n_C']] += 1

    total = grid.sum()
    frac  = grid / total if total > 0 else grid
    max_f = frac.max() if frac.max() > 0 else 1.0
    scale = (0.45 ** 2) * np.pi / max_f

    for nO in range(MAX_O + 1):
        for nC in range(MAX_C + 1):
            f = frac[nO, nC]
            if f == 0:
                continue
            r = np.sqrt(f * scale / np.pi)
            ax.add_patch(plt.Circle((nC, nO), r, color=color, alpha=0.40, zorder=2))
            if f >= 0.02:
                ax.text(nC, nO, f"{_product_label(nC, 0, nO)}\n{f*100:.0f}%",
                        ha='center', va='center', fontsize=11, color='0.15', zorder=3)

    if total > 1:
        nC_v = np.array([r['n_C'] for r in records if r['n_C'] <= MAX_C])
        nO_v = np.array([r['n_O'] for r in records if r['n_O'] <= MAX_O])
        mean_C, mean_O = nC_v.mean(), nO_v.mean()
        w, h = max(4 * nC_v.std(), 0.3), max(4 * nO_v.std(), 0.3)
        for kw in [
            dict(facecolor=color, edgecolor='none', alpha=0.10, zorder=1),
            dict(fill=False, edgecolor=color, lw=2.0, ls='--', alpha=0.8, zorder=1),
        ]:
            ax.add_patch(Ellipse((mean_C, mean_O), width=w, height=h, **kw))
        oc = nO_v.mean() / nC_v.mean() if nC_v.mean() > 0 else 0
        ax.text(0.03, 0.97, f"O:C = {oc:.2f}", transform=ax.transAxes,
                fontsize=12, ha='left', va='top', color=color, fontweight='bold',
                zorder=5, bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                    alpha=0.85, ec=color, lw=0.8))

    for x in np.arange(-0.5, MAX_C + 1):
        ax.axvline(x, color='0.88', lw=0.5, zorder=0)
    for y in np.arange(-0.5, MAX_O + 1):
        ax.axhline(y, color='0.88', lw=0.5, zorder=0)

    ax.set_xlim(-0.5, MAX_C + 0.5)
    ax.set_ylim(-0.5, MAX_O + 0.5)
    ax.set_xticks(range(MAX_C + 1))
    ax.set_yticks(range(MAX_O + 1))
    ax.set_xlabel('C atoms in product')
    ax.set_ylabel('O atoms in product')
    ax.set_aspect('equal')
    ax.set_title(f"{title}  (n = {int(total):,})", fontsize=11)


def _phase_of_impact(impact, spec, ml):
    """Return 0-based phase index for a given impact number."""
    total_cycle = sum(p.fluence_ml for p in spec.phases) * ml
    pos = (impact - 1) % total_cycle
    cum = 0
    for pi, p in enumerate(spec.phases):
        cum += p.fluence_ml * ml
        if pos < cum:
            return pi
    return len(spec.phases) - 1


def plot_product_grid(ep_records, spec=None, ml=0, ax=None):
    """Bubble chart(s): n_C (x) vs n_O (y) for C-containing ejected clusters.

    For cycling simulations (spec.phases present), produces one subplot per
    phase (matching cycle_plots.ipynb style).
    For non-cycling simulations with radical sweeps (cn>0 records exist),
    produces three subplots: All / Ion phase / Radical phase.
    Otherwise produces a single bubble chart.
    """
    _need_mpl()
    ep_c = [r for r in (ep_records or []) if r['n_C'] > 0]
    if not ep_c:
        return None

    is_cyc = spec and spec.phases
    has_radicals = any(r.get('cn', 0) > 0 for r in ep_c)

    if is_cyc and ml > 0:
        # Per-phase panels (cycling)
        phases = spec.phases
        n = len(phases)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
        for pi, p in enumerate(phases):
            phase_recs = [r for r in ep_c if _phase_of_impact(r['impact'], spec, ml) == pi]
            color = _phase_color(p.species, pi)
            _bubble_panel(axes[0][pi], phase_recs, color,
                          f"{_species_ion_label(p.species)} phase")
    elif has_radicals:
        # Ion / Radical / All panels (RIE with radicals)
        ion_recs = [r for r in ep_c if r.get('cn', 0) == 0]
        rad_recs = [r for r in ep_c if r.get('cn', 0) > 0]
        panels = [
            ('All products',    ep_c,     '#4e9be6'),
            ('Ion phase',       ion_recs, '#4e9be6'),
            ('Radical phase',   rad_recs, '#e87034'),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
        for ax_i, (title, recs, color) in zip(axes[0], panels):
            _bubble_panel(ax_i, recs, color, title)
    else:
        # Single panel
        if spec and spec.phases:
            color = _phase_color(spec.phases[0].species)
        elif spec and spec.species:
            color = _phase_color(spec.species)
        else:
            color = '#4e9be6'
        fig, axes = plt.subplots(1, 1, figsize=(6, 5), squeeze=False)
        _bubble_panel(axes[0][0], ep_c, color, 'Carbon-containing products')

    fig.tight_layout()
    return fig


def plot_product_trajectory(ep_records, ml, spec=None, ax=None):
    """Cumulative yield (ML) per product species vs dose (ML), phase-shaded."""
    _need_mpl()
    if not ep_records:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 5))

    phase_patches = _add_phase_shading(ax, spec)

    by_product = defaultdict(list)
    for r in ep_records:
        by_product[(r['n_C'], r['n_H'], r['n_O'])].append(r['impact'])

    products    = sorted(by_product.items(), key=lambda kv: -len(kv[1]))
    max_impact  = max(r['impact'] for r in ep_records)
    x           = np.arange(1, max_impact + 1) / ml

    cmap = plt.get_cmap('tab10')
    for idx, ((nc, nh, no), imp_list) in enumerate(products[:10]):
        counts = np.zeros(max_impact)
        for imp in imp_list:
            if 1 <= imp <= max_impact:
                counts[imp - 1] += 1
        cumulative = np.cumsum(counts) / ml
        ax.plot(x, cumulative, lw=1.5, label=_product_label(nc, nh, no),
                color=cmap(idx % 10))

    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('Cumulative yield (ML)')
    xleft, xright = _cycle_xlim(max_impact / ml, spec)
    ax.set_xlim(xleft, xright)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color='0.88', lw=0.5, zorder=0)

    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(handles=h1 + phase_patches, frameon=False, fontsize=11, loc='best')

    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_amorphous(cna_records, ml, ax=None):
    """Amorphous and sp2 C (ML) vs dose (ML)."""
    _need_mpl()
    if not cna_records:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 5))

    impacts = np.array([r['impact'] for r in cna_records]) / ml
    am_ml   = np.array([r['n_amorphous'] / ml for r in cna_records])

    ax.plot(impacts, am_ml, lw=1.5, color='C0')
    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('Amorphous / sp2 C (ML)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color='0.88', lw=0.5, zorder=0)

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
        fig, ax = plt.subplots(figsize=(8, 5))

    impacts   = np.array([r['impact'] for r in cna_records]) / ml
    thickness = np.array([r['amorphous_thickness_A'] for r in cna_records])

    ax.plot(impacts, thickness, lw=1.5, color='C0',
            label='Amorphous layer thickness')
    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel('Amorphous layer thickness (Å)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color='0.88', lw=0.5, zorder=0)

    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_etch_per_cycle(nc_records, spec, ml, lat_a=None, ax=None):
    """Etch per cycle (ML) vs cycle number — scatter + mean line."""
    _need_mpl()
    if spec is None or spec.phases is None:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))

    total_cycle_ml = sum(p.fluence_ml for p in spec.phases) * ml
    ion_recs = [r for r in nc_records if r['cn'] == 0]
    n0 = nc_records[0]['n_carbon'] if nc_records else 0
    max_impact = max((r['impact'] for r in ion_recs), default=0)

    cycle_etch = []
    prev = n0
    for cyc in range(spec.cycles):
        start = cyc * total_cycle_ml
        end   = (cyc + 1) * total_cycle_ml
        if max_impact < end:
            break  # cycle not yet complete
        recs = [r for r in ion_recs if start < r['impact'] <= end]
        if recs:
            last = recs[-1]['n_carbon']
            cycle_etch.append((prev - last) / ml)
            prev = last
        else:
            cycle_etch.append(0.0)

    if not cycle_etch:
        if own:
            plt.close(fig)
        return None

    xs = np.arange(1, len(cycle_etch) + 1)
    ax.scatter(xs, cycle_etch, color='#4e9be6', s=40, zorder=3)
    ax.plot(xs, cycle_etch, lw=1.0, color='#4e9be6', alpha=0.5, zorder=2)
    mean_etch = np.mean(cycle_etch)
    ax.axhline(mean_etch, ls='--', lw=1.5, color='k', alpha=0.7,
               label=f'Mean: {mean_etch:.3g} ML/cycle')

    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Etch per cycle (ML)')
    ax.set_xlim(0.5, len(cycle_etch) + 0.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color='0.88', lw=0.5, zorder=0)
    ax.legend(frameon=False, fontsize=11, loc='best')

    if lat_a:
        ax.spines['right'].set_visible(True)
        scale = lat_a / 4
        sec = ax.secondary_yaxis(
            'right',
            functions=(lambda y, s=scale: y * s, lambda a, s=scale: a / s),
        )
        sec.set_ylabel('Etch depth (Å)')

    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_per_phase_yield(nc_records, spec, ml, ax=None):
    """Per-phase etch yield: box-and-whisker plot over cycles."""
    _need_mpl()
    if spec is None or spec.phases is None:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 4))

    total_cycle_ml = sum(p.fluence_ml for p in spec.phases) * ml
    phase_yields = defaultdict(list)

    pre_nc = nc_records[0]['n_carbon'] if nc_records else 0
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

    phase_indices = sorted(phase_yields)
    data   = [phase_yields[pi] for pi in phase_indices]
    labels = [spec.phases[pi].species for pi in phase_indices]
    colors = [_phase_color(spec.phases[pi].species, pi) for pi in phase_indices]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='k', lw=2),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker='o', ms=4, alpha=0.5))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Etch yield per impact (C/ion)')
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color='0.88', lw=0.5, zorder=0)

    if own:
        plt.tight_layout()
        return ax.figure
    return ax


def plot_o_per_cycle(nc_records, spec, ml, ax=None):
    """Phase-end O uptake per cycle — same x-axis style as etch_per_cycle."""
    _need_mpl()
    if spec is None or spec.phases is None:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))

    total_cycle_ml = sum(p.fluence_ml for p in spec.phases) * ml
    ion_recs = [r for r in nc_records if r['cn'] == 0]
    max_impact = max((r['impact'] for r in ion_recs), default=0)

    for pi, p in enumerate(spec.phases):
        phase_start_ml = sum(spec.phases[j].fluence_ml for j in range(pi)) * ml
        phase_end_ml   = phase_start_ml + p.fluence_ml * ml
        o_end = []
        for cyc in range(spec.cycles):
            lo = cyc * total_cycle_ml + phase_start_ml
            hi = cyc * total_cycle_ml + phase_end_ml
            if max_impact < hi:
                break  # phase not yet complete
            recs = [r for r in ion_recs if lo < r['impact'] <= hi]
            if recs:
                o_end.append(recs[-1]['n_oxygen'] / ml)
        if o_end:
            xs = np.arange(1, len(o_end) + 1)
            col = _phase_color(p.species, pi)
            ax.scatter(xs, o_end, color=col, s=40, zorder=3)
            ax.plot(xs, o_end, lw=1.0, color=col, alpha=0.5, zorder=2,
                    label=f"After {p.species}$^+$")

    n_cycles = spec.cycles
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Phase-End O Uptake (ML)')
    ax.set_xlim(0.5, n_cycles + 0.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color='0.88', lw=0.5, zorder=0)
    ax.legend(frameon=False, fontsize=11, loc='best')

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
    """
    _need_mpl()
    _apply_style()

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

    nc, _has_radicals_nc = _parse_nc(nc_path)
    ep = _parse_ep(ep_path)
    # is_cyc: True if spec declares multiple phases (phase shading, per-cycle plots).
    # Distinct from _has_radicals_nc (cn>0 rows) which can be true in single-phase RIE.
    is_cyc = bool(spec and spec.phases)
    plot_spec = spec if is_cyc else None

    lat_a = None
    lat_a_path = sim_dir / 'lat_a.txt'
    if lat_a_path.exists():
        try:
            lat_a = float(lat_a_path.read_text().strip())
        except ValueError:
            pass

    figs = {}

    def _save(fig, name):
        figs[name] = fig
        if save:
            fig.savefig(sim_dir / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    ep_carbon = [r for r in ep if r['n_C'] > 0]  # drop O-only clusters globally

    fig = plot_etch(nc, ml, spec=plot_spec, ep_records=ep_carbon,
                   cna_records=cna_records, lat_a=lat_a)
    if fig:
        _save(fig, 'etch_trajectory')

    if ep_carbon:
        fig = plot_product_grid(ep_carbon, spec=spec, ml=ml)
        if fig:
            _save(fig, 'product_grid')

        fig = plot_product_trajectory(ep_carbon, ml, spec=plot_spec)
        if fig:
            _save(fig, 'product_trajectory')

    if is_cyc and spec and spec.phases:
        fig = plot_etch_per_cycle(nc, spec, ml, lat_a=lat_a)
        if fig:
            _save(fig, 'etch_per_cycle')

        fig = plot_per_phase_yield(nc, spec, ml)
        if fig:
            _save(fig, 'per_phase_yield')

        fig = plot_o_per_cycle(nc, spec, ml)
        if fig:
            _save(fig, 'o_per_cycle')

    return figs
