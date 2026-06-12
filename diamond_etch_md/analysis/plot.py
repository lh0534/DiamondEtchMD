"""
analysis/plot.py — analysis plots for DiamondEtchMD simulations.

Main entry point: make_plots(sim_dir, spec=None)

Produces PNGs in sim_dir:
  etch_trajectory.png    — etched C + O uptake vs ion dose; amorphous C layer
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
_AMORPH_COLOR    = "#7c52c8"   # purple — amorphous C line and matching axis


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
    from ..spec import SimSpec, CyclePhase, IonComponent
    import dataclasses
    data = json.loads(Path(spec_path).read_text())
    phases_data   = data.pop('phases', None)
    ion_mix_data  = data.pop('ion_mix', None)
    data['phases']  = [CyclePhase(**p)    for p in phases_data]  if phases_data  else None
    data['ion_mix'] = [IonComponent(**c)  for c in ion_mix_data] if ion_mix_data else None
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
    """Fill background with per-phase colour bands; return one legend patch per phase."""
    if spec is None or spec.phases is None:
        return []
    n_phases = len(spec.phases)
    cmap = plt.get_cmap('tab10')
    phase_colors = [cmap(i % 10) for i in range(n_phases)]
    for fi, (x0, x1, _) in enumerate(_phase_spans(spec)):
        ax.axvspan(x0, x1, color=phase_colors[fi % n_phases],
                   alpha=0.12, lw=0, zorder=0)
    return [mpatches.Patch(color=phase_colors[i], alpha=0.6,
                           label=f"Phase {i + 1}")
            for i in range(n_phases)]


def _cycle_xlim(max_x_ml, spec):
    """Return (0, right) snapped to an integer number of cycles."""
    if spec and spec.phases:
        cycle_ml = sum(p.fluence_ml for p in spec.phases)
        import math
        n_complete = math.ceil(max_x_ml / cycle_ml)
        return 0, max(1, n_complete) * cycle_ml
    return 0, None


# ── spec summary and ML-density helpers ──────────────────────────────────────

_SURFACE_LABEL: Dict[tuple, str] = {
    ("100", "1x1"):           "C(100)",
    ("100", "2x1"):           "C(100) 2×1",
    ("100", "2x1_O"):         "O-C(100) 2×1",
    ("100", "O_ether"):       "Ether-C(100)",
    ("111", "1x1"):           "C(111)",
    ("111", "1x1_O"):         "O-C(111)",
    ("111", "2x1_single"):    "C(111) 2×1",
    ("111", "2x1_single_O"):  "O-C(111) 2×1",
    ("111", "2x1_pandey"):    "C(111) Pandey",
    ("111", "2x1_pandey_O"):  "O-C(111) Pandey",
    ("113", "1x1"):           "C(113)",
    ("113", "1x1_O"):         "O-C(113)",
}


def _surface_label(orientation: str, surface: str) -> str:
    return _SURFACE_LABEL.get((orientation, surface),
                               f"C({orientation}) {surface}")


def _spec_summary_str(spec) -> str:
    """Build a multi-line summary string for figure suptitles.

    Format (all sim types):
        Line 1:  "{Surface} {SimType} Simulation"
        Lines 2+: ion/phase specs
    Returns a single string with embedded newlines.
    """
    if spec is None:
        return ""

    surf = _surface_label(getattr(spec, 'orientation', '100'),
                          getattr(spec, 'surface', '1x1'))

    def _ion_label(sp):
        return _species_ion_label(sp)

    def _ion_line(species, energy, fluence_ml=None, pct=None, flux_ratio=0,
                  radical_energy=0.2, angle=0.0, prefix=""):
        """Build one ion-spec line."""
        parts = [prefix + _ion_label(species)]
        if pct is not None:
            parts.append(f"{pct:.0f}%")
        if energy is not None:
            parts.append(f"{energy:.4g} eV")
        if fluence_ml is not None:
            parts.append(f"{fluence_ml} ML")
        if angle:
            parts.append(f"{angle:.4g}°")
        line = "  ".join(parts)
        if flux_ratio and flux_ratio > 0:
            re_s = f", {radical_energy:.4g} eV O$^\\bullet$" if radical_energy else ""
            line += f"  $J_{{rad^\\bullet}}/J_{{ion^+}}$={flux_ratio}{re_s}"
        return line

    angle = getattr(spec, 'angle', 0.0)

    # ── Multi-ion ──────────────────────────────────────────────────────────────
    if getattr(spec, 'ion_mix', None) is not None:
        fr = getattr(spec, 'flux_ratio', 0)
        sim_type = "Multi-Ion RIE" if fr and fr > 0 else "Multi-Ion"
        total = sum(c.fraction for c in spec.ion_mix)
        fluence = getattr(spec, 'fluence', None)
        lines = [f"{surf} {sim_type} Simulation"]
        # One line per ion component — no RIE prefix, no flux ratio
        for c in spec.ion_mix:
            pct = c.fraction / total * 100
            lines.append(f"{_ion_label(c.species)}  {pct:.0f}%  {c.energy:.4g} eV")
        # Shared RIE / run parameters on a final row
        rie_parts = []
        if fr and fr > 0:
            re_s = f", {getattr(spec, 'radical_energy', 0.2):.4g} eV O$^\\bullet$"
            rie_parts.append(f"$J_{{rad^\\bullet}}/J_{{ion^+}}$={fr}{re_s}")
        if fluence:
            rie_parts.append(f"{fluence} ML total")
        if angle:
            rie_parts.append(f"{angle:.4g}°")
        if rie_parts:
            lines.append("  ".join(rie_parts))
        return "\n".join(lines)

    # ── Cycle ──────────────────────────────────────────────────────────────────
    if spec.phases is not None:
        lines = [f"{surf} Cycle Simulation"]
        for i, p in enumerate(spec.phases):
            rie = p.flux_ratio and p.flux_ratio > 0
            lines.append(
                f"Phase {i + 1}:  " + _ion_line(
                    p.species, p.energy, fluence_ml=p.fluence_ml,
                    flux_ratio=p.flux_ratio,
                    radical_energy=getattr(p, 'radical_energy', 0.2),
                    angle=angle,
                    prefix="RIE " if rie else "",
                )
            )
        return "\n".join(lines)

    # ── Single ion ─────────────────────────────────────────────────────────────
    fr = getattr(spec, 'flux_ratio', 0)
    sim_type = "RIE" if fr and fr > 0 else "Ion Etch"
    fluence = getattr(spec, 'fluence', None)
    ion_ln = _ion_line(spec.species, getattr(spec, 'energy', None),
                       fluence_ml=fluence,
                       flux_ratio=fr,
                       radical_energy=getattr(spec, 'radical_energy', 0.2),
                       angle=angle,
                       prefix="")
    return f"{surf} {sim_type} Simulation\n{ion_ln}"


def _parse_lammps_data_box(path):
    """Return (xlo, xhi, ylo, yhi) in Å from a LAMMPS data file header."""
    xlo = xhi = ylo = yhi = None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[2] == 'xlo' and parts[3] == 'xhi':
                xlo, xhi = float(parts[0]), float(parts[1])
            elif len(parts) >= 4 and parts[2] == 'ylo' and parts[3] == 'yhi':
                ylo, yhi = float(parts[0]), float(parts[1])
            if xlo is not None and ylo is not None:
                break
    if None in (xlo, xhi, ylo, yhi):
        return None
    return xlo, xhi, ylo, yhi


def _ml_density_label(sim_dir, ml) -> str:
    """'1 ML ≡ X.XX × 10^N #/cm²' from box dims in the *0K.dat* data file."""
    sim_dir = Path(sim_dir)
    candidates = (list(sim_dir.glob('*0K.dat*')) +
                  list(sim_dir.glob('*0K.data')))
    if not candidates:
        return ""
    bounds = _parse_lammps_data_box(candidates[0])
    if bounds is None:
        return ""
    xlo, xhi, ylo, yhi = bounds
    area_cm2 = (xhi - xlo) * (yhi - ylo) * 1e-16   # Å² → cm²
    if area_cm2 <= 0:
        return ""
    dens = ml / area_cm2
    exp = int(np.floor(np.log10(max(dens, 1e-30))))
    mantissa = dens / 10**exp
    return f"1 ML ≡ {mantissa:.2f} × 10$^{{{exp}}}$ #/cm²"


def _amorphous_thickness_from_atoms(z_c, zlo, zhi, area_A2):
    """Amorphous layer thickness in Å from z-positions of C atoms.

    Uses a 0.5 Å z-density histogram smoothed with a 5-bin box-car filter.
    Bulk density is estimated from the 15–50% height range of the slab.
    Thickness = distance between the 10% and 90% bulk-density crossings,
    scanning from the top of the slab downward.
    """
    if len(z_c) < 10 or area_A2 <= 0:
        return 0.0
    slab_h = zhi - zlo
    if slab_h <= 0:
        return 0.0
    bin_w  = 0.5
    n_bins = max(int(slab_h / bin_w), 1)
    counts, edges = np.histogram(z_c, bins=n_bins, range=(zlo, zhi))
    z_ctr   = (edges[:-1] + edges[1:]) / 2.0
    density = counts / (bin_w * area_A2)

    # 5-bin box-car smooth (≈ 2.5 Å)
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(density.astype(float), kernel, mode='same')

    # Bulk density: median of density in 15–50% of slab height from bottom
    z_norm = (z_ctr - zlo) / slab_h
    bulk_mask = (z_norm > 0.15) & (z_norm < 0.50)
    if not bulk_mask.any():
        return 0.0
    bulk_dens = np.median(smooth[bulk_mask])
    if bulk_dens <= 0:
        return 0.0
    norm = smooth / bulk_dens

    # Scan from top downward: z_10 = first point ≥ 10%; z_90 = first point ≥ 90%
    z_10 = z_90 = None
    for i in range(len(z_ctr) - 1, -1, -1):
        if z_10 is None and norm[i] >= 0.10:
            z_10 = z_ctr[i]
        if z_10 is not None and z_90 is None and norm[i] >= 0.90:
            z_90 = z_ctr[i]
            break
    if z_10 is None or z_90 is None or z_10 <= z_90:
        return 0.0
    return float(z_10 - z_90)


def _parse_ml_dump(dump_path):
    """Parse ML_impacts.dump; return [(dose_ml, amorphous_thickness_A), ...].

    Each snapshot in the dump corresponds to one ML of fluence (written by
    head.lmp via 'if $(v_c%v_ML)==0 then write_dump').  Box area is read from
    each frame's own box bounds so it tracks box extensions over time.
    """
    results = []
    snap = 0
    with open(dump_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if 'ITEM: TIMESTEP' not in line:
                continue
            f.readline()                           # timestep value
            f.readline()                           # ITEM: NUMBER OF ATOMS
            n_atoms = int(f.readline().strip())
            f.readline()                           # ITEM: BOX BOUNDS ...
            xlo, xhi = map(float, f.readline().split()[:2])
            ylo, yhi = map(float, f.readline().split()[:2])
            zlo, zhi = map(float, f.readline().split()[:2])
            area_A2  = (xhi - xlo) * (yhi - ylo)
            header   = f.readline().split()        # ITEM: ATOMS id type ...
            cols     = header[2:]
            type_col = cols.index('type') if 'type' in cols else 1
            z_col    = cols.index('z')    if 'z'    in cols else 5
            z_c = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                if len(parts) > max(type_col, z_col):
                    if int(parts[type_col]) == 1:
                        z_c.append(float(parts[z_col]))
            thickness = _amorphous_thickness_from_atoms(
                np.array(z_c), zlo, zhi, area_A2
            )
            results.append((snap, thickness))
            snap += 1
    return results


# ── individual plot functions ─────────────────────────────────────────────────

def plot_etch(nc_records, ml, spec=None, ep_records=None,
             cna_records=None, amorphous_rho=None,
             lat_a=None, spec_summary=None, ml_density_str=None, ax=None):
    """Etched C (ML) and O uptake (ML) vs ion dose, with amorphous C layer.

    Amorphous C is shown as a step function on a twin right y-axis (blue, C0),
    labelled in Å but scaled proportionally to the left ML axis so that every
    tick on the right corresponds to the same number of MLs as the left.
    Both y-axes share the same upper limit (in ML-equivalent units), expanded
    automatically if the amorphous C exceeds the etch/uptake range.

    Right y-axis label, ticks, and spine are all colored 'C0' to match the line.
    cna_records takes priority over amorphous_rho (never both).

    Parameters
    ----------
    amorphous_rho   : list of (dose_ml, thickness_A) from _parse_ml_dump
    spec_summary    : string placed above the figure as suptitle
    ml_density_str  : '1 ML = … #/cm²' string placed below the figure
    """
    _need_mpl()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Decide amorphous source (never both)
    use_cna = bool(cna_records)
    use_rho = bool(amorphous_rho) and not use_cna

    # O uptake from ncarbon (ion-impact rows only; skip impact=0 initial-structure entry)
    ion_recs = [r for r in nc_records if r['cn'] == 0 and r['impact'] > 0]
    nc_x = np.array([r['impact'] for r in ion_recs]) / ml
    o_ml = np.array([r['n_oxygen'] / ml for r in ion_recs])

    # Etched C from etch products
    carbon_ep = [r for r in (ep_records or []) if r['n_C'] > 0]
    if carbon_ep:
        nc_max  = max((r['impact'] for r in nc_records), default=0) if nc_records else 0
        max_imp = max(max(r['impact'] for r in carbon_ep), nc_max)
        cumC    = np.zeros(max_imp + 1)
        for r in carbon_ep:
            cumC[r['impact']] += r['n_C']
        depth_y = np.cumsum(cumC) / ml
        depth_x = np.arange(len(depth_y)) / ml
    else:
        depth_x = nc_x
        depth_y = np.zeros(len(nc_x))

    phase_patches = _add_phase_shading(ax, spec)

    ax.step(depth_x, depth_y, where='post', lw=3, alpha=0.7, color='k', label='Etched C')
    ax.plot(nc_x, o_ml, lw=3, alpha=0.7, color=_SPECIES_COLOR["O"],
            label='O uptake')

    # Etch yield linear fit with block-average error — non-cycling sims only
    is_cycling = bool(spec and spec.phases)
    if not is_cycling and len(depth_x) > 5 and float(depth_y[-1]) > 0:
        skip = max(1, len(depth_x) // 10)       # skip first ~10% (transient)
        xf, yf = depth_x[skip:], depth_y[skip:]
        if len(xf) > 4 and xf[-1] > xf[0]:
            slope, intercept = np.polyfit(xf, yf, 1)

            # Block-average SEM: split into N blocks, polyfit each, take std/sqrt(N)
            n_blocks = max(5, min(20, len(xf) // 5))
            bs = len(xf) // n_blocks
            bslopes = [np.polyfit(xf[b*bs:(b+1)*bs], yf[b*bs:(b+1)*bs], 1)[0]
                       for b in range(n_blocks) if (b+1)*bs <= len(xf)]
            sem = (np.std(bslopes, ddof=1) / np.sqrt(len(bslopes))
                   if len(bslopes) > 1 else 0.0)

            # Format as (m ± e) × 10^p C/ion
            p   = int(np.floor(np.log10(max(abs(slope), 1e-30))))
            m   = slope / 10**p
            e   = sem   / 10**p
            lbl = (f"Yield = ({m:.2f} ± {e:.2f})×10$^{{{p}}}$ C/ion"
                   if sem > 0 else
                   f"Yield = {m:.2f}×10$^{{{p}}}$ C/ion")
            ax.plot(depth_x, slope * depth_x + intercept,
                    ls='--', lw=1.5, color='0.45', alpha=0.8, label=lbl)

    ax.set_xlabel('Ion dose (ML)')
    ax.set_ylabel(f"Monolayers (MLs)\n{ml_density_str}" if ml_density_str else "Monolayers (MLs)",
                  labelpad=15)
    max_x = max(
        depth_x[-1] if len(depth_x) else 0,
        nc_x[-1]    if len(nc_x)    else 0,
    )
    xleft, xright = _cycle_xlim(max_x, spec)
    ax.set_xlim(xleft, xright)
    ax.yaxis.grid(True, color='0.88', lw=0.5, zorder=0)

    # Amorphous C — twin right y-axis in Å, scaled proportionally to left ML axis
    am_handle = None
    am_y_A = None   # always in Å so we can set the right ylim proportionally
    if use_rho or use_cna:
        ax2 = ax.twinx()

        if use_rho:
            am_x   = np.array([d for d, _ in amorphous_rho], dtype=float)
            am_y_A = np.array([t for _, t in amorphous_rho], dtype=float)  # Å
            am_label  = r'Amorphous C ($\rho$)'
            am_ylabel = 'Amorphous layer thickness (Å)'
        else:
            # CNA data is in ML; store as Å if lat_a known, else keep as ML
            am_x  = np.array([r['impact'] for r in cna_records]) / ml
            am_ml = np.array([r['n_amorphous'] / ml for r in cna_records])
            am_y_A    = am_ml * (lat_a / 4) if lat_a else am_ml
            am_label  = 'Amorphous C (CNA)'
            am_ylabel = 'Amorphous layer thickness (Å)' if lat_a else 'Amorphous C (ML)'

        am_line, = ax2.plot(am_x, am_y_A, ls='--', lw=3, color='C0',
                            label=am_label, zorder=3)
        am_handle = am_line
        ax2.set_ylabel(am_ylabel, color='C0')
        ax2.tick_params(axis='y', labelcolor='C0', color='C0')
        ax2.yaxis.label.set_color('C0')
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color('C0')
        if xright is not None:
            ax2.set_xlim(xleft, xright)

    # Compute a shared ML top that accommodates every line
    left_top = max(
        float(depth_y[-1]) if len(depth_y) else 0.0,
        float(np.max(o_ml)) if len(o_ml)    else 0.0,
    )
    if am_y_A is not None and len(am_y_A):
        am_top_ml = float(np.max(am_y_A)) / (lat_a / 4) if lat_a else float(np.max(am_y_A))
        ylim_top_ml = max(left_top, am_top_ml) * 1.1
    else:
        ylim_top_ml = left_top * 1.1
    ylim_top_ml = max(ylim_top_ml, 0.1)

    ax.set_ylim(0, ylim_top_ml)

    if use_rho or use_cna:
        # Lock right axis: same ML scale, labelled in Å
        ax2.set_ylim(0, ylim_top_ml * (lat_a / 4) if lat_a else ylim_top_ml)
    elif lat_a:
        # No amorphous data — show Å depth on right axis
        ax.spines['right'].set_visible(True)
        scale = lat_a / 4
        sec = ax.secondary_yaxis(
            'right',
            functions=(lambda y, s=scale: y * s, lambda a, s=scale: a / s),
        )
        sec.set_ylabel('Etch depth (Å)')

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    extra_handles = [p for p in phase_patches]
    extra_labels  = [p.get_label() for p in phase_patches]
    if am_handle is not None:
        extra_handles.append(am_handle)
        extra_labels.append(am_handle.get_label())
    all_handles = h1 + extra_handles
    all_labels  = l1 + extra_labels
    ax.legend(handles=all_handles, labels=all_labels,
              frameon=False, fontsize=11,
              loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncols=max(len(all_handles), 1),
              handlelength=1.0, handletextpad=0.4,
              columnspacing=0.8, labelspacing=0.2,
              borderaxespad=0.0)

    if own:
        if spec_summary:
            n_spec = spec_summary.count('\n') + 1
            plt.tight_layout()
            # Reserve space above axes: ~5% per spec line
            current_top = fig.subplotpars.top
            plt.subplots_adjust(top=max(0.40, current_top - 0.05 * n_spec))
            # First line bold, remaining lines normal weight
            spec_lines  = spec_summary.split('\n')
            line_h      = 0.045   # ~fontsize-11 line height in figure coords
            y_top       = 0.96
            fig.text(0.5, y_top, spec_lines[0],
                     ha='center', va='top', fontsize=11, fontweight='bold',
                     transform=fig.transFigure)
            if len(spec_lines) > 1:
                fig.text(0.5, y_top - line_h, '\n'.join(spec_lines[1:]),
                         ha='center', va='top', fontsize=11, fontweight='normal',
                         transform=fig.transFigure)
        else:
            plt.tight_layout()
        return ax.figure
    return ax


def _apply_suptitle(fig, spec_summary, plot_title=None):
    """Bold first spec line + normal rest + optional plot_title as final line."""
    lines = (spec_summary.split('\n') if spec_summary else [])
    if plot_title:
        lines.append(plot_title)
    if not lines:
        plt.tight_layout()
        return
    n = len(lines)
    plt.tight_layout()
    current_top = fig.subplotpars.top
    plt.subplots_adjust(top=max(0.40, current_top - 0.05 * n))
    line_h = 0.045
    y_top  = 0.96
    fig.text(0.5, y_top, lines[0], ha='center', va='top', fontsize=11,
             fontweight='bold', transform=fig.transFigure)
    if len(lines) > 1:
        fig.text(0.5, y_top - line_h, '\n'.join(lines[1:]), ha='center', va='top',
                 fontsize=11, fontweight='normal', transform=fig.transFigure)


def _bubble_panel(ax, records, color, max_c=6, max_o=6,
                  records2=None, color2=None, label=None, label2=None):
    """Draw bubble-chart panel on ax.

    If records2/color2 provided, overlays a second dataset (hollow circles)
    on the same grid — used to compare ion vs radical products on one panel.
    """
    all_recs = list(records or []) + list(records2 or [])
    raw_max_c = max(r['n_C'] for r in all_recs) if all_recs else 0
    raw_max_o = max(r['n_O'] for r in all_recs) if all_recs else 0
    MAX_C = min(max(raw_max_c, 4), max_c) if max_c is not None else max(raw_max_c, 4)
    MAX_O = min(max(raw_max_o, 3), max_o) if max_o is not None else max(raw_max_o, 3)

    dual = bool(records2)

    def _build_grid(recs):
        g = np.zeros((MAX_O + 1, MAX_C + 1))
        for r in (recs or []):
            if r['n_C'] <= MAX_C and r['n_O'] <= MAX_O:
                g[r['n_O'], r['n_C']] += 1
        return g

    grid1 = _build_grid(records)
    grid2 = _build_grid(records2) if dual else None

    total1 = grid1.sum()
    total2 = grid2.sum() if dual else 0

    # Scale bubble sizes relative to whichever dataset has the larger mode
    combined_max = max(
        (grid1 / total1).max() if total1 > 0 else 0,
        (grid2 / total2).max() if (dual and total2 > 0) else 0,
        1e-9,
    )
    scale = (0.45 ** 2) * np.pi / combined_max

    def _draw_bubbles(grid, total, col, hollow=False):
        frac = grid / total if total > 0 else grid
        for nO in range(MAX_O + 1):
            for nC in range(MAX_C + 1):
                f = frac[nO, nC]
                if f == 0:
                    continue
                radius = np.sqrt(f * scale / np.pi)
                if hollow:
                    ax.add_patch(plt.Circle((nC, nO), radius,
                                            fill=False, edgecolor=col,
                                            lw=2.0, alpha=0.85, zorder=3))
                else:
                    ax.add_patch(plt.Circle((nC, nO), radius,
                                            color=col, alpha=0.40, zorder=2))
                if not dual and f >= 0.02:
                    ax.text(nC, nO, f"{_product_label(nC, 0, nO)}\n{f*100:.0f}%",
                            ha='center', va='center', fontsize=11,
                            color='0.15', zorder=4)

    _draw_bubbles(grid1, total1, color, hollow=False)
    if dual:
        _draw_bubbles(grid2, total2, color2, hollow=True)

    # Ellipse + O:C annotation (single-dataset only to keep dual clean)
    for recs, col, corner in (
        [(records, color, (0.03, 0.97))] +
        ([(records2, color2, (0.97, 0.97))] if dual else [])
    ):
        clipped = [r for r in (recs or []) if r['n_C'] <= MAX_C and r['n_O'] <= MAX_O]
        if len(clipped) > 1:
            nC_v = np.array([r['n_C'] for r in clipped])
            nO_v = np.array([r['n_O'] for r in clipped])
            if not dual:
                mean_C, mean_O = nC_v.mean(), nO_v.mean()
                w = max(4 * nC_v.std(), 0.3)
                h = max(4 * nO_v.std(), 0.3)
                for kw in [
                    dict(facecolor=col, edgecolor='none', alpha=0.10, zorder=1),
                    dict(fill=False, edgecolor=col, lw=2.0, ls='--', alpha=0.8, zorder=1),
                ]:
                    ax.add_patch(Ellipse((mean_C, mean_O), width=w, height=h, **kw))
            oc = nO_v.mean() / nC_v.mean() if nC_v.mean() > 0 else 0
            ha = 'left' if corner[0] < 0.5 else 'right'
            ax.text(*corner, f"O:C = {oc:.2f}", transform=ax.transAxes,
                    fontsize=12, ha=ha, va='top', color=col, fontweight='bold',
                    zorder=5, bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                        alpha=0.85, ec=col, lw=0.8))

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

    if dual:
        lbl1 = label  or 'ion'
        lbl2 = label2 or 'radical'
        ax.legend(
            handles=[
                mpatches.Patch(color=color,  alpha=0.6, label=f"{lbl1} ({int(total1):,})"),
                mpatches.Patch(color=color2, alpha=0.6, label=f"{lbl2} ({int(total2):,})"),
            ],
            frameon=False, fontsize=10, loc='lower right',
        )
    return int(total1), int(total2)


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


def plot_product_grid(ep_records, spec=None, ml=0, ax=None,
                      spec_summary=None, plot_title="Carbon Products",
                      max_c=6, max_o=6, panel_title=None):
    """Bubble chart: n_C (x) vs n_O (y) for C-containing ejected clusters.

    For cycling specs, draws one panel per phase side-by-side.
    Otherwise draws a single panel.  Caller is responsible for splitting
    ion vs radical records before calling (pass filtered ep_records).
    """
    _need_mpl()
    ep_c = [r for r in (ep_records or []) if r['n_C'] > 0]
    if not ep_c:
        return None

    is_cyc = spec and spec.phases

    phase_counts = []   # list of (label, n_ion, n_rad) for suptitle n= line

    if is_cyc and ml > 0:
        phases = spec.phases
        n = len(phases)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
        for pi, p in enumerate(phases):
            phase_recs = [r for r in ep_c if _phase_of_impact(r['impact'], spec, ml) == pi]
            color = _phase_color(p.species, pi)
            phase_has_radicals = any(r.get('cn', 0) > 0 for r in phase_recs)
            lbl = f"Ph.{pi+1}"
            if phase_has_radicals:
                ion_recs = [r for r in phase_recs if r.get('cn', 0) == 0]
                rad_recs = [r for r in phase_recs if r.get('cn', 0) > 0]
                n1, n2 = _bubble_panel(axes[0][pi], ion_recs, color,
                                       max_c=max_c, max_o=max_o,
                                       records2=rad_recs, color2='#e87034',
                                       label='ion', label2='radical')
                phase_counts.append(f"{lbl}: ion {n1:,} / rad {n2:,}")
            else:
                n1, _ = _bubble_panel(axes[0][pi], phase_recs, color,
                                      max_c=max_c, max_o=max_o)
                phase_counts.append(f"{lbl}: n={n1:,}")
    else:
        if spec and spec.phases:
            color = _phase_color(spec.phases[0].species)
        elif spec and getattr(spec, 'species', None):
            color = _phase_color(spec.species)
        else:
            color = '#4e9be6'
        fig, axes = plt.subplots(1, 1, figsize=(6, 5), squeeze=False)
        n1, _ = _bubble_panel(axes[0][0], ep_c, color, max_c=max_c, max_o=max_o)
        phase_counts.append(f"n={n1:,}")

    count_str = "  ".join(phase_counts)
    _apply_suptitle(fig, spec_summary, f"{plot_title}  ({count_str})")
    return fig


def plot_product_trajectory(ep_records, ml, spec=None, ax=None, spec_summary=None):
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
        _apply_suptitle(ax.figure, spec_summary, "Carbon Products")
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
            break
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


def plot_per_phase_yield(nc_records, spec, ml, ax=None, spec_summary=None):
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
        _apply_suptitle(ax.figure, spec_summary, "Per Phase Yield")
        return ax.figure
    return ax


def plot_o_per_cycle(nc_records, spec, ml, ax=None, spec_summary=None):
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
                break
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
        _apply_suptitle(ax.figure, spec_summary, "End-phase O Uptake")
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
    is_cyc = bool(spec and spec.phases)
    plot_spec = spec if is_cyc else None

    lat_a = None
    lat_a_path = sim_dir / 'lat_a.txt'
    if lat_a_path.exists():
        try:
            lat_a = float(lat_a_path.read_text().strip())
        except ValueError:
            pass

    # Spec summary for figure titles
    summary = _spec_summary_str(spec)

    # ML surface density label from initial data file
    ml_dens = _ml_density_label(sim_dir, ml)

    # Density-based amorphous C (default when CNA not available)
    amorphous_rho = None
    if not cna_records:
        dump_path = sim_dir / 'ML_impacts.dump'
        if dump_path.exists():
            try:
                amorphous_rho = _parse_ml_dump(dump_path) or None
            except Exception:
                amorphous_rho = None

    figs = {}

    def _save(fig, name):
        figs[name] = fig
        if save:
            fig.savefig(sim_dir / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    ep_carbon = [r for r in ep if r['n_C'] > 0]

    fig = plot_etch(nc, ml, spec=plot_spec, ep_records=ep_carbon,
                   cna_records=cna_records, amorphous_rho=amorphous_rho,
                   lat_a=lat_a, spec_summary=summary, ml_density_str=ml_dens)
    if fig:
        _save(fig, 'etch_trajectory')

    if ep_carbon:
        is_rie = bool(spec and not is_cyc and getattr(spec, 'flux_ratio', 0) > 0)
        has_cn_data = any(r.get('cn', 0) > 0 for r in ep_carbon)

        if is_rie and has_cn_data:
            # Split by radical (cn>0) vs ion (cn==0) impacts
            ep_ion = [r for r in ep_carbon if r.get('cn', 0) == 0]
            ep_rad = [r for r in ep_carbon if r.get('cn', 0) > 0]
            if ep_ion:
                fig = plot_product_grid(ep_ion, spec=spec, ml=ml,
                                        spec_summary=summary,
                                        plot_title="Ion-Phase Carbon Products")
                if fig:
                    _save(fig, 'product_grid_ion')
            if ep_rad:
                fig = plot_product_grid(ep_rad, spec=spec, ml=ml,
                                        spec_summary=summary,
                                        plot_title="Radical-Phase Carbon Products")
                if fig:
                    _save(fig, 'product_grid_radical')
        else:
            fig = plot_product_grid(ep_carbon, spec=spec, ml=ml,
                                    spec_summary=summary)
            if fig:
                _save(fig, 'product_grid')

        fig = plot_product_trajectory(ep_carbon, ml, spec=plot_spec,
                                      spec_summary=summary)
        if fig:
            _save(fig, 'product_trajectory')

    if is_cyc and spec and spec.phases:
        fig = plot_etch_per_cycle(nc, spec, ml, lat_a=lat_a)
        if fig:
            _save(fig, 'etch_per_cycle')

        fig = plot_per_phase_yield(nc, spec, ml, spec_summary=summary)
        if fig:
            _save(fig, 'per_phase_yield')

        fig = plot_o_per_cycle(nc, spec, ml, spec_summary=summary)
        if fig:
            _save(fig, 'o_per_cycle')

    return figs
