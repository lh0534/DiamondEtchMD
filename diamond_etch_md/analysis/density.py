"""
analysis/density.py — C/O depth-density profiles for cycling simulations.

All profiles are read from impact_snaps/{impact}_0.data (one per ML of dose
for the animation; one per phase end for the static grids).  Height is
measured from the lowest C atom of each snapshot so that the frame stays
stationary while fresh diamond is added at the bottom of the box.
Density is normalized to bulk diamond, rho_dia = 8 / a^3.

Outputs (in sim_dir):
  density_phase_ends.png         rows = phase, cols = cycle
  density_vs_fluence.gif / .mp4  profile animated over dose
  phase_ends_render_density.png  [OVITO render | profile] per phase end
                                 (needs renders/phase_ends/ from render_phase_ends)
  render_density_vs_fluence.gif / .mp4
                                 OVITO render | profile | etch trajectory, animated
                                 over dose (all sim types)
                                 (needs renders/ml_frames/ from render_ml_frames)
"""

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from ._common import _phase_of_impact, kde_density

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

C_TYPE, O_TYPE = 1, 3
C_COLOR = 'k'
O_COLOR = '#0096ff'   # OVITO's O atom color in the render preset
X_MAX = 1.1
X_TICKS = np.round(np.arange(0.0, 1.01, 0.2), 1)
RENDER_DIR = Path('renders') / 'phase_ends'
ML_FRAME_DIR = Path('renders') / 'ml_frames'


def _snap_path(sim_dir, impact):
    """Snapshot after `impact`: {impact}_0.data (cycling/RIE: ion, before its
    radicals) or {impact}.data (ion etch); impact 0 is 0.data."""
    snaps = Path(sim_dir) / 'impact_snaps'
    if impact == 0:
        return snaps / '0.data'
    p = snaps / f'{impact}_0.data'
    return p if p.exists() else snaps / f'{impact}.data'


_SNAP_RE = re.compile(r'^(\d+)(?:_0)?\.data$')


def movie_impacts(sim_dir, ml, min_frames=100):
    """Impacts to use as movie frames: one per ML, or finer (down to every
    impact) when that gives fewer than min_frames."""
    snaps = Path(sim_dir) / 'impact_snaps'
    if not snaps.is_dir():
        return []
    have = {int(m.group(1)) for n in os.listdir(snaps) if (m := _SNAP_RE.match(n))}
    if not have:
        return []
    last = max(have)
    stride = max(1, min(ml, last // min_frames))
    return [i for i in range(0, last + 1, stride) if i in have]


def _banner(spec, ml, impact, colors):
    """Three-line (text, color) label for a movie frame."""
    dose = impact / ml
    d = f"{dose:g} ML" if float(dose).is_integer() else f"{dose:.2f} ML"
    if impact == 0:
        return f"{d}\ninitial surface\n", 'k'
    if spec is not None and spec.phases:
        pi = _phase_of_impact(impact, spec, ml)
        cycle_imp = sum(p.fluence_ml for p in spec.phases) * ml
        return (f"{d}\nCycle {(impact - 1) // cycle_imp + 1}\n{_phase_label(spec, pi)}",
                colors[pi])
    return f"{d}\n{_ion_desc(spec) if spec is not None else ''}\n", 'k'


def read_snapshot(path):
    """Return (area_A2, z, types) from a LAMMPS data file (atom style charge)."""
    with open(path) as f:
        lines = f.readlines()
    n_atoms = xlo = xhi = ylo = yhi = None
    atoms_at = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2 and parts[1] == 'atoms':
            n_atoms = int(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ['xlo', 'xhi']:
            xlo, xhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[2:4] == ['ylo', 'yhi']:
            ylo, yhi = float(parts[0]), float(parts[1])
        elif parts and parts[0] == 'Atoms':
            atoms_at = i + 2
            break
    rows = np.array([lines[j].split()[:6] for j in range(atoms_at, atoms_at + n_atoms)],
                    dtype=float)
    return (xhi - xlo) * (yhi - ylo), rows[:, 5], rows[:, 1].astype(int)


def density_profile(z, types, area_A2, lat_a, h_max=None):
    """Return (h, rho_C, rho_O): Gaussian-KDE densities (sigma = a/8, see
    _common.kde_density) normalized to bulk diamond, vs height in Å above the
    lowest C atom."""
    h = z - z[types == C_TYPE].min()
    top = h_max if h_max is not None else h.max() + 1.0
    grid, rc = kde_density(h[types == C_TYPE], area_A2, lat_a, top)
    _, ro = kde_density(h[types == O_TYPE], area_A2, lat_a, top)
    return grid, rc, ro


def _crop_cols(img, pad=3):
    """Column range of the non-white part of a render (the cell outline spans the
    full height, so this is stable from frame to frame)."""
    cols = np.where((np.asarray(img)[..., :3] < 235).any(axis=2).any(axis=0))[0]
    return max(cols.min() - pad, 0), min(cols.max() + pad + 1, img.shape[1])


def phase_end_snapshots(sim_dir, spec, ml):
    """[(cycle, phase_idx, dose_ml, path), ...] for every completed phase end."""
    out = []
    dose = 0
    for cyc in range(spec.cycles):
        for pi, p in enumerate(spec.phases):
            dose += p.fluence_ml
            path = _snap_path(sim_dir, int(round(dose * ml)))
            if path.exists():
                out.append((cyc, pi, dose, path))
    return out


def _ion_desc(obj, pct=True):
    """'Ar 215 eV', 'O 40% + O2 60% 200 eV' (pct=False: 'O/O2 200 eV'), plus
    ', R=2', for a phase or a non-cycling spec; an ion_mix overrides the
    placeholder species/energy."""
    if obj.ion_mix:
        tot = sum(c.fraction for c in obj.ion_mix)
        e = sorted({c.energy for c in obj.ion_mix})
        ions = (" + ".join(f"{c.species} {100 * c.fraction / tot:.0f}%" for c in obj.ion_mix)
                if pct else "/".join(c.species for c in obj.ion_mix))
        s = ions + f" {'/'.join(f'{x:g}' for x in e)} eV"
    else:
        s = f"{obj.species} {obj.energy:g} eV"
    if getattr(obj, 'flux_ratio', 0):
        s += f", R={obj.flux_ratio:g}"
    return s


def _phase_label(spec, pi, sep=": "):
    return f"Phase {pi + 1}{sep}{_ion_desc(spec.phases[pi], pct=False)}"


def _phase_colors(spec):
    from .plot import _phase_index_color
    return [_phase_index_color(i) for i in range(len(spec.phases) if spec and spec.phases else 0)]


def _load_profiles(snaps, lat_a):
    profs = {}
    for cyc, pi, dose, path in snaps:
        area, z, t = read_snapshot(path)
        profs[(cyc, pi)] = (dose, *density_profile(z, t, area, lat_a))
    return profs


def _grid_axes(n_rows, n_cols, width_per_col, height, pair=False, img_aspect=0.5,
               n_img=1):
    """Rows = cycles, cols = phases; y shared within a row only.  With pair=True
    each cell is n_img render columns + a profile column + a spacer."""
    if pair:
        # widths in inches: image sized to its own aspect so aspect='auto' does not distort
        ratios = [img_aspect * 0.8 * height] * n_img + [2.6, 0.5]
        per = n_img + 2
        fig, axes = plt.subplots(
            n_rows, per * n_cols, figsize=(sum(ratios) * n_cols, height * n_rows),
            sharey='row', squeeze=False,
            gridspec_kw={'width_ratios': ratios * n_cols, 'wspace': 0.03})
        for row in axes:
            for c in range(n_cols):
                row[per * c + per - 1].axis('off')
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(width_per_col * n_cols, height * n_rows),
            sharey='row', squeeze=False)
    return fig, axes


def _views(meta, img_dir):
    """[(angle, png_path, h_lo, h_hi, width_A), ...] for every view in a render
    sidecar (older sidecars without "views" hold one 0° view)."""
    img_dir = Path(img_dir)
    if 'views' not in meta:
        return [(0, img_dir / Path(meta['snapshot']).name, meta['h_lo'], meta['h_hi'],
                 meta['width_A'])]
    return [(int(a), img_dir / v['png'], v['h_lo'], v['h_hi'], v['width_A'])
            for a, v in sorted(meta['views'].items(), key=lambda kv: int(kv[0]))]


def _style_img_axes(ax, angle=None):
    ax.set_xticks([])
    for s_ in ('top', 'right', 'bottom'):
        ax.spines[s_].set_visible(False)
    if angle is not None:
        ax.set_xlabel(f"{angle}°", fontsize=10, color='0.35')


def _draw_profile(ax, h, rc, ro):
    ax.plot(rc, h, color=C_COLOR, lw=1.8, label='C')
    ax.plot(ro, h, color=O_COLOR, lw=1.8, label='O')
    ax.axvline(1.0, color='0.75', lw=0.8, ls=':', zorder=0)
    _set_xaxis(ax)


def _set_xaxis(ax):
    ax.set_xlim(0, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([f'{t:.1f}' for t in X_TICKS], fontsize=10)


def _label_grid(fig, axes, spec, done, cols_per_cell, prof_col, y_label):
    """Row labels = cycles, column titles = phases, x label under the lowest
    panel of each phase column, shared y label and C/O legend."""
    colors = _phase_colors(spec)
    n_rows = len(axes)
    for cyc in range(n_rows):
        axes[cyc][0].set_ylabel(f"Cycle {cyc + 1}", fontweight='bold', fontsize=12)
    for pi in range(len(spec.phases)):
        axes[0][pi * cols_per_cell].set_title(_phase_label(spec, pi), loc='left',
                                              color=colors[pi], fontweight='bold',
                                              fontsize=11)
        rows = [c for c, p in done if p == pi]
        if rows:
            axes[max(rows)][pi * cols_per_cell + prof_col].set_xlabel(
                r'$\rho/\rho_{\mathrm{dia}}$')
    left = min(ax.get_position().x0 for ax in fig.axes)
    fig.supylabel(y_label, x=left - 1.0 / fig.get_figwidth(), ha='center')
    handles = [plt.Line2D([], [], color=C_COLOR, lw=1.8, label='C'),
               plt.Line2D([], [], color=O_COLOR, lw=1.8, label='O')]
    bottom = min(ax.get_position().y0 for ax in fig.axes if ax.axison)
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, bottom - 0.55 / fig.get_figheight()),
               frameon=False, ncol=2)


def plot_phase_end_density(sim_dir, spec, ml, lat_a, spec_summary=None):
    """Grid of end-of-phase C/O density profiles (rows = cycle, cols = phase)."""
    snaps = phase_end_snapshots(sim_dir, spec, ml)
    if not snaps:
        return None
    profs = _load_profiles(snaps, lat_a)
    n_rows = max(c for c, _, _, _ in snaps) + 1
    fig, axes = _grid_axes(n_rows, len(spec.phases), 3.2, 3.6)
    for (cyc, pi), (dose, h, rc, ro) in profs.items():
        ax = axes[cyc][pi]
        _draw_profile(ax, h, rc, ro)
        ax.text(0.97, 0.97, f"{dose:g} ML", transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color='0.35')
    for cyc in range(n_rows):
        tops = [profs[(c, p)][1][-1] for c, p in profs if c == cyc]
        axes[cyc][0].set_ylim(0, max(tops) + 1.0)
        for pi in range(len(spec.phases)):
            if (cyc, pi) not in profs:
                axes[cyc][pi].axis('off')
    _label_grid(fig, axes, spec, set(profs), 1, 0, 'Height above lattice bottom (Å)')
    _suptitle(fig, spec_summary, "End-of-Phase Density Profiles")
    return fig


def _suptitle(fig, spec_summary, title):
    """Bold first summary line, then the rest + title, stacked just above the axes."""
    lines = (spec_summary.split('\n') if spec_summary else []) + [title]
    top = max(ax.get_position().y1 for ax in fig.axes)
    body = fig.text(0.5, top + 0.35 / fig.get_figheight(), '\n'.join(lines[1:]), ha='center', va='bottom',
                    fontsize=11)
    if len(lines) > 1:
        bb = body.get_window_extent(renderer=fig.canvas.get_renderer())
        y1 = fig.transFigure.inverted().transform((0, bb.y1))[1]
        fig.text(0.5, y1 + 0.003, lines[0], ha='center', va='bottom',
                 fontsize=11, fontweight='bold')


def _save_movie(anim, sim_dir, out_stem, fps, gif_width, mp4_dpi):
    """Write <stem>.mp4 and <stem>.gif.  With ffmpeg the gif is made from the
    mp4 with a global palette (Pillow's per-frame palette merges pale colors
    such as the phase shading)."""
    import subprocess
    from matplotlib.animation import PillowWriter, FFMpegWriter
    sim_dir = Path(sim_dir)
    gif = sim_dir / f'{out_stem}.gif'
    if not shutil.which('ffmpeg'):
        anim.save(gif, writer=PillowWriter(fps=fps), dpi=80)
        return [gif]
    mp4 = sim_dir / f'{out_stem}.mp4'
    anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=mp4_dpi)
    vf = (f"scale={gif_width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=none")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-vf", vf, str(gif)], check=True)
    return [gif, mp4]


def animate_density(sim_dir, spec, ml, lat_a, fps=10, out_stem='density_vs_fluence',
                    spec_summary=None):
    """Animate the C/O profile over dose (frames from movie_impacts). Returns written paths."""
    from matplotlib.animation import FuncAnimation
    sim_dir = Path(sim_dir)
    frames = []
    for imp in movie_impacts(sim_dir, ml):
        area, z, t = read_snapshot(_snap_path(sim_dir, imp))
        frames.append((imp, area, z, t))
    if len(frames) < 2:
        return []
    h_top = max((z - z[t == C_TYPE].min()).max() for _, _, z, t in frames) + 1.0
    profs = [(k, *density_profile(z, t, a, lat_a, h_max=h_top)) for k, a, z, t in frames]
    colors = _phase_colors(spec)

    fig, ax = plt.subplots(figsize=(5, 6.5))
    fig.subplots_adjust(top=0.82)
    k0, h, rc, ro = profs[0]
    line_c, = ax.plot(rc, h, color=C_COLOR, lw=2, label='C')
    line_o, = ax.plot(ro, h, color=O_COLOR, lw=2, label='O')
    ax.axvline(1.0, color='0.75', lw=0.8, ls=':', zorder=0)
    _set_xaxis(ax)
    ax.set_ylim(0, h_top)
    ax.set_xlabel(r'$\rho/\rho_{\mathrm{dia}}$')
    ax.set_ylabel('Height above lattice bottom (Å)')
    ax.legend(frameon=False, loc='upper right')
    if spec_summary:
        fig.text(0.5, 0.995, spec_summary, ha='center', va='top', fontsize=8)
    banner = ax.set_title('', loc='left', fontsize=11, fontweight='bold')

    def update(i):
        k, h, rc, ro = profs[i]
        line_c.set_xdata(rc)
        line_o.set_xdata(ro)
        text, color = _banner(spec, ml, k, colors)
        banner.set_text(text)
        banner.set_color(color)
        return line_c, line_o, banner

    anim = FuncAnimation(fig, update, frames=len(profs), blit=False)
    written = _save_movie(anim, sim_dir, out_stem, fps, gif_width=450, mp4_dpi=150)
    plt.close(fig)
    return written


def _frame_stem(impact):
    return f"{impact:06d}"


def _trajectory_axes(ax, sim_dir, spec, ml, lat_a, cna_records):
    """Draw the etch_trajectory.png plot into ax (same inputs as make_plots);
    return the Line2D artists to reveal progressively."""
    from .plot import (plot_etch, _parse_nc, _parse_ep, _parse_ml_dump,
                       _ml_density_label)
    sim_dir = Path(sim_dir)
    nc, _ = _parse_nc(sim_dir / 'ncarbon.txt')
    ep = [r for r in _parse_ep(sim_dir / 'etch_products.txt') if r['n_C'] > 0]
    amorphous_rho = None
    if not cna_records and (sim_dir / 'ML_impacts.dump').exists():
        amorphous_rho = _parse_ml_dump(sim_dir / 'ML_impacts.dump', lat_a=lat_a) or None
    before = set(ax.figure.axes)
    plot_etch(nc, ml, spec=spec if spec and spec.phases else None, ep_records=ep,
              cna_records=cna_records, amorphous_rho=amorphous_rho, lat_a=lat_a,
              ml_density_str=_ml_density_label(sim_dir, ml), ax=ax)
    axes = [ax] + [a for a in ax.figure.axes if a not in before]
    return [ln for a in axes for ln in a.get_lines()]


def animate_render_density(sim_dir, spec, ml, lat_a, cna_records=None, fps=10,
                           out_stem='render_density_vs_fluence'):
    """Movie: OVITO side view(s) | C/O profile (shared height axis) |
    etch_trajectory plot revealed up to the current dose.  Needs
    renders/ml_frames/ from render.render_ml_frames.  Returns written paths."""
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    from PIL import Image
    sim_dir = Path(sim_dir)
    frames = []
    for imp in movie_impacts(sim_dir, ml):
        meta = sim_dir / ML_FRAME_DIR / f'{_frame_stem(imp)}.json'
        if not meta.exists():
            continue
        views = _views(json.loads(meta.read_text()), meta.parent)
        if not all(v[1].exists() for v in views):
            continue
        area, z, t = read_snapshot(_snap_path(sim_dir, imp))
        frames.append((imp, [v[1] for v in views], views,
                       density_profile(z, t, area, lat_a)))
    if len(frames) < 2:
        return []
    views0 = frames[0][2]
    h_lo, h_hi = views0[0][2], views0[0][3]
    crops, widths, first = [], [], []
    for angle, png, lo, hi, width in views0:
        a = np.asarray(Image.open(png))
        c0, c1 = _crop_cols(a)
        crops.append((c0, c1))
        widths.append((c1 - c0) / a.shape[1] * width)
        first.append(a[:, c0:c1])
    colors = _phase_colors(spec)
    n_img = len(views0)

    height = 7.0
    img_ws = [w / (h_hi - h_lo) * 0.8 * height for w in widths]
    ratios = img_ws + [3.0, 1.4, 6.5]
    fig, axs = plt.subplots(
        1, n_img + 3, figsize=(sum(ratios) + 1.6, height),
        gridspec_kw={'width_ratios': ratios, 'wspace': 0.03})
    ax_imgs, (ax_p, ax_gap, ax_t) = list(axs[:n_img]), axs[n_img:]
    ax_gap.axis('off')
    for ax in ax_imgs[1:] + [ax_p]:
        ax.sharey(ax_imgs[0])
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.07, right=0.93)

    ims = []
    for ax, a, w, (angle, *_rest) in zip(ax_imgs, first, widths, views0):
        ims.append(ax.imshow(a, extent=[0, w, h_lo, h_hi], aspect='auto'))
        _style_img_axes(ax, angle if n_img > 1 else None)
    ax_imgs[0].set_ylabel('Height above lattice bottom (Å)')
    ax_imgs[0].set_ylim(h_lo, h_hi)
    h, rc, ro = frames[0][3]
    line_c, = ax_p.plot(rc, h, color=C_COLOR, lw=2, label='C')
    line_o, = ax_p.plot(ro, h, color=O_COLOR, lw=2, label='O')
    ax_p.axvline(1.0, color='0.75', lw=0.8, ls=':', zorder=0)
    _set_xaxis(ax_p)
    ax_p.set_xlabel(r'$\rho/\rho_{\mathrm{dia}}$')
    ax_p.legend(frameon=False, loc='upper right')

    reveal = _trajectory_axes(ax_t, sim_dir, spec, ml, lat_a, cna_records)
    x0 = ax_t.get_xlim()[0]
    clip_tf = blended_transform_factory(ax_t.transData, ax_t.transAxes)

    def reveal_to(dose):
        # matplotlib caches clip paths, so a fresh rectangle is set every frame
        clip = Rectangle((x0, 0), dose - x0, 1, transform=clip_tf)
        for ln in reveal:
            ln.set_clip_path(clip)

    reveal_to(frames[0][0] / ml)
    cursor = ax_t.axvline(0, color='0.3', lw=1.0, zorder=5)

    banner = fig.text(0.07, 0.86, '', ha='left', va='bottom', fontsize=12,
                      fontweight='bold')

    def update(i):
        imp, pngs, _, (h, rc, ro) = frames[i]
        for im, png, (c0, c1) in zip(ims, pngs, crops):
            im.set_data(np.asarray(Image.open(png))[:, c0:c1])
        line_c.set_data(rc, h)
        line_o.set_data(ro, h)
        dose = imp / ml
        reveal_to(dose)
        cursor.set_xdata([dose, dose])
        text, color = _banner(spec, ml, imp, colors)
        banner.set_text(text)
        banner.set_color(color)
        return (*ims, line_c, line_o, cursor, banner)

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    written = _save_movie(anim, sim_dir, out_stem, fps, gif_width=1100, mp4_dpi=120)
    plt.close(fig)
    return written


def plot_phase_end_render_density(sim_dir, spec, ml, lat_a, spec_summary=None):
    """Grid of [OVITO side view | density profile] pairs (rows = cycle, cols = phase),
    each row on its own height axis.

    Reads renders/phase_ends/*.png and their .json sidecars (written by
    render.render_phase_ends), which give each image's extent in Å above the
    lattice bottom so atoms line up with the profile.
    """
    from PIL import Image
    sim_dir = Path(sim_dir)
    snaps = phase_end_snapshots(sim_dir, spec, ml)
    cells = []
    for cyc, pi, dose, path in snaps:
        meta = sim_dir / RENDER_DIR / f'{render_stem(cyc, pi, dose)}.json'
        if meta.exists():
            views = _views(json.loads(meta.read_text()), meta.parent)
            if all(v[1].exists() for v in views):
                cells.append((cyc, pi, dose, path, views))
    if not cells:
        return None
    n_rows = max(c[0] for c in cells) + 1
    n_ph = len(spec.phases)
    n_img = max(len(c[-1]) for c in cells)
    per = n_img + 2
    imgs = {}
    for cyc, pi, dose, path, views in cells:
        for vi, (angle, png, lo, hi, width) in enumerate(views):
            a = np.asarray(Image.open(png))
            c0, c1 = _crop_cols(a)
            imgs[(cyc, pi, vi)] = (a[:, c0:c1], (c1 - c0) / a.shape[1] * width, lo, hi)
    img_aspect = max(w / (hi - lo) for _, w, lo, hi in imgs.values())
    fig, axes = _grid_axes(n_rows, n_ph, None, 4.6, pair=True, img_aspect=img_aspect,
                           n_img=n_img)
    for cyc, pi, dose, path, views in cells:
        for vi, (angle, *_rest) in enumerate(views):
            a, w, lo, hi = imgs[(cyc, pi, vi)]
            ax_img = axes[cyc][per * pi + vi]
            ax_img.imshow(a, extent=[0, w, lo, hi], aspect='auto')
            _style_img_axes(ax_img, angle if n_img > 1 else None)
        area, z, t = read_snapshot(path)
        h, rc, ro = density_profile(z, t, area, lat_a)
        ax_p = axes[cyc][per * pi + n_img]
        _draw_profile(ax_p, h, rc, ro)
        ax_p.text(0.97, 0.97, f"{dose:g} ML", transform=ax_p.transAxes,
                  ha='right', va='top', fontsize=10, color='0.35')
    done = {(c[0], c[1]) for c in cells}
    for cyc in range(n_rows):
        row = [v for c, *_, views in cells if c == cyc for v in views]
        axes[cyc][0].set_ylim(min(v[2] for v in row), max(v[3] for v in row))
        for pi in range(n_ph):
            if (cyc, pi) not in done:
                for k in range(n_img + 1):
                    axes[cyc][per * pi + k].axis('off')
    _label_grid(fig, axes, spec, done, per, n_img, 'Height above lattice bottom (Å)')
    _suptitle(fig, spec_summary, "End-of-Phase Structure and Density")
    return fig


def render_stem(cyc, pi, dose):
    return f"c{cyc + 1}_p{pi + 1}_{dose:g}ML"
