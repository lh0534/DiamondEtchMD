"""
analysis/summary.py — block-averaged summary statistics and summary.txt writer.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ncarbon import parse_ncarbon, parse_ncarbon_cycling, is_cycling_format
from .etch_products import parse_etch_products


# ── block averaging ───────────────────────────────────────────────────────────

def block_stats(values, n_blocks: int = 10) -> Tuple[float, float]:
    """Mean and standard error via block averaging.

    Divides `values` into `n_blocks` contiguous blocks, computes each block
    mean, then returns (grand_mean, std_of_block_means / sqrt(n_blocks - 1)).

    Returns (mean, 0.0) when fewer than 2 blocks are possible.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    n = min(n_blocks, len(values))
    if n < 2:
        return float(np.mean(values)), 0.0
    block_means = np.array([b.mean() for b in np.array_split(values, n)])
    return float(block_means.mean()), float(block_means.std(ddof=1) / np.sqrt(n - 1))


# ── log parsing ───────────────────────────────────────────────────────────────

def _wall_time_seconds(sim_dir) -> Optional[float]:
    """Parse total wall time from log.lammps or log*.lammps. Returns None if absent."""
    candidates = sorted(Path(sim_dir).glob('log*.lammps'))
    plain = Path(sim_dir) / 'log.lammps'
    if plain.exists() and plain not in candidates:
        candidates.append(plain)
    for path in candidates:
        m = re.search(r'Total wall time:\s*(\d+):(\d+):(\d+)', path.read_text())
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return float(h * 3600 + mn * 60 + s)
    return None


# ── per-impact sequence helpers ───────────────────────────────────────────────

def _ion_etch_sequence(nc_records: List[Dict]) -> List[float]:
    """Return per-ion-impact C-removed values.

    For single-species (cn always 0), uses successive diffs.
    For cycling, computes diff from state just before the ion (last radical or
    previous ion) to state after ion.
    """
    if not nc_records:
        return []

    is_cycling = any(r['cn'] > 0 for r in nc_records)
    if not is_cycling:
        # single-species: diffs between consecutive records
        carbons = [r['n_carbon'] for r in nc_records]
        return [carbons[i - 1] - carbons[i] for i in range(1, len(carbons))]

    # cycling: track last-seen n_carbon before each ion
    yields = []
    prev_nc = nc_records[0]['n_carbon']
    for r in nc_records:
        if r['cn'] == 0:
            yields.append(float(prev_nc - r['n_carbon']))
        prev_nc = r['n_carbon']
    return yields


def _radical_etch_sequence(nc_records: List[Dict]) -> List[float]:
    """Return per-radical-impact C-removed values (only for cycling records)."""
    if not any(r['cn'] > 0 for r in nc_records):
        return []
    yields = []
    prev_nc = nc_records[0]['n_carbon']
    for r in nc_records:
        if r['cn'] > 0:
            yields.append(float(prev_nc - r['n_carbon']))
        prev_nc = r['n_carbon']
    return yields


# ── main analysis function ────────────────────────────────────────────────────

def analyze_run(
    sim_dir,
    spec=None,
    ml: int = 0,
    n_blocks: int = 10,
    cna_records: Optional[List[Dict]] = None,
) -> Dict:
    """Compute all summary statistics for a completed simulation directory.

    Parameters
    ----------
    sim_dir    : path to simulation directory (contains ncarbon.txt etc.)
    spec       : SimSpec object; used for ml and cycling phase info if provided
    ml         : atoms per monolayer — required if spec is None
    n_blocks   : number of blocks for block averaging
    cna_records: pre-computed output of load_cna_series(), or None to skip CNA stats

    Returns
    -------
    dict of scalar statistics ready for write_summary()
    """
    sim_dir = Path(sim_dir)
    ml = ml or (spec.ml if spec else 0)
    if ml <= 0:
        raise ValueError("ml must be > 0 (provide via spec or ml= keyword)")

    nc_path = sim_dir / 'ncarbon.txt'
    ep_path = sim_dir / 'etch_products.txt'
    if not nc_path.exists():
        raise FileNotFoundError(f"ncarbon.txt not found in {sim_dir}")

    nc = parse_ncarbon(nc_path)
    ep = parse_etch_products(ep_path) if ep_path.exists() else []

    is_cyc = any(r['cn'] > 0 for r in nc)
    n_ion = sum(1 for r in nc if r['cn'] == 0)
    n_radical = sum(1 for r in nc if r['cn'] > 0)
    n0 = nc[0]['n_carbon'] if nc else 0

    stats: Dict = {'is_cycling': is_cyc, 'ml': ml,
                   'n_ion_impacts': n_ion, 'n_radical_impacts': n_radical}

    # ── etch yield ────────────────────────────────────────────────────────────
    ion_yields = _ion_etch_sequence(nc)
    ey_mean, ey_err = block_stats(ion_yields, n_blocks)
    stats['etch_yield_per_ion'] = ey_mean
    stats['etch_yield_per_ion_err'] = ey_err

    if is_cyc:
        rad_yields = _radical_etch_sequence(nc)
        ry_mean, ry_err = block_stats(rad_yields, n_blocks)
        stats['etch_yield_per_radical'] = ry_mean
        stats['etch_yield_per_radical_err'] = ry_err
        total_impacts = n_ion + n_radical
        if total_impacts > 0:
            stats['etch_yield_total'] = (
                (ey_mean * n_ion + ry_mean * n_radical) / total_impacts
            )
        else:
            stats['etch_yield_total'] = 0.0
    else:
        stats['etch_yield_per_radical'] = None
        stats['etch_yield_total'] = ey_mean

    # ── etch depth ────────────────────────────────────────────────────────────
    n_final = nc[-1]['n_carbon'] if nc else n0
    stats['etch_depth_ml'] = (n0 - n_final) / ml

    # ── oxygen uptake ─────────────────────────────────────────────────────────
    # Report mean O content across all records (snapshot after each event)
    o_vals = [r['n_oxygen'] / ml for r in nc]
    o_mean, o_err = block_stats(o_vals, n_blocks)
    stats['o_uptake_ml'] = o_mean
    stats['o_uptake_ml_err'] = o_err

    # ── amorphous carbon (CNA) ────────────────────────────────────────────────
    if cna_records:
        am_vals = [r['n_amorphous'] / ml for r in cna_records]
        am_mean, am_err = block_stats(am_vals, n_blocks)
        stats['amorphous_ml'] = am_mean
        stats['amorphous_ml_err'] = am_err

        th_vals = [r['amorphous_thickness_A'] for r in cna_records]
        th_mean, th_err = block_stats(th_vals, n_blocks)
        stats['amorphous_thickness_A'] = th_mean
        stats['amorphous_thickness_A_err'] = th_err
    else:
        stats['amorphous_ml'] = None
        stats['amorphous_ml_err'] = None
        stats['amorphous_thickness_A'] = None
        stats['amorphous_thickness_A_err'] = None

    # ── product composition ───────────────────────────────────────────────────
    if ep:
        oc_ratios = [r['n_O'] / r['n_C'] if r['n_C'] > 0 else 0.0 for r in ep]
        stats['avg_oc_ratio'] = float(np.mean(oc_ratios))
        stats['avg_o_per_product'] = float(np.mean([r['n_O'] for r in ep]))
        stats['avg_c_per_product'] = float(np.mean([r['n_C'] for r in ep]))
    else:
        stats['avg_oc_ratio'] = 0.0
        stats['avg_o_per_product'] = 0.0
        stats['avg_c_per_product'] = 0.0

    # ── cycling per-phase stats ───────────────────────────────────────────────
    if is_cyc and spec is not None and spec.phases is not None:
        cyc_recs = parse_ncarbon_cycling(nc_path, spec=spec)
        phase_stats = {}
        for rec in cyc_recs:
            key = rec['phase_idx']
            if key not in phase_stats:
                phase_stats[key] = {
                    'name': rec['phase_name'],
                    'ion_etch': [],
                    'radical_etch': [],
                    'o_content': [],
                }
            phase_stats[key]['ion_etch'].append(float(rec['ion_etch']))
            phase_stats[key]['radical_etch'].append(float(rec['radical_etch']))
            phase_stats[key]['o_content'].append(rec['n_oxygen'] / ml)

        stats['per_phase'] = {
            pi: {
                'name':              ps['name'],
                'etch_yield_mean':   float(np.mean(ps['ion_etch'])) if ps['ion_etch'] else 0.0,
                'etch_yield_err':    block_stats(ps['ion_etch'], n_blocks)[1],
                'radical_etch_mean': float(np.mean(ps['radical_etch'])) if ps['radical_etch'] else 0.0,
                'o_content_mean':    float(np.mean(ps['o_content'])) if ps['o_content'] else 0.0,
            }
            for pi, ps in phase_stats.items()
        }
    else:
        stats['per_phase'] = {}

    # ── throughput ────────────────────────────────────────────────────────────
    wall_s = _wall_time_seconds(sim_dir)
    if wall_s and wall_s > 0 and n_ion > 0:
        stats['impacts_per_day_ml'] = (n_ion / ml) * 86400 / wall_s
    else:
        stats['impacts_per_day_ml'] = None

    return stats


# ── summary.txt writer ────────────────────────────────────────────────────────

def write_summary(stats: Dict, path) -> None:
    """Write a human-readable summary.txt to `path`."""

    def _fmt(val, err=None, unit=''):
        if val is None:
            return '  N/A'
        s = f'  {val:.4g}'
        if err is not None and err > 0:
            s += f' ± {err:.2g}'
        if unit:
            s += f'  {unit}'
        return s

    is_cyc = stats.get('is_cycling', False)
    lines = ['# DiamondEtchMD Analysis Summary', '']

    lines += [
        '## Etch statistics',
        f"Etch yield (C/ion):{_fmt(stats.get('etch_yield_per_ion'), stats.get('etch_yield_per_ion_err'))}",
    ]
    if is_cyc:
        lines.append(
            f"  (total):{_fmt(stats.get('etch_yield_total'))}"
        )
        if stats.get('etch_yield_per_radical') is not None:
            lines.append(
                f"  (radical):{_fmt(stats.get('etch_yield_per_radical'), stats.get('etch_yield_per_radical_err'))}"
            )
    lines.append(f"Etch depth (ML):{_fmt(stats.get('etch_depth_ml'))}")

    if stats.get('per_phase'):
        lines.append('\nPer-phase etch yield (cycle-averaged):')
        for pi, ps in sorted(stats['per_phase'].items()):
            lines.append(
                f"  Phase {pi} ({ps['name']}):  {ps['etch_yield_mean']:.4g} C/ion"
                + (f"  radical: {ps['radical_etch_mean']:.4g} C/radical" if ps['radical_etch_mean'] else '')
            )

    lines += [
        '',
        '## Surface oxygen',
        f"O uptake (ML):{_fmt(stats.get('o_uptake_ml'), stats.get('o_uptake_ml_err'))}",
        '',
        '## Amorphous carbon (CNA)',
        f"Amorphous C (ML):{_fmt(stats.get('amorphous_ml'), stats.get('amorphous_ml_err'))}",
        f"Layer thickness (Å):{_fmt(stats.get('amorphous_thickness_A'), stats.get('amorphous_thickness_A_err'))}",
        '',
        '## Product composition',
        f"Avg O:C ratio:{_fmt(stats.get('avg_oc_ratio'))}",
        f"Avg O per product:{_fmt(stats.get('avg_o_per_product'))}",
        f"Avg C per product:{_fmt(stats.get('avg_c_per_product'))}",
        '',
        '## Throughput',
    ]

    tpd = stats.get('impacts_per_day_ml')
    if tpd is not None:
        lines.append(f'Ion impacts per day:{_fmt(tpd, unit="ML/day")}')
    else:
        lines.append('Ion impacts per day:  N/A  (log.lammps not found)')
    lines.append(f"Total ion impacts:  {stats.get('n_ion_impacts', 'N/A')}")
    if is_cyc:
        lines.append(f"Total radical impacts:  {stats.get('n_radical_impacts', 'N/A')}")

    Path(path).write_text('\n'.join(lines) + '\n')
