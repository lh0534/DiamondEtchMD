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
    """Return total simulation wall time in seconds.

    Sums actual LAMMPS compute time from numbered log files (log2.lammps, …),
    excluding log_make_surf.lammps which times surface creation only.

    For each log:
      - If 'Total wall time: H:M:S' is present (job completed cleanly), use it.
      - Otherwise sum all 'Loop time of X …' lines (robust to SLURM kills,
        user restarts, and error-gap periods — only counts actual MD compute time).
    """
    import re as _re
    sim_dir = Path(sim_dir)

    # Numbered logs only (log2.lammps, log3.lammps, …) — exclude log_make_surf etc.
    numbered = [p for p in sim_dir.glob('log*.lammps')
                if _re.match(r'log\d+\.lammps$', p.name)]
    plain = sim_dir / 'log.lammps'
    if plain.exists():
        numbered.append(plain)

    if not numbered:
        return None

    total_s = 0.0
    for path in sorted(numbered):
        text = path.read_text()
        m = _re.search(r'Total wall time:\s*(\d+):(\d+):(\d+)', text)
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            total_s += h * 3600 + mn * 60 + s
        else:
            # Sum all 'Loop time of X' entries — counts only actual MD compute,
            # immune to timestamps and restart gaps.
            total_s += sum(float(lm.group(1))
                           for lm in _re.finditer(r'^Loop time of (\S+)', text,
                                                   _re.MULTILINE))

    return total_s if total_s > 0 else None


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
        # single-species: diffs between consecutive records.
        # The first record (impact=0) is the make_surf baseline; use it only
        # when its n_carbon is within one ML of the first simulation record,
        # i.e. when they are from the same box.  If the box changed between
        # runs (legacy data) the first diff would be garbage, so skip it.
        sim_recs = [r for r in nc_records if r['impact'] > 0]
        if not sim_recs:
            return []
        make_surf = next((r for r in nc_records if r['impact'] == 0), None)
        if make_surf is not None:
            delta = abs(make_surf['n_carbon'] - sim_recs[0]['n_carbon'])
            ml_est = max(1, sim_recs[0]['n_carbon'] // 30)  # rough atoms-per-ML guess
            if delta <= ml_est:
                # Compatible baseline — include it
                carbons = [make_surf['n_carbon']] + [r['n_carbon'] for r in sim_recs]
            else:
                # Incompatible baseline (different box/run) — skip make_surf row
                carbons = [r['n_carbon'] for r in sim_recs]
        else:
            carbons = [r['n_carbon'] for r in sim_recs]
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


def _phase_of_impact(impact: int, spec, ml: int) -> int:
    """Return phase index for a given impact number in a cycling simulation."""
    total_cycle = sum(p.fluence_ml for p in spec.phases) * ml
    pos = (impact - 1) % total_cycle
    cum = 0
    for pi, p in enumerate(spec.phases):
        cum += p.fluence_ml * ml
        if pos < cum:
            return pi
    return len(spec.phases) - 1


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

    # filter to carbon-containing products only
    ep_carbon = [r for r in ep if r['n_C'] > 0]

    has_radicals_nc = any(r['cn'] > 0 for r in nc)
    # is_cyc: True when spec declares multiple phases (phase breakdown in output).
    # Distinct from has_radicals_nc (cn>0 rows) which is True only in RIE-style phases.
    is_cyc = bool(spec is not None and spec.phases)
    n_ion = sum(1 for r in nc if r['cn'] == 0)
    n_radical = sum(1 for r in nc if r['cn'] > 0)
    n0 = nc[0]['n_carbon'] if nc else 0

    stats: Dict = {'is_cycling': is_cyc, 'ml': ml,
                   'n_ion_impacts': n_ion, 'n_radical_impacts': n_radical}

    # ── etch yield ────────────────────────────────────────────────────────────
    ion_yields = [max(0.0, v) for v in _ion_etch_sequence(nc)]
    ey_mean, ey_err = block_stats(ion_yields, n_blocks)
    stats['etch_yield_per_ion'] = ey_mean
    stats['etch_yield_per_ion_err'] = ey_err

    if has_radicals_nc:
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
        stats['etch_yield_per_radical_err'] = None
        stats['etch_yield_total'] = ey_mean

    # ── cumulative etch ───────────────────────────────────────────────────────
    # Use total C in ejected products (correct even when addfix replenishes slab).
    stats['etch_depth_ml'] = sum(r['n_C'] for r in ep_carbon) / ml if ep_carbon else 0.0

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
    if ep_carbon:
        oc_ratios = [r['n_O'] / r['n_C'] if r['n_C'] > 0 else 0.0 for r in ep_carbon]
        stats['avg_oc_ratio'] = float(np.mean(oc_ratios))
        stats['avg_o_per_product'] = float(np.mean([r['n_O'] for r in ep_carbon]))
        stats['avg_c_per_product'] = float(np.mean([r['n_C'] for r in ep_carbon]))
        stats['n_products_carbon'] = len(ep_carbon)
    else:
        stats['avg_oc_ratio'] = 0.0
        stats['avg_o_per_product'] = 0.0
        stats['avg_c_per_product'] = 0.0
        stats['n_products_carbon'] = 0

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
            phase_stats[key]['ion_etch'].append(max(0.0, float(rec['ion_etch'])))
            phase_stats[key]['radical_etch'].append(float(rec['radical_etch']))
            phase_stats[key]['o_content'].append(rec['n_oxygen'] / ml)

        # ── per-phase O uptake at phase-end ───────────────────────────────────
        ion_recs = [r for r in nc if r['cn'] == 0]
        max_impact = max((r['impact'] for r in ion_recs), default=0)
        total_cycle_ml = sum(p.fluence_ml for p in spec.phases) * ml
        for pi, p in enumerate(spec.phases):
            phase_start = sum(spec.phases[j].fluence_ml for j in range(pi)) * ml
            phase_end_off = phase_start + p.fluence_ml * ml
            o_end_list = []
            for cyc in range(spec.cycles):
                lo = cyc * total_cycle_ml + phase_start
                hi = cyc * total_cycle_ml + phase_end_off
                if max_impact < hi:
                    break
                recs = [r for r in ion_recs if lo < r['impact'] <= hi]
                if recs:
                    o_end_list.append(recs[-1]['n_oxygen'] / ml)
            o_end_m, o_end_e = block_stats(o_end_list, n_blocks) if o_end_list else (0.0, 0.0)
            if pi not in phase_stats:
                phase_stats[pi] = {'name': spec.phases[pi].species,
                                   'ion_etch': [], 'radical_etch': [], 'o_content': []}
            phase_stats[pi]['o_end_mean'] = o_end_m
            phase_stats[pi]['o_end_err'] = o_end_e

        # ── per-phase amorphous (CNA) ─────────────────────────────────────────
        if cna_records:
            for rec in cna_records:
                pi = _phase_of_impact(rec['impact'], spec, ml)
                if pi in phase_stats:
                    phase_stats[pi].setdefault('cna_amorphous_ml', []).append(
                        rec['n_amorphous'] / ml)
                    phase_stats[pi].setdefault('cna_thickness', []).append(
                        rec['amorphous_thickness_A'])
            for pi, ps in phase_stats.items():
                am_list = ps.get('cna_amorphous_ml', [])
                th_list = ps.get('cna_thickness', [])
                ps['amorphous_ml_mean'] = float(np.mean(am_list)) if am_list else None
                ps['amorphous_thickness_mean'] = float(np.mean(th_list)) if th_list else None

        # ── per-phase product composition ─────────────────────────────────────
        if ep_carbon:
            phase_ep: Dict[int, list] = {pi: [] for pi in range(len(spec.phases))}
            for r in ep_carbon:
                pi = _phase_of_impact(r['impact'], spec, ml)
                if pi in phase_ep:
                    phase_ep[pi].append(r)
            for pi, recs in phase_ep.items():
                if recs:
                    if pi not in phase_stats:
                        phase_stats[pi] = {'name': spec.phases[pi].species,
                                           'ion_etch': [], 'radical_etch': [], 'o_content': []}
                    oc = [r['n_O'] / r['n_C'] if r['n_C'] > 0 else 0.0 for r in recs]
                    phase_stats[pi]['ep_oc_ratio'] = float(np.mean(oc))
                    phase_stats[pi]['ep_avg_c'] = float(np.mean([r['n_C'] for r in recs]))
                    phase_stats[pi]['ep_avg_o'] = float(np.mean([r['n_O'] for r in recs]))
                    phase_stats[pi]['ep_n'] = len(recs)

        stats['per_phase'] = {
            pi: {
                'name':                   ps['name'],
                'etch_yield_mean':        float(np.mean(ps['ion_etch'])) if ps['ion_etch'] else 0.0,
                'etch_yield_err':         block_stats(ps['ion_etch'], n_blocks)[1],
                'radical_etch_mean':      float(np.mean(ps['radical_etch'])) if ps['radical_etch'] else 0.0,
                'o_content_mean':         float(np.mean(ps['o_content'])) if ps['o_content'] else 0.0,
                'o_end_mean':             ps.get('o_end_mean', 0.0),
                'o_end_err':              ps.get('o_end_err', 0.0),
                'amorphous_ml_mean':      ps.get('amorphous_ml_mean'),
                'amorphous_thickness_mean': ps.get('amorphous_thickness_mean'),
                'ep_oc_ratio':            ps.get('ep_oc_ratio'),
                'ep_avg_c':               ps.get('ep_avg_c'),
                'ep_avg_o':               ps.get('ep_avg_o'),
                'ep_n':                   ps.get('ep_n'),
            }
            for pi, ps in phase_stats.items()
        }
    else:
        stats['per_phase'] = {}

    # ── throughput ────────────────────────────────────────────────────────────
    wall_s = _wall_time_seconds(sim_dir)
    if wall_s and wall_s > 0 and n_ion > 0:
        impacts_per_day = n_ion * 86400.0 / wall_s
        stats['impacts_per_day'] = impacts_per_day
        stats['ml_per_day'] = (n_ion / ml) * 86400.0 / wall_s
        if is_cyc and spec is not None and spec.phases is not None:
            cycle_ml = sum(p.fluence_ml for p in spec.phases)
            stats['cycles_per_day'] = stats['ml_per_day'] / cycle_ml if cycle_ml > 0 else None
        else:
            stats['cycles_per_day'] = None
    else:
        stats['impacts_per_day'] = None
        stats['ml_per_day'] = None
        stats['cycles_per_day'] = None

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
    has_radicals = stats.get('n_radical_impacts', 0) > 0
    per_phase = stats.get('per_phase', {})
    lines = ['# DiamondEtchMD Analysis Summary', '']

    # ── Etch statistics ───────────────────────────────────────────────────────
    lines.append('## Etch statistics')
    lines.append(
        f"Cumulative etch (MLs):{_fmt(stats.get('etch_depth_ml'))}"
    )
    n_ion_s = stats.get('n_ion_impacts', 0)
    ml_s = stats.get('ml', 1) or 1
    lines.append(f"Total ion dose (MLs):{_fmt(n_ion_s / ml_s)}")
    lines.append(
        f"Etch yield (C/ion):{_fmt(stats.get('etch_yield_per_ion'), stats.get('etch_yield_per_ion_err'))}"
    )
    if has_radicals:
        lines.append(
            f"  Ion etch yield (C/ion):{_fmt(stats.get('etch_yield_per_ion'), stats.get('etch_yield_per_ion_err'))}"
        )
        lines.append(
            f"  Radical etch yield (C/radical):{_fmt(stats.get('etch_yield_per_radical'), stats.get('etch_yield_per_radical_err'))}"
        )
        lines.append(
            f"  Total etch yield:{_fmt(stats.get('etch_yield_total'))}"
        )

    if per_phase:
        for pi, ps in sorted(per_phase.items()):
            radical_str = (
                f"  radical: {ps['radical_etch_mean']:.4g} C/radical"
                if ps.get('radical_etch_mean') else ''
            )
            lines.append(
                f"  Phase {pi} ({ps['name']}): "
                f"{ps['etch_yield_mean']:.4g} ± {ps['etch_yield_err']:.2g} C/ion"
                + radical_str
            )

    # ── Surface oxygen ────────────────────────────────────────────────────────
    lines += ['', '## Surface oxygen']
    if per_phase:
        lines.append('Avg phase-end O uptake (MLs):')
        for pi, ps in sorted(per_phase.items()):
            o_m = ps.get('o_end_mean', 0.0)
            o_e = ps.get('o_end_err', 0.0)
            lines.append(f"  Phase {pi} ({ps['name']}):{_fmt(o_m, o_e if o_e else None)}")
    else:
        lines.append(
            f"Avg O uptake (MLs):{_fmt(stats.get('o_uptake_ml'), stats.get('o_uptake_ml_err'))}"
        )

    # ── Amorphous carbon (CNA) ────────────────────────────────────────────────
    if stats.get('amorphous_ml') is not None:
        lines += ['', '## Amorphous carbon (CNA)']
        lines.append(
            f"Amorphous C (MLs):{_fmt(stats.get('amorphous_ml'), stats.get('amorphous_ml_err'))}"
        )
        lines.append(
            f"Amorphous layer thickness (Å):{_fmt(stats.get('amorphous_thickness_A'), stats.get('amorphous_thickness_A_err'))}"
        )
        if per_phase:
            for pi, ps in sorted(per_phase.items()):
                am = ps.get('amorphous_ml_mean')
                th = ps.get('amorphous_thickness_mean')
                am_str = f'{am:.4g}' if am is not None else 'N/A'
                th_str = f'{th:.4g}' if th is not None else 'N/A'
                lines.append(
                    f"  Phase {pi} ({ps['name']}): amorphous C  {am_str}, thickness  {th_str} Å"
                )

    # ── Carbon-containing product composition ─────────────────────────────────
    n_prod = stats.get('n_products_carbon', 0)
    if n_prod > 0:
        lines += ['', '## Carbon-containing product composition']
        if not per_phase:
            lines.append(f"Avg O:C ratio:{_fmt(stats.get('avg_oc_ratio'))}")
            lines.append(f"Avg O per product:{_fmt(stats.get('avg_o_per_product'))}")
            lines.append(f"Avg C per product:{_fmt(stats.get('avg_c_per_product'))}")
            lines.append(f"n products:  {n_prod}")
        else:
            # overall first
            lines.append(f"Avg O:C ratio:{_fmt(stats.get('avg_oc_ratio'))}")
            lines.append(f"Avg O per product:{_fmt(stats.get('avg_o_per_product'))}")
            lines.append(f"Avg C per product:{_fmt(stats.get('avg_c_per_product'))}")
            lines.append(f"n products:  {n_prod}")
            for pi, ps in sorted(per_phase.items()):
                ep_n = ps.get('ep_n')
                if ep_n:
                    oc = ps.get('ep_oc_ratio')
                    avg_c = ps.get('ep_avg_c')
                    avg_o = ps.get('ep_avg_o')
                    oc_s = f'{oc:.4g}' if oc is not None else 'N/A'
                    c_s = f'{avg_c:.4g}' if avg_c is not None else 'N/A'
                    o_s = f'{avg_o:.4g}' if avg_o is not None else 'N/A'
                    lines.append(
                        f"  Phase {pi} ({ps['name']}): O:C = {oc_s}, avg C = {c_s}, avg O = {o_s}, n = {ep_n}"
                    )

    # ── Throughput ────────────────────────────────────────────────────────────
    lines += ['', '## Throughput']
    ipd = stats.get('impacts_per_day')
    if ipd is not None:
        lines.append(f'Ion impacts per day:{_fmt(ipd)}')
    else:
        lines.append('Ion impacts per day:  N/A')
    ml_day = stats.get('ml_per_day')
    if ml_day is not None:
        lines.append(f'Monolayers per day:{_fmt(ml_day)}')
    else:
        lines.append('Monolayers per day:  N/A')
    if is_cyc:
        cpd = stats.get('cycles_per_day')
        if cpd is not None:
            lines.append(f'Cycles per day:{_fmt(cpd)}')
    lines.append(f"Total ion impacts:  {stats.get('n_ion_impacts', 'N/A')}")
    if has_radicals:
        lines.append(f"Total radical impacts:  {stats.get('n_radical_impacts', 'N/A')}")

    Path(path).write_text('\n'.join(lines) + '\n')
