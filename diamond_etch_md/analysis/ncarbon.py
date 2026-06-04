"""
analysis/ncarbon.py — parsing and analysis of ncarbon.txt output files.

ncarbon.txt format
------------------
**Single-species** (head.lmp): one line per completed impact:

  impact#  n_carbon  n_hydrogen  n_oxygen

**Cycling** (head_cycling.py): one line per radical AND per ion impact:

  impact#  cn  n_carbon  n_hydrogen  n_oxygen

  cn > 0 after each O• radical (1-indexed within the current ion's radical
  loop); cn = 0 after each ion impact.  This enables mid-radical-loop
  restarts.

The file is used at job startup to determine the resume point
(col 1 = impact#, col 2 = cn for cycling).

Example single-species lines:
  1  648  0  0
  50 630  0  12

Example cycling lines:
  1  1  648  0  3
  1  2  648  0  5
  1  0  647  0  6
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple


def parse_ncarbon(path) -> List[Dict[str, Any]]:
    """Parse an ncarbon.txt file (single-species or cycling) into records.

    Auto-detects 4-column (single-species) vs 5-column (cycling) format.
    In both cases returns dicts with a unified key set; cycling records carry
    an extra 'cn' key (always 0 for single-species).

    Returns
    -------
    list of dict with keys:
        impact (int), cn (int, 0 for single-species),
        n_carbon (int), n_hydrogen (int), n_oxygen (int)
    """
    # Use an ordered dict keyed by (impact, cn) so that if the same impact+cn
    # appears more than once (e.g. a legacy 4-col entry followed by a 5-col
    # re-run from a restart), the LAST occurrence wins (5-col overwrites 4-col).
    by_key: dict = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 5:
            rec = {
                "impact":     int(parts[0]),
                "cn":         int(parts[1]),
                "n_carbon":   int(parts[2]),
                "n_hydrogen": int(parts[3]),
                "n_oxygen":   int(parts[4]),
            }
        elif len(parts) >= 4:
            rec = {
                "impact":     int(parts[0]),
                "cn":         0,
                "n_carbon":   int(parts[1]),
                "n_hydrogen": int(parts[2]),
                "n_oxygen":   int(parts[3]),
            }
        else:
            continue
        by_key[(rec["impact"], rec["cn"])] = rec
    return list(by_key.values())


def is_cycling_format(path) -> bool:
    """Return True if ncarbon.txt uses the 5-column cycling format."""
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            return len(line.split()) == 5
    return False


def parse_ncarbon_cycling(path, spec=None) -> List[Dict[str, Any]]:
    """Parse a cycling ncarbon.txt into per-ion-impact records with phase info.

    Requires the 5-column cycling format.  Each returned record represents
    one ion impact (cn == 0 rows); associated radical rows supply pre-ion
    state.  If `spec` is provided, phase_idx and phase_name are annotated.

    Returns
    -------
    list of dict with keys:
        impact (int), cycle_idx (int), phase_idx (int), phase_name (str),
        n_carbon_pre_ion (int), n_carbon (int), n_hydrogen (int), n_oxygen (int),
        radical_etch (int), ion_etch (int)
    """
    raw = parse_ncarbon(path)
    if not raw:
        return []

    # Build phase cycle boundaries if spec available
    phase_boundaries = []  # list of cumulative ML impact counts (1-indexed)
    if spec is not None and spec.phases is not None:
        cum = 0
        for p in spec.phases:
            cum += p.fluence_ml * spec.ml
            phase_boundaries.append((cum, p.species))

    records = []
    # Use the first simulation record (impact > 0) as baseline, not the make_surf
    # row (impact == 0), which may come from a different box size / prior run.
    sim_start = next((r for r in raw if r['impact'] > 0), raw[0])
    prev_ion_nc = sim_start['n_carbon']
    last_radical_nc = prev_ion_nc

    for r in raw:
        if r['impact'] == 0:
            continue  # skip make_surf row
        if r['cn'] == 0:
            # ion impact
            ion_etch = last_radical_nc - r['n_carbon']
            radical_etch = prev_ion_nc - last_radical_nc

            rec = {
                'impact':           r['impact'],
                'n_carbon_pre_ion': last_radical_nc,
                'n_carbon':         r['n_carbon'],
                'n_hydrogen':       r['n_hydrogen'],
                'n_oxygen':         r['n_oxygen'],
                'ion_etch':         ion_etch,
                'radical_etch':     radical_etch,
                'cycle_idx':        0,
                'phase_idx':        0,
                'phase_name':       '',
            }

            if spec is not None and spec.phases is not None:
                total_cycle = sum(p.fluence_ml for p in spec.phases) * spec.ml
                cycle_idx = (r['impact'] - 1) // total_cycle
                pos_in_cycle = (r['impact'] - 1) % total_cycle
                cum = 0
                for pi, p in enumerate(spec.phases):
                    cum += p.fluence_ml * spec.ml
                    if pos_in_cycle < cum:
                        rec['cycle_idx'] = cycle_idx
                        rec['phase_idx'] = pi
                        rec['phase_name'] = p.species
                        break

            records.append(rec)
            prev_ion_nc = r['n_carbon']
            last_radical_nc = r['n_carbon']
        else:
            last_radical_nc = r['n_carbon']

    return records


def etch_depth(
    records: List[Dict[str, Any]],
    ml: int,
    box_x: int,
    box_y: int,
    orientation: str,
) -> List[float]:
    """Compute etch depth in monolayers as a function of impact number.

    Parameters
    ----------
    records:
        List of per-impact records as returned by parse_ncarbon().
    ml:
        Atoms per monolayer (ML = ml_factor * box_x * box_y).
    box_x:
        Lateral box size in x lattice units (unused; reserved for future Å conversion).
    box_y:
        Lateral box size in y lattice units (unused; reserved for future Å conversion).
    orientation:
        Surface orientation string ('100', '111', or '113') (unused; reserved for
        future layer-density conversion).

    Returns
    -------
    list of float
        Etch depth in monolayers at each recorded impact, relative to the
        initial carbon count (records[0]['n_carbon']).  Returns [] if records is empty.
    """
    if not records:
        return []
    n0 = records[0]["n_carbon"]
    return [(n0 - r["n_carbon"]) / ml for r in records]
