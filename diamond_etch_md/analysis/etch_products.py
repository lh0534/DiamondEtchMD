"""
analysis/etch_products.py — parsing and analysis of etch_products.txt output files.

etch_products.txt format
------------------------
Each line corresponds to one ejected cluster detected during the simulation.

Current 5-column format (written by sweep.lmp):
  impact  n_C  n_H  n_O  n_Ar

Legacy 4-column format (runs before Ar tracking):
  impact  n_C  n_H  n_O   (n_Ar assumed 0)

Legacy '*' format (oldest runs):
  * Cluster of C<n>H<n>O<n> sputtered on impact <n>

Lines beginning with '#' are comments and are skipped.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

_LEGACY_RE = re.compile(
    r"^\*\s*Cluster of C(\d+)H(\d+)O(\d+)\s+sputtered on impact\s+(\d+)",
    re.IGNORECASE,
)


def parse_etch_products(path) -> List[Dict[str, Any]]:
    """Parse an etch_products.txt file into a list of cluster records.

    Returns
    -------
    list of dict with keys: impact (int), n_C (int), n_H (int), n_O (int), n_Ar (int)
    """
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("*"):
            m = _LEGACY_RE.match(line)
            if m:
                records.append({
                    "impact": int(m.group(4)),
                    "cn":    0,
                    "n_C":   int(m.group(1)),
                    "n_H":   int(m.group(2)),
                    "n_O":   int(m.group(3)),
                    "n_Ar":  0,
                })
            continue
        parts = line.split()
        if len(parts) >= 7:
            # 7-column format: impact cn n_C n_H n_O n_Ar vcm_z
            records.append({
                "impact": int(parts[0]),
                "cn":     int(parts[1]),
                "n_C":    int(parts[2]),
                "n_H":    int(parts[3]),
                "n_O":    int(parts[4]),
                "n_Ar":   int(parts[5]),
                "vcm_z":  float(parts[6]),
            })
        elif len(parts) >= 6:
            # 6-column format: impact cn n_C n_H n_O n_Ar
            records.append({
                "impact": int(parts[0]),
                "cn":     int(parts[1]),
                "n_C":    int(parts[2]),
                "n_H":    int(parts[3]),
                "n_O":    int(parts[4]),
                "n_Ar":   int(parts[5]),
                "vcm_z":  None,
            })
        elif len(parts) >= 5:
            records.append({
                "impact": int(parts[0]),
                "cn":     0,
                "n_C":    int(parts[1]),
                "n_H":    int(parts[2]),
                "n_O":    int(parts[3]),
                "n_Ar":   int(parts[4]),
                "vcm_z":  None,
            })
        elif len(parts) >= 4:
            records.append({
                "impact": int(parts[0]),
                "cn":     0,
                "n_C":    int(parts[1]),
                "n_H":    int(parts[2]),
                "n_O":    int(parts[3]),
                "n_Ar":   0,
                "vcm_z":  None,
            })
    return records


def etch_yield(records: List[Dict[str, Any]], ml: int) -> float:
    """Compute the average etch yield in carbon atoms per incident particle."""
    if not records:
        return 0.0
    total_carbon = sum(r["n_C"] for r in records)
    n_impacts = records[-1]["impact"]
    return total_carbon / n_impacts
