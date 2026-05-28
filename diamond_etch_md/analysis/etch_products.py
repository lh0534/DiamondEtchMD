"""
analysis/etch_products.py — parsing and analysis of etch_products.txt output files.

etch_products.txt format
------------------------
Each line corresponds to one ejected cluster detected during the simulation.

Current 5-column format (written by sweep.lmp):
  impact  cn  n_C  n_H  n_O
  cn=0 means ejected during the ion phase; cn>0 means during radical sweep #cn.

Legacy 4-column format (runs before cn tracking):
  impact  n_C  n_H  n_O   (cn assumed 0)

Legacy '*' format (oldest runs):
  * Cluster of C<n>H<n>O<n> sputtered on impact <n> [<cn>]
  The trailing cn field is optional.

Lines beginning with '#' are comments and are skipped.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

_LEGACY_RE = re.compile(
    r"^\*\s*Cluster of C(\d+)H(\d+)O(\d+)\s+sputtered on impact\s+(\d+)(?:\s+(\d+))?",
    re.IGNORECASE,
)


def parse_etch_products(path) -> List[Dict[str, Any]]:
    """Parse an etch_products.txt file into a list of cluster records.

    Returns
    -------
    list of dict with keys: impact (int), cn (int), n_C (int), n_H (int), n_O (int)
    cn=0 means ejected during ion phase; cn>0 means during radical sweep cn.
    """
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("*"):
            m = _LEGACY_RE.match(line)
            if m:
                nc  = int(m.group(1))
                nh  = int(m.group(2))
                no  = int(m.group(3))
                imp = int(m.group(4))
                cn  = int(m.group(5)) if m.group(5) is not None else 0
                records.append({"impact": imp, "cn": cn, "n_C": nc, "n_H": nh, "n_O": no})
            continue
        parts = line.split()
        if len(parts) >= 5:
            records.append({
                "impact": int(parts[0]),
                "cn":     int(parts[1]),
                "n_C":    int(parts[2]),
                "n_H":    int(parts[3]),
                "n_O":    int(parts[4]),
            })
        elif len(parts) >= 4:
            records.append({
                "impact": int(parts[0]),
                "cn":     0,
                "n_C":    int(parts[1]),
                "n_H":    int(parts[2]),
                "n_O":    int(parts[3]),
            })
    return records


def etch_yield(records: List[Dict[str, Any]], ml: int) -> float:
    """Compute the average etch yield in carbon atoms per incident particle."""
    if not records:
        return 0.0
    total_carbon = sum(r["n_C"] for r in records)
    n_impacts = records[-1]["impact"]
    return total_carbon / n_impacts
