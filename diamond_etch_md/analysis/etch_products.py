"""
analysis/etch_products.py — parsing and analysis of etch_products.txt output files.

etch_products.txt format
------------------------
Each line corresponds to one ejected cluster detected during the simulation.
Columns (space-separated):

  impact#   integer   Impact number (1-indexed) at which the cluster was ejected
  atom_type string    LAMMPS atom type label of the cluster centroid atom
  n_C       integer   Number of carbon atoms in the cluster
  n_H       integer   Number of hydrogen atoms in the cluster
  n_O       integer   Number of oxygen atoms in the cluster
  vx        float     x-component of cluster centre-of-mass velocity (Å/fs)
  vy        float     y-component of cluster centre-of-mass velocity (Å/fs)
  vz        float     z-component of cluster centre-of-mass velocity (Å/fs)

The file is appended to by sweep.lmp after each impact event that produces
detached clusters.  A cluster counts as ejected when it separates from the
main substrate body and has positive vz (moving away from the surface).

Example lines:
  12 C 1 0 0  0.001  0.003  0.412
  47 C 2 1 0 -0.002  0.001  0.387
"""

from pathlib import Path
from typing import List, Dict, Any


def parse_etch_products(path) -> List[Dict[str, Any]]:
    """Parse an etch_products.txt file into a list of cluster records.

    Parameters
    ----------
    path:
        Path-like or str pointing to an etch_products.txt file.

    Returns
    -------
    list of dict, each with keys:
        impact (int), atom_type (str), n_C (int), n_H (int), n_O (int),
        vx (float), vy (float), vz (float)
    """
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        records.append({
            "impact":    int(parts[0]),
            "atom_type": parts[1],
            "n_C":       int(parts[2]),
            "n_H":       int(parts[3]),
            "n_O":       int(parts[4]),
            "vx":        float(parts[5]),
            "vy":        float(parts[6]),
            "vz":        float(parts[7]),
        })
    return records


def etch_yield(records: List[Dict[str, Any]], ml: int) -> float:
    """Compute the average etch yield in carbon atoms per incident particle.

    Parameters
    ----------
    records:
        List of cluster records as returned by parse_etch_products().
    ml:
        Atoms per monolayer (used to determine total number of impacts from
        the last impact number in records).

    Returns
    -------
    float
        Mean number of carbon atoms ejected per incident particle (impact).
        Returns 0.0 if records is empty.
    """
    if not records:
        return 0.0
    total_carbon = sum(r["n_C"] for r in records)
    n_impacts = records[-1]["impact"]
    return total_carbon / n_impacts
