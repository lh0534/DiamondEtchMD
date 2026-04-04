"""
analysis/ncarbon.py — parsing and analysis of ncarbon.txt output files.

ncarbon.txt format
------------------
One line is appended after each impact cycle by head.lmp.
Columns (space-separated):

  impact#     integer   Impact number (1-indexed, matching etch_products.txt)
  n_carbon    integer   Total number of carbon atoms currently in the simulation box
  n_hydrogen  integer   Total number of hydrogen atoms currently in the simulation box
  n_oxygen    integer   Total number of oxygen atoms currently in the simulation box

The file is also used at job startup to determine the resume impact number
(via `tail -1 ncarbon.txt | awk '{print $1}'` in the submit script).

Example lines:
  1  648  0  0
  2  647  0  0
  50 630  0  12

Notes:
  - n_carbon decreasing over time represents etching; the etch depth can be
    computed from (n_carbon_initial - n_carbon) / (ML * rho_layer), where
    rho_layer is the number of carbon atoms per Å of depth.
  - n_hydrogen and n_oxygen > 0 indicate surface termination or radical uptake.
"""

from pathlib import Path
from typing import List, Dict, Any


def parse_ncarbon(path) -> List[Dict[str, Any]]:
    """Parse an ncarbon.txt file into a list of per-impact records.

    Parameters
    ----------
    path:
        Path-like or str pointing to an ncarbon.txt file.

    Returns
    -------
    list of dict, each with keys:
        impact (int), n_carbon (int), n_hydrogen (int), n_oxygen (int)
    """
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        records.append({
            "impact":     int(parts[0]),
            "n_carbon":   int(parts[1]),
            "n_hydrogen": int(parts[2]),
            "n_oxygen":   int(parts[3]),
        })
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
