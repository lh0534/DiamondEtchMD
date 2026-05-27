#!/usr/bin/env python3
"""Convert impact_snaps/*.data -> all_impacts.dump in LAMMPS custom dump format.

Usage:
    python make_impact_dump.py [condition_dir]

condition_dir defaults to the current directory.
Output is written to <condition_dir>/all_impacts.dump.
"""

import re
import sys
from pathlib import Path

import numpy as np


def parse_data_file(path):
    """Return (timestep, xlo, xhi, ylo, yhi, zlo, zhi, atoms_array).

    atoms_array columns: id type q x y z  (from charge/kk format)
    """
    text = path.read_text()
    lines = text.splitlines()

    # Box bounds
    xlo = xhi = ylo = yhi = zlo = zhi = None
    for line in lines:
        m = re.match(r"([\S]+)\s+([\S]+)\s+xlo xhi", line)
        if m:
            xlo, xhi = float(m.group(1)), float(m.group(2))
        m = re.match(r"([\S]+)\s+([\S]+)\s+ylo yhi", line)
        if m:
            ylo, yhi = float(m.group(1)), float(m.group(2))
        m = re.match(r"([\S]+)\s+([\S]+)\s+zlo zhi", line)
        if m:
            zlo, zhi = float(m.group(1)), float(m.group(2))

    # Atom rows (after "Atoms # charge/kk")
    atom_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atom_start = i + 2  # blank line then data
            break

    rows = []
    for line in lines[atom_start:]:
        line = line.strip()
        if not line:
            break
        rows.append(line)

    # charge/kk: atom_id type charge x y z ix iy iz
    arr = np.fromstring(" ".join(rows), sep=" ").reshape(-1, 9)
    # return cols: id(0) type(1) q(2) x(3) y(4) z(5)
    return xlo, xhi, ylo, yhi, zlo, zhi, arr


def timestep_from_name(name):
    # "{N}_0.data" -> N,  "0.data" -> 0
    m = re.match(r"^(\d+)_0\.data$", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)\.data$", name)
    if m:
        return int(m.group(1))
    return None


def main():
    cond_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    data_dir = cond_dir / "impact_snaps"
    out_path = cond_dir / "all_impacts.dump"

    files = []
    for p in data_dir.glob("*.data"):
        ts = timestep_from_name(p.name)
        if ts is not None:
            files.append((ts, p))

    files.sort(key=lambda x: x[0])
    print(f"Found {len(files)} data files → {out_path}")

    with out_path.open("w") as fout:
        for i, (ts, p) in enumerate(files):
            xlo, xhi, ylo, yhi, zlo, zhi, arr = parse_data_file(p)
            n = len(arr)

            fout.write("ITEM: TIMESTEP\n")
            fout.write(f"{ts}\n")
            fout.write("ITEM: NUMBER OF ATOMS\n")
            fout.write(f"{n}\n")
            fout.write("ITEM: BOX BOUNDS pp pp mm\n")
            fout.write(f"{xlo:.16e} {xhi:.16e}\n")
            fout.write(f"{ylo:.16e} {yhi:.16e}\n")
            fout.write(f"{zlo:.16e} {zhi:.16e}\n")
            fout.write("ITEM: ATOMS id type x y z vx vy vz q\n")

            # arr cols: id(0) type(1) q(2) x(3) y(4) z(5)  ix iy iz unused
            ids   = arr[:, 0].astype(int)
            types = arr[:, 1].astype(int)
            q     = arr[:, 2]
            x, y, z = arr[:, 3], arr[:, 4], arr[:, 5]

            for j in range(n):
                fout.write(
                    f"{ids[j]} {types[j]} "
                    f"{x[j]:.6g} {y[j]:.6g} {z[j]:.6g} "
                    f"0 0 0 "
                    f"{q[j]:.6g}\n"
                )

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)} frames written")

    print("Done.")


if __name__ == "__main__":
    main()
