#!/usr/bin/env python3
"""Convert impact_snaps/*.data -> LAMMPS custom dump file.

Default mode (ion snapshots only):
    python make_impact_dump.py [condition_dir]
    Output: all_impacts.dump  (one frame per ion impact, i.e. *_0.data files)

All-events mode (ions + radicals, in temporal order):
    python make_impact_dump.py [condition_dir] --all-events
    Output: all_impacts_withrads.dump

Temporal ordering for --all-events:
    Radicals precede their associated ion. For each ion cycle c:
        0_1, 0_2, ..., 0_FR  (radicals before ion 1)
        1_0                   (post-ion-1 snapshot)
        1_1, ..., 1_FR        (radicals before ion 2)
        2_0                   (post-ion-2 snapshot)
        ...
    The ITEM:TIMESTEP value encodes this order as c*(FR+1)+cn for radicals
    and c*(FR+1) for ion snapshots, where FR = max radical index seen.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def parse_data_file(path):
    """Return (xlo, xhi, ylo, yhi, zlo, zhi, atoms_array).

    atoms_array columns: id type q x y z  (from charge/kk format)
    """
    text = path.read_text()
    lines = text.splitlines()

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

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError(f"missing box bounds (truncated file?): {path}")

    atom_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atom_start = i + 2
            break

    if atom_start is None:
        raise ValueError(f"no Atoms section found (truncated file?): {path}")

    rows = []
    for line in lines[atom_start:]:
        line = line.strip()
        if not line:
            break
        rows.append(line)

    # charge/kk: atom_id type charge x y z ix iy iz
    arr = np.fromstring(" ".join(rows), sep=" ").reshape(-1, 9)
    return xlo, xhi, ylo, yhi, zlo, zhi, arr


def collect_ion_only(data_dir):
    """Return sorted list of (sort_key, path) for ion-only snapshots (*_0.data)."""
    files = []
    for p in data_dir.glob("*.data"):
        m = re.match(r"^(\d+)_0\.data$", p.name)
        if m:
            files.append((int(m.group(1)), p))
        else:
            m = re.match(r"^(\d+)\.data$", p.name)
            if m:
                files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return files


def collect_all_events(data_dir):
    """Return sorted list of (sort_key, label, path) for every snapshot.

    sort_key encodes temporal order: radicals before their associated ion.
    label is a human-readable string like 'radical_1_3' or 'ion_2'.
    """
    entries = []
    max_cn = 0

    for p in data_dir.glob("*.data"):
        m = re.match(r"^(\d+)_(\d+)\.data$", p.name)
        if m:
            c, cn = int(m.group(1)), int(m.group(2))
            max_cn = max(max_cn, cn)
            entries.append((c, cn, p))

    if not entries:
        return []

    fr = max_cn  # flux ratio inferred from max radical index

    def epoch(c, cn):
        # cn==0 → post-ion snapshot; cn>0 → radical cn of cycle c
        # Radicals of cycle c come BEFORE ion c+1, i.e. before (c+1)*()
        if cn == 0:
            return c * (fr + 1)
        else:
            return c * (fr + 1) + cn

    result = []
    for c, cn, p in entries:
        key = epoch(c, cn)
        label = f"ion_{c}" if cn == 0 else f"radical_{c}_{cn}"
        result.append((key, label, p))

    result.sort(key=lambda x: x[0])
    return result


def write_dump(fout, sort_key, xlo, xhi, ylo, yhi, zlo, zhi, arr):
    n = len(arr)
    ids   = arr[:, 0].astype(int)
    types = arr[:, 1].astype(int)
    q     = arr[:, 2]
    x, y, z = arr[:, 3], arr[:, 4], arr[:, 5]

    fout.write("ITEM: TIMESTEP\n")
    fout.write(f"{sort_key}\n")
    fout.write("ITEM: NUMBER OF ATOMS\n")
    fout.write(f"{n}\n")
    fout.write("ITEM: BOX BOUNDS pp pp mm\n")
    fout.write(f"{xlo:.16e} {xhi:.16e}\n")
    fout.write(f"{ylo:.16e} {yhi:.16e}\n")
    fout.write(f"{zlo:.16e} {zhi:.16e}\n")
    fout.write("ITEM: ATOMS id type x y z vx vy vz q\n")

    for j in range(n):
        fout.write(
            f"{ids[j]} {types[j]} "
            f"{x[j]:.6g} {y[j]:.6g} {z[j]:.6g} "
            f"0 0 0 "
            f"{q[j]:.6g}\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cond_dir", nargs="?", default=".",
                        help="simulation directory (default: .)")
    parser.add_argument("--all-events", action="store_true",
                        help="include radical snapshots in temporal order")
    args = parser.parse_args()

    cond_dir = Path(args.cond_dir)
    data_dir = cond_dir / "impact_snaps"

    if args.all_events:
        entries = collect_all_events(data_dir)
        out_path = cond_dir / "all_impacts_withrads.dump"
        print(f"Found {len(entries)} snapshots (ions + radicals) → {out_path}")

        skipped = 0
        with out_path.open("w") as fout:
            for i, (key, label, p) in enumerate(entries):
                try:
                    xlo, xhi, ylo, yhi, zlo, zhi, arr = parse_data_file(p)
                except ValueError as e:
                    print(f"  WARNING: skipping {p.name} — {e}")
                    skipped += 1
                    continue
                write_dump(fout, key, xlo, xhi, ylo, yhi, zlo, zhi, arr)
                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(entries)} frames written ({label})")
        if skipped:
            print(f"  Skipped {skipped} corrupt file(s).")

    else:
        files = collect_ion_only(data_dir)
        out_path = cond_dir / "all_impacts.dump"
        print(f"Found {len(files)} data files → {out_path}")

        skipped = 0
        with out_path.open("w") as fout:
            for i, (ts, p) in enumerate(files):
                try:
                    xlo, xhi, ylo, yhi, zlo, zhi, arr = parse_data_file(p)
                except ValueError as e:
                    print(f"  WARNING: skipping {p.name} — {e}")
                    skipped += 1
                    continue
                write_dump(fout, ts, xlo, xhi, ylo, yhi, zlo, zhi, arr)
                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(files)} frames written")
        if skipped:
            print(f"  Skipped {skipped} corrupt file(s).")

    print("Done.")


if __name__ == "__main__":
    main()
