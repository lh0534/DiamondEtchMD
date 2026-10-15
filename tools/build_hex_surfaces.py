"""Build the hexagonal-diamond surfaces with make_surf.lmp for inspection in OVITO.

For each hex orientation: make_sim into <outdir>/<orientation>, then run that
directory's make_surf.lmp on the CPU (Kokkos package line removed).  Writes, per
orientation:
  hex_1x1_0K.dat       minimized slab
  hex_1x1_300K.dat     after 10 ps at 300 K (= impact_snaps/0.data)
and prints a summary: atom count, coordination of the top layers, and the
fraction of slab carbon the CNA labels as diamond.

Run (one CPU core):  python tools/build_hex_surfaces.py <outdir> [orientation ...]
(default: all three; a surface whose hex_1x1_300K.dat exists is only summarized)
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
from lammps import lammps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo, if not installed
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate
from diamond_etch_md.analysis.cna import DIAMOND, diamond_cna, read_lammps_data
from diamond_etch_md.orientations import ORIENT


def build(o, outdir):
    bx, by, bz = ORIENT[o]["default_box"]
    spec = SimSpec(orientation=o, surface="1x1", box_x=bx, box_y=by, box_depth=bz,
                   ml=compute_ml(o, bx, by), species="Ar", energy=50.0, name=f"hexsurf_{o}")
    validate(spec)
    d = outdir / o
    make_sim(spec, d)
    p = d / "make_surf.lmp"
    p.write_text(re.sub(r"^package\s+kokkos.*$", "", p.read_text(), flags=re.M))
    (d / "impact_snaps").mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        L = lammps(cmdargs=["-log", "log_make_surf.lammps", "-screen", "none", "-nocite"])
        L.file("make_surf.lmp")
        L.close()
    finally:
        os.chdir(cwd)
    return d, spec


def summarize(d, spec):
    for name in ("hex_1x1_0K.dat", "hex_1x1_300K.dat"):
        data = read_lammps_data(d / name)
        x = np.column_stack([data["x"], data["y"], data["z"]])
        box = data["box"]; Lxy = np.array([box[0, 1] - box[0, 0], box[1, 1] - box[1, 0]])
        dd = x[:, None, :] - x[None, :, :]
        dd[..., :2] -= np.round(dd[..., :2] / Lxy) * Lxy
        r = np.linalg.norm(dd, axis=-1); np.fill_diagonal(r, np.inf)
        n = (r < 1.85).sum(axis=1)
        z = x[:, 2]
        top = z > z.max() - 1.0
        lab = diamond_cna(data)
        mid = (z > z.min() + 4) & (z < z.max() - 4)
        print(f"  {name}: {len(x)} C, ML {spec.ml}, top 1 Å: coordination "
              f"{dict(zip(*np.unique(n[top], return_counts=True)))}, "
              f"interior diamond {np.mean(lab[mid] == DIAMOND):.0%}, "
              f"all-slab diamond {np.mean(lab == DIAMOND):.0%}")


if __name__ == "__main__":
    outdir = Path(sys.argv[1]).resolve()
    for o in sys.argv[2:] or ("hex0001", "hex1120", "hex1010"):
        d = outdir / o
        if (d / "hex_1x1_300K.dat").exists():
            from diamond_etch_md.analysis.plot import _load_spec
            spec = _load_spec(d / "spec.json")
            print(f"{o}: already built, {d}", flush=True)
        else:
            d, spec = build(o, outdir)
            print(f"{o}: built, {d}", flush=True)
        summarize(d, spec)
