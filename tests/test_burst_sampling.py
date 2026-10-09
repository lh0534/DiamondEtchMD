"""
tests/test_burst_sampling.py — run burst-mode radical insertion in real LAMMPS.

Deposit harness (pair_style zero, any LAMMPS build): the deposit section that
get_head_lmp() generates for each burst chunk is cut out of the head and run on a
small carbon slab, many impacts in a row.  For every inserted O atom:
  - its kinetic energy (compute ke/atom) equals the logged draw, and its polar
    angle the logged angle — this fails if the insertion uses fix deposit's
    default lattice units, which scale vx/vy/vz by the lattice spacings;
  - all atoms of a chunk sit in one ±0.1 Å slab, at least radical_burst_near
    apart, and no simulated time passes while the chunk is inserted;
  - draws logged as skipped (below radical_energy_threshold) are not inserted.
Over all draws: flux mean 2kT, gas mean 1.5kT, Lambert <cos θ> = 2/3, and the
skipped fraction 1-(1+x)e^-x with x = threshold/kT.

End-to-end (ReaxFF, LAMMPS 23 Jun 2022 only — production's sweep.lmp uses
atom-style formulas newer builds reject): a tiny RIE and a tiny cycling sim are
built with make_sim/make_ale and run for two burst slots on the CPU.

Run with (myenv has LAMMPS 23 Jun 2022 with REAXFF):
    /home/lh0534/.conda/envs/myenv/bin/python -m pytest tests/test_burst_sampling.py -m lammps -v
"""
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

from diamond_etch_md.spec import SimSpec, CyclePhase, validate

lammps_mod = pytest.importorskip("lammps", reason="LAMMPS Python module not available — run with myenv")
pytestmark = pytest.mark.lammps

KB = 8.617333262e-5          # eV/K
KCAL_PER_EV = 23.060548      # kcal/mol per eV
M_O = 15.9994


def _lmp(log):
    return lammps_mod.lammps(cmdargs=["-log", str(log), "-screen", "none", "-nocite"])


def _lammps_version(L):
    return L.version()


# ---------------------------------------------------------------------------
# Deposit harness
# ---------------------------------------------------------------------------

def _burst_spec(**kw):
    base = dict(orientation="100", surface="2x1", box_x=6, box_y=6, ml=36, species="O",
                energy=20.0, flux_ratio=13, radical_burst=True, radical_burst_chunk=10,
                radical_burst_near=3.5, dump_mode="none")
    spec = SimSpec(**{**base, **kw})
    validate(spec)
    return spec


def _deposit_sections(spec):
    """Generated deposit section of each burst chunk (between the chunk's deposit
    and dynamics headers), with the all-skipped jump target appended."""
    from diamond_etch_md.lammps.head import get_head_lmp
    head = get_head_lmp(spec)
    starts = [m.start() for m in re.finditer(r"# --- Burst chunk \d+/\d+: \d+ atoms \(deposit phase\) ---", head)]
    out = []
    for ci, a in enumerate(starts):
        b = head.index("dynamics phase ---", a)
        b = head.rindex("\n", 0, b) + 1
        out.append(head[a:b] + f"label       burst_chunk_done_{ci}\n")
    return out


def _setup(L, spec, temperature=None, energy=0.2, angle=(0.0, 90.0), thresh=None):
    for cmd in [
        "units real", "atom_style charge", "boundary p p m",
        "lattice diamond 3.5654 orient z 0 0 1 orient x 1 1 0 orient y -1 1 0 "
        "spacing $(1/sqrt(2)) $(1/sqrt(2)) 1 origin 0 0 0",
        f"region box block 0 {spec.box_x} 0 {spec.box_y} -1 14 units lattice",
        "create_box 3 box",
        f"region slab block 0 {spec.box_x} 0 {spec.box_y} 0 0.9 units lattice",
        "create_atoms 1 region slab",
        "mass 1 12.011", "mass 2 1.008", f"mass 3 {M_O}",
        "pair_style zero 6.0", "pair_coeff * *",
        "region bbox block EDGE EDGE EDGE EDGE EDGE EDGE",
        "group anchor type 1", "group insert empty", "group mobile subtract all anchor",
        "compute kea all ke/atom",
        "thermo_style custom step time", "timestep 1e-10",
        "variable seed_adjust equal 0", f"variable M_O equal {M_O}",
        f"variable radical_i_above equal {spec.radical_i_above}",
        f"variable radical_energy equal {energy}",
        f"variable rad_angl equal {angle[0]}", f"variable rad_azimuth equal {angle[1]}",
    ]:
        L.command(cmd)
    if temperature is not None:
        L.command(f"variable kT_rad equal {KB * temperature}")
    if thresh is not None:
        L.command(f"variable radical_e_thresh equal {thresh}")


def _atoms(L):
    n = L.get_natoms()
    ids = np.array([int(L.extract_atom("id")[i]) for i in range(n)])
    typ = np.array([int(L.extract_atom("type")[i]) for i in range(n)])
    x = np.array([[L.extract_atom("x")[i][k] for k in range(3)] for i in range(n)])
    v = np.array([[L.extract_atom("v")[i][k] for k in range(3)] for i in range(n)])
    L.command("run 0 post no")
    ke = np.array([L.extract_compute("kea", 1, 1)[i] for i in range(n)])
    order = np.argsort(ids)
    return ids[order], typ[order], x[order], v[order], ke[order]


def _run_bursts(tmp_path, spec, n_impacts, **setup_kw):
    """Run every chunk's generated deposit section for n_impacts impacts.

    Returns (log rows, per-impact list of inserted atoms per chunk)."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return _run_bursts_in_cwd(tmp_path, spec, n_impacts, **setup_kw)
    finally:
        os.chdir(cwd)


def _run_bursts_in_cwd(tmp_path, spec, n_impacts, **setup_kw):
    sections = _deposit_sections(spec)
    for ci, sec in enumerate(sections):
        (tmp_path / f"chunk{ci}.lmp").write_text(sec)
    L = _lmp(tmp_path / "log.lammps")
    _setup(L, spec, **setup_kw)
    box_xy = (L.extract_box()[1][0] - L.extract_box()[0][0],
              L.extract_box()[1][1] - L.extract_box()[0][1])
    inserted = []
    for c in range(1, n_impacts + 1):
        L.command(f"variable c equal {c}")
        L.command("variable n_burst_sim equal 0")
        L.command("variable n_burst_fail equal 0")
        per_chunk = []
        for ci in range(len(sections)):
            t0 = L.get_thermo("time")
            n0 = L.get_natoms()
            ids0 = set(_atoms(L)[0])
            L.file(str(tmp_path / f"chunk{ci}.lmp"))
            assert L.get_thermo("time") - t0 < 1e-6      # no dynamics while inserting
            ids, typ, x, v, ke = _atoms(L)
            new = np.array([i not in ids0 for i in ids])
            per_chunk.append(dict(x=x[new], v=v[new], ke=ke[new], typ=typ[new]))
            assert L.get_natoms() - n0 == new.sum()
        assert L.extract_variable("n_burst_fail", None, 0) == 0
        inserted.append(per_chunk)
        L.command("group oxy type 3")
        L.command("delete_atoms group oxy")
        L.command("group oxy delete")
    L.close()
    rows = np.loadtxt(tmp_path / "radical_log.txt", ndmin=2)
    return rows, inserted, box_xy


def _check_inserted_match_log(rows, inserted, spec, box_xy):
    """Inserted atoms carry exactly the logged (simulated) draws, in one layer per chunk."""
    for c, per_chunk in enumerate(inserted, start=1):
        r = rows[rows[:, 0] == c]
        assert sorted(r[:, 1].astype(int)) == list(range(1, round(spec.flux_ratio) + 1))
        sim = r[r[:, 5] > 0.5]
        ke = np.concatenate([ch["ke"] for ch in per_chunk])
        v = np.concatenate([ch["v"] for ch in per_chunk])
        assert len(ke) == len(sim)
        np.testing.assert_allclose(ke / KCAL_PER_EV, sim[:, 2], rtol=2e-3)
        speed = np.linalg.norm(v, axis=1)
        polar = np.degrees(np.arccos(np.clip(-v[:, 2] / speed, -1, 1)))
        np.testing.assert_allclose(polar, sim[:, 3], atol=1e-3)
        assert (v[:, 2] < 0).all()
        for ch in per_chunk:
            if len(ch["x"]) == 0:
                continue
            assert (ch["typ"] == 3).all()
            assert np.ptp(ch["x"][:, 2]) <= 0.2 + 1e-9                 # one slab
            d = ch["x"][:, None, :] - ch["x"][None, :, :]
            d[..., :2] -= np.round(d[..., :2] / box_xy) * box_xy
            dist = np.linalg.norm(d, axis=-1) + np.eye(len(ch["x"])) * 1e9
            assert dist.min() >= spec.radical_burst_near - 1e-6       # near is 3-D, in Å


def test_burst_mono_off_normal_energy_and_angle(tmp_path):
    """Fixed energy at 30° off normal: catches lattice-unit scaling of vx/vy/vz."""
    spec = _burst_spec(radical_energy=0.0893, radical_angle=(30.0, 45.0))
    rows, inserted, box_xy = _run_bursts(tmp_path, spec, 5, energy=0.0893, angle=(30.0, 45.0))
    _check_inserted_match_log(rows, inserted, spec, box_xy)
    np.testing.assert_allclose(rows[:, 2], 0.0893, rtol=1e-9)
    np.testing.assert_allclose(rows[:, 3], 30.0, atol=1e-6)
    v = np.concatenate([ch["v"] for imp in inserted for ch in imp])
    np.testing.assert_allclose(np.degrees(np.arctan2(v[:, 1], v[:, 0])), 45.0, atol=1e-6)


def test_burst_flux_cosine_threshold_statistics(tmp_path):
    T, thr = 518.0, 0.1
    spec = _burst_spec(radical_temperature=T, radical_angle_distribution=True,
                       radical_energy_threshold=thr)
    rows, inserted, box_xy = _run_bursts(tmp_path, spec, 150, temperature=T, thresh=thr)
    _check_inserted_match_log(rows, inserted, spec, box_xy)
    kT = KB * T
    n = len(rows)
    assert abs(rows[:, 2].mean() / (2 * kT) - 1) < 4 * math.sqrt(2) / math.sqrt(n) / 2 + 0.01
    x = thr / kT
    p_skip = 1 - (1 + x) * math.exp(-x)
    skipped = rows[:, 5] < 0.5
    assert abs(skipped.mean() - p_skip) < 4 * math.sqrt(p_skip * (1 - p_skip) / n)
    assert (rows[skipped, 2] < thr).all() and (rows[~skipped, 2] >= thr).all()
    cos_th = np.cos(np.radians(rows[:, 3]))
    assert abs(cos_th.mean() - 2 / 3) < 4 * 0.236 / math.sqrt(n)


def test_burst_gas_distribution_mean(tmp_path):
    T = 1000.0
    spec = _burst_spec(radical_temperature=T, radical_energy_dist="gas")
    rows, inserted, box_xy = _run_bursts(tmp_path, spec, 80, temperature=T)
    _check_inserted_match_log(rows, inserted, spec, box_xy)
    kT = KB * T
    n = len(rows)
    # gas-phase MB energy ~ Gamma(3/2, kT): mean 1.5 kT, sd sqrt(1.5) kT
    assert abs(rows[:, 2].mean() - 1.5 * kT) < 4 * math.sqrt(1.5) * kT / math.sqrt(n)
    np.testing.assert_allclose(rows[:, 3], 0.0, atol=1e-6)          # fixed normal incidence


# ---------------------------------------------------------------------------
# End-to-end ReaxFF runs (LAMMPS 23 Jun 2022)
# ---------------------------------------------------------------------------

def _needs_2022_build():
    L = _lmp("none")
    ver, pkgs = L.version(), L.installed_packages
    L.close()
    if "REAXFF" not in pkgs:
        pytest.skip("LAMMPS build without REAXFF")
    if ver > 20221231:
        pytest.skip(f"LAMMPS {ver}: sweep.lmp needs the 23 Jun 2022 build (use myenv)")


def _set_var(path, name, value):
    s = path.read_text()
    s, n = re.subn(rf"^variable\s+{name}\s+equal\s+\S+", f"variable    {name} equal {value}", s, flags=re.M)
    assert n == 1, name
    path.write_text(s)


def _run_sim(d):
    """make_surf.lmp, then head.lmp from scratch, the way submit runs them (CPU)."""
    for f in ("make_surf.lmp", "head.lmp"):
        p = d / f
        p.write_text(re.sub(r"^package\s+kokkos.*$", "", p.read_text(), flags=re.M))
    (d / "impact_snaps").mkdir(exist_ok=True)
    (d / "etch_event_trajs").mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        return _run_sim_in_cwd(d)
    finally:
        os.chdir(cwd)


def _run_sim_in_cwd(d):
    L = _lmp(d / "log_make_surf.lammps")
    L.file("make_surf.lmp")
    L.close()
    n_lat_0 = next(l.split()[0] for l in (d / "impact_snaps/0.data").read_text().splitlines()
                   if l.strip().endswith(" atoms"))
    L = lammps_mod.lammps(cmdargs=[
        "-log", "log1.lammps", "-screen", "none", "-nocite",
        "-var", "data_file", "impact_snaps/0.data", "-var", "n_complete", "0",
        "-var", "neut_complete", "0", "-var", "log_file", "log1.lammps",
        "-var", "n_events", "0", "-var", "n_lat_0", n_lat_0])
    L.file("head.lmp")
    L.close()
    nc = np.loadtxt(d / "ncarbon.txt", ndmin=2)
    rows = np.loadtxt(d / "radical_log.txt", ndmin=2)
    return nc, rows


def _check_e2e(d, nc, rows, n_bursts, R):
    assert not (d / "BURST_INCOMPLETE").exists()
    assert "ERROR" not in (d / "log1.lammps").read_text()
    bursts = sorted(set(rows[:, 0].astype(int)))
    assert len(bursts) == n_bursts
    for c in bursts:
        r = rows[rows[:, 0] == c]
        assert sorted(r[:, 1].astype(int)) == list(range(1, R + 1))
        k = np.flatnonzero((nc[:, 0] == c) & (nc[:, 1] == 1))
        assert len(k) == 1                                  # one ncarbon row per burst
        # O gained over the burst can't exceed the simulated (inserted) draws
        assert nc[k[0], 4] - nc[k[0] - 1, 4] <= (r[:, 5] > 0.5).sum()


def test_e2e_rie_burst_sampled(tmp_path):
    _needs_2022_build()
    from diamond_etch_md import make_sim
    spec = SimSpec(orientation="100", surface="2x1", box_x=4, box_y=4, box_depth=2, ml=16,
                   species="O", energy=20.0, flux_ratio=6, fluence=1,
                   radical_burst=True, radical_burst_chunk=4,
                   radical_temperature=518, radical_angle_distribution=True,
                   radical_energy_threshold=0.05,
                   impact_time=200.0, thermalization_time=100.0, inter_neutral_time=200.0,
                   skip_radical_thermalization=True, dump_mode="none", name="t_burst_rie")
    validate(spec)
    d = tmp_path / "rie"
    make_sim(spec, d)
    _set_var(d / "config.lmp", "end_fluence", 0.125)          # 2 ion slots
    nc, rows = _run_sim(d)
    _check_e2e(d, nc, rows, n_bursts=2, R=6)


def test_e2e_cycle_burst_sampled(tmp_path):
    _needs_2022_build()
    from diamond_etch_md import make_ale
    spec = SimSpec(orientation="100", surface="2x1", box_x=4, box_y=4, box_depth=2, ml=16,
                   impact_time=200, thermalization_time=100, skip_radical_thermalization=True,
                   phases=[CyclePhase(species="Ar", energy=50, fluence_ml=1),
                           CyclePhase(species="O", energy=15, fluence_ml=1, flux_ratio=6,
                                      radical_burst=True, radical_burst_chunk=4,
                                      radical_temperature=518, radical_angle_distribution=True,
                                      radical_energy_threshold=0.05)],
                   cycles=1, dump_mode="none", name="t_burst_cyc")
    validate(spec)
    d = tmp_path / "cyc"
    make_ale(spec, d)
    cfg = d / "config.lmp"
    _set_var(cfg, "phase_0_ml", 0.0625)                       # 1 Ar impact
    _set_var(cfg, "phase_1_ml", 0.125)                        # 2 O slots with bursts
    _set_var(cfg, "end_fluence", 0.1875)
    nc, rows = _run_sim(d)
    _check_e2e(d, nc, rows, n_bursts=2, R=6)
