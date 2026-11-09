"""
tests/test_product_cleanup_lammps.py — product_cleanup.lmp and the slab-only
thermostat in real LAMMPS (pair_style zero: both are geometric, no chemistry).

  1. product_cleanup.lmp on a diamond(100) slab with an exposed CO2 2.6 Å above the
     surface (removed, logged cn = 0 / flag 2), a CO bonded to a surface atom, a CO2
     trapped in a subsurface void and an Ar atom above the surface (all kept).
  2. thermalize.lmp: with thermostat_slab_only = 1 a molecule not bonded to the slab
     keeps its velocity; with 0 the Berendsen thermostat rescales it.

Run with:
    /home/lh0534/.conda/envs/myenv/bin/python -m pytest tests/test_product_cleanup_lammps.py -m lammps -v
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

lammps_mod = pytest.importorskip("lammps", reason="LAMMPS Python module not available — run with myenv")
pytestmark = pytest.mark.lammps

A = 3.567
TOP = 3 * A                                     # top atomic plane of the slab (z = 3 lattice units)
TEMPLATES = Path(__file__).parents[1] / "diamond_etch_md" / "lammps" / "templates"


def _setup(tmp_path, monkeypatch, slab_only=1):
    for f in ("product_cleanup.lmp", "thermalize.lmp", "sweep.lmp"):
        shutil.copy(TEMPLATES / f, tmp_path / f)
    monkeypatch.chdir(tmp_path)
    L = lammps_mod.lammps(cmdargs=["-log", "log.lammps", "-screen", "none", "-nocite"])
    for cmd in [
        "units real", "atom_style charge", "boundary p p m",
        f"lattice diamond {A}",
        "region box block 0 4 0 4 -1 8", "create_box 4 box",
        "region slab block 0 4 0 4 0 3", "create_atoms 1 region slab",
        "mass 1 12.011", "mass 2 1.008", "mass 3 16.0", "mass 4 39.948",
        "pair_style zero 3.5", "pair_coeff * *",
        "region bbox block EDGE EDGE EDGE EDGE EDGE EDGE",
        "region anchor block INF INF INF INF 0 0.5 units lattice",
        "group anchor region anchor", "group insert empty",
        "group mobile subtract all anchor", "group carbon type 1",
        "variable c equal 1", "variable cn equal 0",
        "variable cleanup_rc equal 2.0", "variable cleanup_cap equal 2.0",
        f"variable thermostat_slab_only equal {slab_only}",
        "variable T equal 300.0", "variable thermalization_time equal 50.0",
        "variable above_surf_eject equal 1", "variable n_c_ejected_total equal 0",
        # what head.lmp defines and the templates' thermo_style reset refers to
        "variable ninject equal count(insert)",
        "compute ake all ke", "compute ape all pe", "compute ike insert ke",
        "variable ate equal c_ake+c_ape+ecouple",
        "compute clusts all cluster/atom 3.0",
        "compute clust_min all reduce min c_clusts", "compute clust_max all reduce max c_clusts",
        'variable one_clust equal "c_clust_min == c_clust_max"',
        "compute cclusts carbon cluster/atom 1.8",
        "compute cc1 carbon chunk/atom c_cclusts compress yes",
        "compute nclusts carbon reduce max c_cc1",
        "compute mtemp mobile temp", "compute_modify mtemp dynamic/dof yes",
        "thermo_style custom step time v_ninject temp c_ike dt ecouple v_ate v_one_clust c_nclusts",
        "thermo_modify temp mtemp", "thermo_modify lost warn flush yes",
    ]:
        L.command(cmd)
    return L


def _add(L, typ, pos):
    L.command(f"create_atoms {typ} single {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} units box")


def _types(L):
    n = L.get_natoms()
    t = L.extract_atom("type")
    return np.array([t[i] for i in range(n)])


def test_cleanup_removes_only_exposed_unbonded_molecules(tmp_path, monkeypatch):
    L = _setup(tmp_path, monkeypatch)
    # subsurface void with a trapped CO2 (nearest slab atom > 2.4 Å away, slab above it)
    void = (2 * A, 2 * A, 1.5 * A)
    L.command(f"region void sphere {void[0]} {void[1]} {void[2]} 3.6 units box")
    L.command("delete_atoms region void compress no")
    for dx, t in ((-1.16, 3), (0.0, 1), (1.16, 3)):
        _add(L, t, (void[0] + dx, void[1], void[2]))
    # CO bonded to the top-layer atom at (a, a, 3a)
    _add(L, 1, (A, A, TOP + 1.5)); _add(L, 3, (A, A, TOP + 2.65))
    # exposed CO2 2.6 Å above the surface, far from the CO
    ex = (3 * A, 2.5 * A, TOP + 2.6)
    for dx, t in ((-1.16, 3), (0.0, 1), (1.16, 3)):
        _add(L, t, (ex[0] + dx, ex[1], ex[2]))
    _add(L, 4, (0.6 * A, 3.2 * A, TOP + 3.8))          # Ar above the surface
    L.command("velocity all set 0.0 0.0 0.0")
    L.command("run 0 post no")
    n0, t0 = L.get_natoms(), _types(L)
    L.file("product_cleanup.lmp")
    t1 = _types(L)
    assert L.get_natoms() == n0 - 3                     # only the exposed CO2
    assert (t1 == 3).sum() == (t0 == 3).sum() - 2 and (t1 == 4).sum() == 1
    rows = [l.split() for l in Path("etch_products.txt").read_text().splitlines()]
    assert len(rows) == 1
    c, cn, nC, nH, nO, nAr, vz, flag = rows[0]
    assert (c, cn, nC, nH, nO, nAr, flag) == ("1", "0", "1", "0", "2", "0", "2")
    assert "PRODUCT_CLEANUP 1 0: removed 1" in Path("log.lammps").read_text()
    L.command("run 0 post no")                          # thermo_style restored, computes valid
    L.close()


def test_cleanup_with_nothing_detached(tmp_path, monkeypatch):
    L = _setup(tmp_path, monkeypatch)
    L.command("run 0 post no")
    n0 = L.get_natoms()
    L.file("product_cleanup.lmp")
    assert L.get_natoms() == n0 and not Path("etch_products.txt").exists()
    L.close()


@pytest.mark.parametrize("slab_only", [1, 0])
def test_thermostat_skips_unbonded_molecules(slab_only, tmp_path, monkeypatch):
    L = _setup(tmp_path, monkeypatch, slab_only)
    ex = (2 * A, 2 * A, TOP + 2.6)
    for dx, t in ((-1.16, 3), (0.0, 1), (1.16, 3)):
        _add(L, t, (ex[0] + dx, ex[1], ex[2]))
    L.command(f"region molr block INF INF INF INF {TOP + 1.5} INF units box")
    L.command("group mol region molr")
    L.command("group mobile clear")
    L.command("group mobile subtract all anchor")
    L.command("group carbon clear")
    L.command("group carbon type 1")                     # as head.lmp keeps it: all C
    L.command("velocity mobile create 600.0 4928459 dist gaussian")  # hot: the thermostat must act
    L.command("velocity mol set 0.0 0.0 0.002 units box")   # leaving slowly (2 Å per ps)
    L.command("fix 2 mobile nve")
    L.command("run 0 post no")
    L.file("thermalize.lmp")
    L.command("variable vzm equal vcm(mol,z)")
    vz = L.extract_variable("vzm", None, 0)
    L.close()
    if slab_only:
        assert vz == pytest.approx(0.002, rel=1e-6)     # no forces, no thermostat
    else:
        assert vz < 0.00198                             # its C is rescaled with the slab toward 300 K


# ─── end to end (LAMMPS 23 Jun 2022 with REAXFF, through head.lmp) ────────────

def _run_head(d, patch=None):
    """make_surf.lmp, optionally patch impact_snaps/0.data, then head.lmp (CPU)."""
    import os, re
    from tests.test_burst_sampling import _lmp
    for f in ("make_surf.lmp", "head.lmp"):
        p = d / f
        p.write_text(re.sub(r"^package\s+kokkos.*$", "", p.read_text(), flags=re.M))
    (d / "impact_snaps").mkdir(exist_ok=True)
    (d / "etch_event_trajs").mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        L = _lmp(d / "log_make_surf.lammps")
        L.file("make_surf.lmp")
        L.close()
        n_lat_0 = next(l.split()[0] for l in (d / "impact_snaps/0.data").read_text().splitlines()
                       if l.strip().endswith(" atoms"))
        if patch:
            patch(d / "impact_snaps/0.data")
        L = lammps_mod.lammps(cmdargs=[
            "-log", "log1.lammps", "-screen", "none", "-nocite",
            "-var", "data_file", "impact_snaps/0.data", "-var", "n_complete", "0",
            "-var", "neut_complete", "0", "-var", "log_file", "log1.lammps",
            "-var", "n_events", "0", "-var", "n_lat_0", n_lat_0])
        L.file("head.lmp")
        L.close()
    finally:
        os.chdir(cwd)


def _add_co2_above(path, gap=2.8):
    """Rewrite a slab data file with a CO2 at rest `gap` Å above its highest atom."""
    L = lammps_mod.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    for cmd in ["units real", "atom_style charge", "boundary p p m",
                "region dummy block 0 1 0 1 0 1", "create_box 4 dummy",
                f"read_data {path} add merge", "mass * 12.011", "mass 3 16.0", "mass 4 39.948",
                # immediate ($()): each created atom would otherwise raise bound(all,zmax)
                "variable xc equal $((xlo+xhi)/2)", "variable yc equal $((ylo+yhi)/2)",
                f"variable zc equal $(bound(all,zmax)+{gap})",
                "change_box all z final $(zlo) $(v_zc+10) units box"]:
        L.command(cmd)
    for dx, t in ((-1.16, 3), (0.0, 1), (1.16, 3)):
        L.command(f"create_atoms {t} single $(v_xc+{dx}) ${{yc}} ${{zc}} units box")
    L.command(f"write_data {path} nocoeff")
    L.close()


def test_e2e_physisorbed_co2_logged_as_ion_product(tmp_path):
    """A CO2 resting on the surface through an ion stage is logged with that ion
    (cn = 0), never with a later radical.  Radicals all fall below the energy
    threshold, so no radical MD runs and nothing may be logged with cn > 0."""
    from tests.test_burst_sampling import _needs_2022_build, _set_var
    from diamond_etch_md import SimSpec, make_sim, validate
    from diamond_etch_md.analysis.etch_products import parse_etch_products
    _needs_2022_build()
    spec = SimSpec(orientation="100", surface="2x1", box_x=4, box_y=4, box_depth=2, ml=16,
                   species="Ar", energy=20.0, flux_ratio=2, fluence=1,
                   radical_temperature=518, radical_energy_threshold=50.0,
                   impact_time=100.0, thermalization_time=100.0, dump_mode="none",
                   name="t_cleanup_co2")
    validate(spec)
    d = tmp_path / "co2"
    make_sim(spec, d)
    assert "include     product_cleanup.lmp" in (d / "head.lmp").read_text()
    _set_var(d / "config.lmp", "end_fluence", 0.125)          # 2 ions
    _run_head(d, patch=_add_co2_above)
    log = (d / "log1.lammps").read_text()
    assert "ERROR" not in log
    rows = parse_etch_products(d / "etch_products.txt")
    assert rows and all(r["cn"] == 0 for r in rows)
    co2 = [r for r in rows if (r["n_C"], r["n_O"]) == (1, 2)]
    assert len(co2) == 1 and co2[0]["impact"] == 1             # with the first ion
    assert co2[0]["removal"] in (0, 2)
    assert len((d / "etch_products.txt").read_text().split("\n")[0].split()) == 8
    assert "group       th_c intersect carbon th_slab" in (d / "therm.log").read_text()


def test_e2e_rie_with_cleanup_runs(tmp_path):
    """Ordinary RIE (Ar+ 100 eV, O radicals at 0.2 eV): the cleanup and the slab-only
    thermostat run in every ion stage without errors; flag-2 rows are ion products."""
    from tests.test_burst_sampling import _needs_2022_build, _set_var
    from diamond_etch_md import SimSpec, make_sim, validate
    from diamond_etch_md.analysis.etch_products import parse_etch_products
    _needs_2022_build()
    # depth 3: addfix's new layer stays a small fraction of the slab (ReaxFF in the 2022
    # build stops with "Too many ghost atoms" when the atom count jumps ~20% at once)
    spec = SimSpec(orientation="100", surface="2x1", box_x=4, box_y=4, box_depth=3, ml=16,
                   species="Ar", energy=100.0, flux_ratio=2, fluence=1, radical_energy=0.2,
                   impact_time=300.0, thermalization_time=100.0, inter_neutral_time=300.0,
                   dump_mode="none", name="t_cleanup_rie")
    validate(spec)
    d = tmp_path / "rie"
    make_sim(spec, d)
    _set_var(d / "config.lmp", "end_fluence", 0.1875)         # 3 ions
    _run_head(d)
    log = (d / "log1.lammps").read_text()
    assert "ERROR" not in log
    nc = np.loadtxt(d / "ncarbon.txt", ndmin=2)
    assert sorted({int(r[0]) for r in nc if r[1] == 0}) == [0, 1, 2, 3]
    ep = d / "etch_products.txt"
    if ep.exists():
        for line in ep.read_text().splitlines():
            assert len(line.split()) == 8
        assert all(r["cn"] == 0 for r in parse_etch_products(ep) if r["removal"] == 2)
