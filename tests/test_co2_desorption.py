"""
tests/test_co2_desorption.py — round-trip tests for CO2* site detection and removal.

These tests mirror the logic of co2_desorption.lmp with hand-crafted minimal geometries
where the correct outcome is exact and verifiable.

Notes on the LAMMPS Python API used here:
- pair_style zero 3.0  builds a neighbor list (cutoff > 1.8 coord/atom cutoff) with no forces
- extract_variable(name, None, 0)  reads an equal-style variable — NOT get_thermo(), which
  only reflects the last *printed* thermo line and returns 0 if run 0 never printed
- group NAME variable VARNAME  is a static (snapshot) group evaluated at command time,
  requiring the coord/atom compute to already be primed

Run with:
    /home/lh0534/.conda/envs/jaxmd/bin/python3 -m pytest tests/test_co2_desorption.py -m lammps -v
"""
import pytest

lammps_mod = pytest.importorskip("lammps", reason="LAMMPS Python module not available")
pytestmark = pytest.mark.lammps

# ---------------------------------------------------------------------------
# Geometry: a minimal CO2* site over one anchor carbon.
#
#   atom 1  anchor C   (5.0, 5.0, 5.00)  — bonded C-C to CO2*C, no O neighbours
#   atom 2  CO2* C     (5.0, 5.0, 6.50)  — 1.50 Å from anchor; 1.13 Å from O1 and O2
#   atom 3  excl O1    (4.2, 5.0, 7.30)  — 1.13 Å from CO2*C only (anchor is 2.44 Å away)
#   atom 4  excl O2    (5.8, 5.0, 7.30)  — 1.13 Å from CO2*C only
#
# coord/atom cutoff 1.8 Å:
#   coord_o(CO2*C)  = 2  →  goes into co2c_grp
#   coord_c(O1/O2)  = 1  →  both go into excl_o_snap
#   coord_o(anchor) = 0  →  anchor NOT a CO2* site
# ---------------------------------------------------------------------------
_SETUP_CO2_SITE = """\
units        real
atom_style   atomic
boundary     p p p

region       box block 0 20 0 20 0 20
create_box   3 box

mass  1 12.011
mass  2  1.008
mass  3 16.000

pair_style   zero 3.0
pair_coeff   * *

create_atoms 1 single  5.0 5.0 5.00
create_atoms 1 single  5.0 5.0 6.50
create_atoms 3 single  4.2 5.0 7.30
create_atoms 3 single  5.8 5.0 7.30
"""


def _lmp(log_path):
    return lammps_mod.lammps(cmdargs=["-log", str(log_path), "-screen", "none"])


def _ivar(L, expr):
    """Evaluate a LAMMPS expression and return it as int via a temporary variable."""
    L.command(f"variable _tmp equal {expr}")
    val = int(L.extract_variable("_tmp", None, 0))
    L.command("variable _tmp delete")
    return val


def _run_detection(L):
    """Prime coord/atom and build co2c_grp + excl_o_snap.

    Mirrors co2_desorption.lmp: the excl_o_snap group (accessing coord_c) is created
    FIRST so that the neighbour list is live when co2c_grp (accessing coord_o) is built.
    """
    L.commands_string("""\
compute      coord_o all coord/atom cutoff 1.8 3
compute      coord_c all coord/atom cutoff 1.8 1
run          0

variable     is_excl_o atom "type==3 && c_coord_c==1"
group        excl_o_snap variable is_excl_o

variable     is_co2c atom "type==1 && c_coord_o>=2"
group        co2c_grp variable is_co2c
""")


def _delete_orphan_o(L):
    """Post-deletion orphan-O cleanup.  Re-primes coord_c, intersects with excl_o_snap."""
    L.commands_string("""\
run          0

variable     is_no_c_o atom "type==3 && c_coord_c==0"
group        no_c_o   variable is_no_c_o
group        orphan_o intersect excl_o_snap no_c_o
""")
    n_orphan = _ivar(L, "count(orphan_o)")
    if n_orphan > 0:
        L.command("delete_atoms group orphan_o")
    return n_orphan


# ---------------------------------------------------------------------------
# Test 1: detection
# ---------------------------------------------------------------------------

def test_co2_site_detected(tmp_path):
    """coord/atom + group variable must identify exactly 1 CO2* C and 2 exclusive Os."""
    L = _lmp(tmp_path / "log.lammps")
    try:
        L.commands_string(_SETUP_CO2_SITE)
        _run_detection(L)
        n_co2c = _ivar(L, "count(co2c_grp)")
        n_excl_o = _ivar(L, "count(excl_o_snap)")
    finally:
        L.close()

    assert n_co2c == 1, f"Expected 1 CO2* C, detected {n_co2c}"
    assert n_excl_o == 2, f"Expected 2 exclusive Os, detected {n_excl_o}"


# ---------------------------------------------------------------------------
# Test 2: full removal round-trip
# ---------------------------------------------------------------------------

def test_co2_site_and_orphan_o_removed(tmp_path):
    """With fraction=1.0, CO2*C and both orphan Os are deleted; anchor C survives."""
    L = _lmp(tmp_path / "log.lammps")
    try:
        L.commands_string(_SETUP_CO2_SITE)
        _run_detection(L)
        L.command("delete_atoms random fraction 1.0 no co2c_grp NULL 12345 compress no")
        n_orphan = _delete_orphan_o(L)
        n_remaining = L.get_natoms()
    finally:
        L.close()

    assert n_orphan == 2, f"Expected 2 orphan Os queued for deletion, got {n_orphan}"
    assert n_remaining == 1, (
        f"Only the anchor C should remain (1 atom), but {n_remaining} atoms present"
    )


# ---------------------------------------------------------------------------
# Test 3: bridging O is not deleted
# ---------------------------------------------------------------------------

def test_bridging_o_not_deleted(tmp_path):
    """An O bonded to two C atoms (coord_c=2) must NOT enter excl_o_snap and must
    survive even after the adjacent CO2* C is desorbed.

    Extra atom added to the CO2* geometry:
        atom 3  bridge O  (5.0, 5.0, 5.75) — 0.75 Å from both anchor and CO2*C → coord_c=2
        atom 4  excl O1   (4.2, 5.0, 7.30) — exclusive to CO2*C
        atom 5  excl O2   (5.8, 5.0, 7.30) — exclusive to CO2*C

    After desorption: excl O1 and O2 deleted; anchor C and bridge O survive (2 atoms).
    """
    L = _lmp(tmp_path / "log.lammps")
    try:
        L.commands_string("""\
units        real
atom_style   atomic
boundary     p p p

region       box block 0 20 0 20 0 20
create_box   3 box

mass  1 12.011
mass  2  1.008
mass  3 16.000

pair_style   zero 3.0
pair_coeff   * *

create_atoms 1 single  5.0 5.0 5.00
create_atoms 1 single  5.0 5.0 6.50
create_atoms 3 single  5.0 5.0 5.75
create_atoms 3 single  4.2 5.0 7.30
create_atoms 3 single  5.8 5.0 7.30
""")
        _run_detection(L)

        n_co2c = _ivar(L, "count(co2c_grp)")
        n_excl_o = _ivar(L, "count(excl_o_snap)")
        assert n_co2c == 1, f"CO2*C not detected (n={n_co2c})"
        assert n_excl_o == 2, (
            f"Bridge O (coord_c=2) must not be in excl_o_snap — expected 2, got {n_excl_o}"
        )

        L.command("delete_atoms random fraction 1.0 no co2c_grp NULL 99991 compress no")
        n_orphan = _delete_orphan_o(L)
        n_remaining = L.get_natoms()
    finally:
        L.close()

    assert n_orphan == 2, f"Expected 2 exclusive Os as orphans, got {n_orphan}"
    assert n_remaining == 2, (
        f"Anchor C and bridge O should both survive (2 atoms), got {n_remaining}"
    )


# ---------------------------------------------------------------------------
# Tests 4-6: run the real co2_desorption.lmp template via `include`
# ---------------------------------------------------------------------------
from pathlib import Path

_TEMPLATE = (Path(__file__).resolve().parents[1]
             / "diamond_etch_md" / "lammps" / "templates" / "co2_desorption.lmp")

# Names head.lmp defines that the template relies on: loop counters c/cn, the
# desorption fraction, and the thermo_style columns it restores when done.
_HEAD_STUBS = """\
variable     c equal 7
variable     cn equal 3
variable     co2_desorb_fraction equal {fraction}
variable     ninject equal 0
variable     ate equal 0
variable     one_clust equal 0
compute      ike all ke
compute      nclusts all ke
"""

_GAS_O_RADICAL = "create_atoms 3 single 15.0 15.0 15.0\n"   # isolated O, coord_c = 0


def _run_template(tmp_path, monkeypatch, setup, fraction=1.0):
    """Build `setup`, include the real template, then `run 0` so the thermo_style
    it restores is exercised. Returns (natoms, {output filename: text})."""
    monkeypatch.chdir(tmp_path)
    L = _lmp(tmp_path / "log.lammps")
    try:
        L.commands_string(setup)
        L.commands_string(_HEAD_STUBS.format(fraction=fraction))
        L.command(f"include {_TEMPLATE}")
        L.command("run 0 post no")
        natoms = L.get_natoms()
    finally:
        L.close()
    outputs = {p.name: p.read_text() for p in tmp_path.iterdir()
               if p.name in ("co2_desorption.txt", "etch_products.txt")}
    return natoms, outputs


def test_template_desorbs_co2_site(tmp_path, monkeypatch):
    """Real template, fraction=1.0: the CO2* C and its two exclusive Os are removed,
    the anchor C survives, and both log files record 1 C + 2 O at (c, cn) = (7, 3)."""
    natoms, out = _run_template(tmp_path, monkeypatch, _SETUP_CO2_SITE)

    assert natoms == 1, f"Only the anchor C should remain, got {natoms} atoms"
    assert out.get("co2_desorption.txt", "").split() == ["7", "3", "1", "2"]
    assert out.get("etch_products.txt", "").split() == ["7", "3", "1", "0", "2", "0", "0.0"]


def test_template_spares_gas_phase_o_radical(tmp_path, monkeypatch):
    """A free O radical has coord_c == 0 after desorption, like the orphaned Os, but it
    was never in the pre-deletion exclusive-O snapshot, so it must survive."""
    natoms, out = _run_template(tmp_path, monkeypatch, _SETUP_CO2_SITE + _GAS_O_RADICAL)

    assert natoms == 2, f"Anchor C and the gas-phase O should remain, got {natoms} atoms"
    assert out.get("co2_desorption.txt", "").split() == ["7", "3", "1", "2"]


def test_template_no_site_is_noop(tmp_path, monkeypatch):
    """With no CO2* site (just a lone C and a gas-phase O), nothing is deleted and
    nothing is logged."""
    setup = _SETUP_CO2_SITE.split("create_atoms")[0] + (
        "create_atoms 1 single  5.0 5.0 5.00\n" + _GAS_O_RADICAL)
    natoms, out = _run_template(tmp_path, monkeypatch, setup)

    assert natoms == 2, f"No atoms should be deleted, got {natoms}"
    assert out == {}, f"Nothing should be logged, got {sorted(out)}"
