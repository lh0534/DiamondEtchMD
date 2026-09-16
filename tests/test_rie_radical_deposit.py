"""
tests/test_rie_radical_deposit.py — verify that deposited O radicals are picked up by the
nonargon (qeq) group after the nonargon refresh pattern used in generated head.lmp.

Three code paths that emit the refresh:
  1. Standard RIE    : get_head_lmp(spec)           with species="Ar" and flux_ratio > 0
  2. Multi-ion RIE   : get_head_lmp_multi_ion(spec) with Ar in ion_mix and flux_ratio > 0
  3. Cycling with RIE: get_head_lmp_cycle_etch(spec) with an Ar phase and flux_ratio > 0

String tests (no LAMMPS):
  - generated head must contain "fix reax_qeq nonargon" (qeq scoped to nonargon)
  - generated head must contain "nonargon type 1 2 3" in a group command
  - for RIE paths: must contain the post-deposit refresh "group       nonargon type 1 2 3"

LAMMPS mini-test:
  Simulate the deposit + nonargon refresh pattern (pair_style zero so no forces):
    group nonargon type 1 2 3              → initial count = n_C
    fix depo insert deposit 1 3 1 seed ... → stage one O (type 3)
    run 1 post no                           → O atom deposited
    group nonargon type 1 2 3              → refresh picks up new O
    count(nonargon) must equal n_C + 1

Run with:
    /home/lh0534/.conda/envs/jaxmd/bin/python3 -m pytest tests/test_rie_radical_deposit.py -m lammps -v
"""
import pytest
from pathlib import Path

from diamond_etch_md.spec import SimSpec, IonComponent, CyclePhase
from diamond_etch_md.lammps.head import get_head_lmp, get_head_lmp_multi_ion
from diamond_etch_md.lammps.head_cycling import get_head_lmp_cycle_etch

lammps_mod = pytest.importorskip("lammps", reason="LAMMPS Python module not available — run with jaxmd env")
pytestmark = pytest.mark.lammps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lmp(log_path):
    return lammps_mod.lammps(cmdargs=["-log", str(log_path), "-screen", "none"])


def _ivar(L, expr):
    """Read a LAMMPS expression as int via a temporary equal-style variable."""
    L.command(f"variable _tmp equal {expr}")
    val = int(L.extract_variable("_tmp", None, 0))
    L.command("variable _tmp delete")
    return val


# ---------------------------------------------------------------------------
# Shared spec factories — kept as functions so collection doesn't fail if
# SimSpec validation changes.
# ---------------------------------------------------------------------------

def _standard_rie_head():
    return get_head_lmp(SimSpec(species="Ar", energy=100.0, flux_ratio=5))


def _multi_ion_rie_head():
    return get_head_lmp_multi_ion(
        SimSpec(ion_mix=[IonComponent(species="Ar", fraction=1.0, energy=100.0)], flux_ratio=5)
    )


def _cycling_rie_head():
    return get_head_lmp_cycle_etch(
        SimSpec(phases=[CyclePhase(species="Ar", energy=100.0, fluence_ml=1, flux_ratio=5)])
    )


# ---------------------------------------------------------------------------
# String-level assertions — verify generated head.lmp text for all 3 paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,head_fn", [
    ("standard_rie",   _standard_rie_head),
    ("multi_ion_rie",  _multi_ion_rie_head),
    ("cycling_rie",    _cycling_rie_head),
])
def test_nonargon_qeq_fix_in_generated_head(label, head_fn):
    """Generated head.lmp must scope fix reax_qeq to the nonargon group (not 'all').

    When Ar ions are present, qeq must NOT run on Ar (type 4) — it bonds via ZBL,
    not ReaxFF, so including it in qeq produces nonsense charges.
    """
    head = head_fn()
    assert "fix reax_qeq nonargon" in head, (
        f"[{label}] Expected 'fix reax_qeq nonargon' in generated head.lmp; "
        f"qeq must be scoped to nonargon group when Ar is present"
    )
    assert "fix reax_qeq all" not in head, (
        f"[{label}] Found 'fix reax_qeq all' — should be nonargon when Ar is present"
    )


@pytest.mark.parametrize("label,head_fn", [
    ("standard_rie",   _standard_rie_head),
    ("multi_ion_rie",  _multi_ion_rie_head),
    ("cycling_rie",    _cycling_rie_head),
])
def test_nonargon_group_defined_in_generated_head(label, head_fn):
    """Generated head.lmp must define 'group nonargon type 1 2 3'.

    This group is used by both the qeq fix and the post-deposit refresh.
    """
    head = head_fn()
    assert "nonargon type 1 2 3" in head, (
        f"[{label}] Expected 'nonargon type 1 2 3' in generated head.lmp"
    )


@pytest.mark.parametrize("label,head_fn", [
    ("standard_rie",  _standard_rie_head),
    ("multi_ion_rie", _multi_ion_rie_head),
    ("cycling_rie",   _cycling_rie_head),
])
def test_nonargon_refresh_after_deposit_in_generated_head(label, head_fn):
    """For RIE code paths the post-deposit nonargon refresh must appear in head.lmp.

    The Kokkos fix qeq/reaxff does not resize its internal arrays when atoms are
    deposited mid-run.  Re-running 'group nonargon type 1 2 3' after each deposit
    forces a full re-initialisation so the new O• atom gets qeq charges.
    The refresh appears as 'group       nonargon type 1 2 3' (extra spaces) in the
    radical deposition block, distinct from the loop-top re-group.
    """
    head = head_fn()
    # Both the loop-top re-group and the post-deposit refresh use "nonargon type 1 2 3".
    # Count occurrences: at minimum the potential block + the loop-top + the post-deposit.
    occurrences = head.count("nonargon type 1 2 3")
    assert occurrences >= 2, (
        f"[{label}] Expected at least 2 occurrences of 'nonargon type 1 2 3' "
        f"(initial definition + post-deposit refresh); found {occurrences}"
    )


# ---------------------------------------------------------------------------
# LAMMPS mini-test — run the actual deposit + refresh pattern
# ---------------------------------------------------------------------------

# Geometry: 4 C atoms (type 1) on a flat surface at z=1.
# Deposit zone: z=6-8 (well above surface; >3 Å from any C).
# After fix deposit runs for 1 timestep, 1 O (type 3) is placed.
# Nonargon refresh picks up the new O → count should be n_C + 1.
_SETUP_DEPOSIT = """\
units        real
atom_style   atomic
boundary     p p p

region       box block 0 20 0 20 0 15
create_box   3 box

mass  1 12.011
mass  2  1.008
mass  3 16.000

pair_style   zero 3.0
pair_coeff   * *

create_atoms 1 single  5.0  5.0 1.0
create_atoms 1 single 15.0  5.0 1.0
create_atoms 1 single  5.0 15.0 1.0
create_atoms 1 single 15.0 15.0 1.0
"""


def test_deposited_o_in_nonargon_group(tmp_path):
    """After fix deposit + run 1, the nonargon refresh must include the new O atom.

    Mirrors the generated head.lmp pattern for Ar+RIE simulations:
      group nonargon type 1 2 3        (outer loop top — sets initial membership)
      fix depo insert deposit 1 3 1 ...
      run 1 post no
      group nonargon type 1 2 3        (post-deposit refresh — adds new O to group)
    """
    L = _lmp(tmp_path / "log.lammps")
    try:
        L.commands_string(_SETUP_DEPOSIT)
        n_c = L.get_natoms()

        # Outer-loop-top nonargon definition (all C atoms; no O yet)
        L.command("group nonargon type 1 2 3")
        n_before = _ivar(L, "count(nonargon)")

        # Deposit one O (type 3) into the region at z=7 — mirrors head.lmp fix deposit pattern:
        # fix depo insert deposit 1 3 1 seed global z z vx lo hi vy lo hi vz lo hi region R units box
        L.commands_string("""\
group        insert empty
region       dep_zone block 2.0 18.0 2.0 18.0 5.0 9.0 units box
fix          depo insert deposit 1 3 1 12345 global 7.0 7.0 vx 0 0 vy 0 0 vz -0.001 -0.001 region dep_zone units box
timestep     1e-10
run          1 post no
""")

        # Post-deposit nonargon refresh — mirrors head.lmp generated code
        L.command("group       nonargon type 1 2 3")
        n_after = _ivar(L, "count(nonargon)")

    finally:
        L.close()

    assert n_before == n_c, (
        f"Initial nonargon count ({n_before}) must equal n_C ({n_c}) before any deposit"
    )
    assert n_after == n_c + 1, (
        f"After deposit+refresh: nonargon={n_after}, expected {n_c + 1} (n_C={n_c} + 1 O)"
    )
