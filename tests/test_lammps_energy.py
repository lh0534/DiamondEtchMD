"""
tests/test_lammps_energy.py — single-point ReaxFF energy via the LAMMPS Python module.

Requires the jaxmd conda environment (CPU-only LAMMPS with ReaxFF + qeq/reaxff):

    /home/lh0534/.conda/envs/jaxmd/bin/python3 -m pytest tests/test_lammps_energy.py -m lammps -v

Skip in normal CI:

    pytest -m "not lammps and not slurm"
"""
import pytest
from pathlib import Path

lammps_mod = pytest.importorskip("lammps", reason="LAMMPS Python module not available — run with jaxmd env")

FFIELD = Path(__file__).parents[1] / "diamond_etch_md" / "lammps" / "templates" / "ffield.reax"
KCAL_TO_EV = 1.0 / 23.060549

pytestmark = pytest.mark.lammps


def _make_lammps(log_path):
    return lammps_mod.lammps(cmdargs=["-log", str(log_path), "-screen", "none"])


def test_ffield_exists():
    """Bundled force field file must be present."""
    assert FFIELD.exists(), f"ffield.reax not found at {FFIELD}"


def test_diamond_c_single_point_energy(tmp_path):
    """Single-point ReaxFF energy of a 2×2×2 diamond-cubic C supercell (64 atoms).

    Physical expectations (bulk diamond C with ffield.reax):
    - PE is negative (bound solid)
    - Per-atom energy between -8.1 and -6.5 eV  (ffield.reax gives ~-7.87 eV/atom; literature ~-7.3 eV)
    """
    L = _make_lammps(tmp_path / "log.lammps")
    try:
        L.commands_string(f"""
units        real
atom_style   charge
boundary     p p p

lattice      diamond 3.5669
region       box block 0 2 0 2 0 2
create_box   1 box
create_atoms 1 box

mass         1 12.011

pair_style   reaxff NULL
pair_coeff   * * {FFIELD} C

fix          qeq all qeq/reaxff 1 0.0 6.0 1e-6 reaxff

thermo_style custom step pe
run          0
""")
        pe_kcal = L.get_thermo("pe")
        n_atoms = L.get_natoms()
    finally:
        L.close()

    assert n_atoms == 64, f"Expected 64 atoms, got {n_atoms}"

    pe_ev = pe_kcal * KCAL_TO_EV
    pe_per_atom_ev = pe_ev / n_atoms

    assert pe_kcal < 0, f"PE should be negative for bulk diamond, got {pe_kcal:.2f} kcal/mol"
    assert -8.1 < pe_per_atom_ev < -6.5, (
        f"Per-atom PE {pe_per_atom_ev:.3f} eV outside expected range [-8.1, -6.5] eV "
        f"for bulk C diamond with ffield.reax"
    )


def test_diamond_c_energy_decreases_after_minimize(tmp_path):
    """After CG minimization, PE must be <= the single-point value."""
    L = _make_lammps(tmp_path / "log.lammps")
    try:
        L.commands_string(f"""
units        real
atom_style   charge
boundary     p p p

lattice      diamond 3.5669
region       box block 0 1 0 1 0 1
create_box   1 box
create_atoms 1 box

mass         1 12.011

pair_style   reaxff NULL
pair_coeff   * * {FFIELD} C

fix          qeq all qeq/reaxff 1 0.0 6.0 1e-6 reaxff

thermo_style custom step pe
run          0
""")
        pe_sp = L.get_thermo("pe")

        L.commands_string("""
min_style    cg
minimize     1.0e-8 1.0e-10 5000 50000
""")
        pe_min = L.get_thermo("pe")
    finally:
        L.close()

    assert pe_min <= pe_sp + 1e-6, (
        f"Minimized PE {pe_min:.4f} should be <= single-point PE {pe_sp:.4f} kcal/mol"
    )
