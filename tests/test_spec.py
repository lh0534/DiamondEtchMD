"""
tests/test_spec.py — tests for SimSpec, compute_ml, and validate().
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md.spec import SimSpec, compute_ml, validate


# ─── ML computation ──────────────────────────────────────────────────────────

def test_ml_100_9x9():
    assert compute_ml("100", 9, 9) == 81


def test_ml_111_5x9():
    assert compute_ml("111", 5, 9) == 90


def test_ml_113_9x3():
    assert compute_ml("113", 9, 3) == 108


# ─── validate() error paths ───────────────────────────────────────────────────

def test_validate_bad_orientation():
    spec = SimSpec(orientation="999", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_species():
    spec = SimSpec(orientation="100", reconstruction="bare_1x1", species="Xe", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_reconstruction_100():
    spec = SimSpec(orientation="100", reconstruction="nonexistent", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_reconstruction_111():
    spec = SimSpec(orientation="111", reconstruction="nonexistent", ml=90)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_reconstruction_113():
    spec = SimSpec(orientation="113", reconstruction="nonexistent", ml=108)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_termination_100():
    spec = SimSpec(orientation="100", reconstruction="bare_1x1", termination="fluorine", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_termination_H_dropped():
    """H termination has been removed; should fail validation."""
    spec = SimSpec(orientation="100", reconstruction="bare_1x1", termination="H", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_mismatched_termination_111():
    """O_2x1_pandey termination is not valid with bare_1x1 reconstruction."""
    spec = SimSpec(orientation="111", reconstruction="bare_1x1",
                   termination="O_2x1_pandey", ml=90)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_ml_zero():
    spec = SimSpec(orientation="100", ml=0)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_ml_negative():
    spec = SimSpec(orientation="100", ml=-5)
    with pytest.raises(SystemExit):
        validate(spec)


# ─── validate() success paths ────────────────────────────────────────────────

def test_validate_100_valid_reconstructions():
    for recon in ("bare_1x1", "bare_2x1"):
        spec = SimSpec(orientation="100", reconstruction=recon, ml=81)
        validate(spec)


def test_validate_100_valid_terminations():
    for term in ("bare", "O", "O_ether"):
        spec = SimSpec(orientation="100", reconstruction="bare_1x1",
                       termination=term, ml=81)
        validate(spec)


def test_validate_111_valid_reconstructions():
    for recon in ("bare_1x1", "bare_2x1_single", "bare_2x1_pandey"):
        spec = SimSpec(orientation="111", reconstruction=recon, ml=90)
        validate(spec)


def test_validate_111_valid_O_terminations():
    pairs = [
        ("bare_1x1",        "O_1x1"),
        ("bare_2x1_single", "O_2x1_single"),
        ("bare_2x1_pandey", "O_2x1_pandey"),
    ]
    for recon, term in pairs:
        spec = SimSpec(orientation="111", reconstruction=recon,
                       termination=term, ml=90)
        validate(spec)


def test_validate_113_valid():
    for term in ("bare", "O"):
        spec = SimSpec(orientation="113", reconstruction="bare",
                       termination=term, ml=108)
        validate(spec)


# ─── SimSpec defaults ─────────────────────────────────────────────────────────

def test_simspec_defaults():
    s = SimSpec()
    assert s.orientation == "100"
    assert s.reconstruction == "bare_1x1"
    assert s.termination == "bare"
    assert s.temperature == 300.0
    assert s.species == "O"
    assert s.energy == 0.5
    assert s.angle == 0.0
    assert s.fluence == 50
    assert s.ml == 0
    assert s.box_x == 9
    assert s.box_y == 9
    assert s.box_depth == 3
    assert s.impact_time == 2000.0
    assert s.thermalization_time == 500.0
    assert s.wall_hours == 24
    assert s.name == ""
    assert s.account == "dgraves"
    assert s.email == ""
    assert s.lammps_module == "lammps/kokkos/gpu_della9_2022"


def test_validate_species_Ar():
    spec = SimSpec(orientation="100", reconstruction="bare_1x1", species="Ar", ml=81)
    validate(spec)


def test_validate_species_O2():
    spec = SimSpec(orientation="100", reconstruction="bare_1x1", species="O2", ml=81)
    validate(spec)


def test_simspec_custom_fields():
    s = SimSpec(
        orientation="113",
        reconstruction="bare",
        termination="O",
        temperature=500.0,
        species="O",
        energy=2.0,
        angle=15.0,
        fluence=100,
        ml=108,
        box_x=9,
        box_y=3,
        box_depth=4,
        impact_time=3000.0,
        thermalization_time=800.0,
        wall_hours=48,
        name="test_job",
        account="mygroup",
        email="user@example.com",
    )
    assert s.orientation == "113"
    assert s.energy == 2.0
    assert s.ml == 108
    assert s.name == "test_job"
    assert s.account == "mygroup"
    assert s.email == "user@example.com"
