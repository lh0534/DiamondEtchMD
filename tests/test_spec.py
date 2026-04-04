"""
tests/test_spec.py — tests for SimSpec, compute_ml, and validate().
"""

import sys
import pytest
from pathlib import Path

# Ensure the package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md.spec import SimSpec, compute_ml, validate


# ─── ML computation ──────────────────────────────────────────────────────────

def test_ml_001_9x9():
    """001 orientation, 9×9 box → ML = 1*9*9 = 81."""
    assert compute_ml("001", 9, 9) == 81


def test_ml_111_5x9():
    """111 orientation, 5×9 box → ML = 2*5*9 = 90."""
    assert compute_ml("111", 5, 9) == 90


def test_ml_113_9x3():
    """113 orientation, 9×3 box → ML = 4*9*3 = 108."""
    assert compute_ml("113", 9, 3) == 108


# ─── validate() error paths ───────────────────────────────────────────────────

def test_validate_bad_orientation():
    spec = SimSpec(orientation="999", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_species():
    spec = SimSpec(orientation="001", species="Ar", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_reconstruction_111():
    """111 does not have a '*' wildcard; 'nonexistent' should fail."""
    spec = SimSpec(orientation="111", reconstruction="nonexistent", ml=90)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_reconstruction_113():
    """113 does not have a '*' wildcard; 'nonexistent' should fail."""
    spec = SimSpec(orientation="113", reconstruction="nonexistent", ml=108)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_termination():
    spec = SimSpec(orientation="001", termination="fluorine", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_ml_zero():
    """ml=0 should fail (must be > 0)."""
    spec = SimSpec(orientation="001", ml=0)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_ml_negative():
    spec = SimSpec(orientation="001", ml=-5)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_001_wildcard_any_reconstruction():
    """001 uses '*' wildcard, so any reconstruction value should be accepted."""
    spec = SimSpec(orientation="001", reconstruction="2x1", ml=81)
    validate(spec)  # should not raise


def test_validate_111_valid_reconstructions():
    for recon in ("bare", "1x1", "2x1_single", "2x1_pandey"):
        spec = SimSpec(orientation="111", reconstruction=recon, ml=90)
        validate(spec)  # should not raise


def test_validate_113_valid_reconstructions():
    for recon in ("bare", "O"):
        spec = SimSpec(orientation="113", reconstruction=recon, ml=108)
        validate(spec)  # should not raise


# ─── SimSpec defaults ─────────────────────────────────────────────────────────

def test_simspec_defaults():
    s = SimSpec()
    assert s.orientation == "001"
    assert s.reconstruction == "bare"
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


def test_simspec_custom_fields():
    s = SimSpec(
        orientation="113",
        reconstruction="O",
        termination="H",
        temperature=500.0,
        species="H",
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
    )
    assert s.orientation == "113"
    assert s.energy == 2.0
    assert s.ml == 108
    assert s.name == "test_job"
