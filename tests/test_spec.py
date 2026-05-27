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


def test_ml_110_4x6():
    assert compute_ml("110", 4, 6) == 96


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
    spec = SimSpec(orientation="100", surface="1x1", species="Xe", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_surface_100():
    spec = SimSpec(orientation="100", surface="nonexistent", ml=81)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_surface_111():
    spec = SimSpec(orientation="111", surface="nonexistent", ml=90)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_bad_surface_113():
    spec = SimSpec(orientation="113", surface="nonexistent", ml=108)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_O_ether_not_valid_for_111():
    spec = SimSpec(orientation="111", surface="O_ether", ml=90)
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

def test_validate_100_surfaces():
    for surf in ("1x1", "O_ether"):
        spec = SimSpec(orientation="100", surface=surf, ml=81)
        validate(spec)
    for surf in ("2x1", "2x1_O"):
        spec = SimSpec(orientation="100", surface=surf, ml=64, box_x=8, box_y=8)
        validate(spec)


def test_validate_110_surfaces():
    for surf in ("", "O"):
        spec = SimSpec(orientation="110", surface=surf, ml=96,
                       box_x=4, box_y=6, box_depth=5)
        validate(spec)


def test_validate_111_surfaces():
    for surf in ("1x1", "2x1_single", "2x1_pandey",
                 "1x1_O", "2x1_single_O", "2x1_pandey_O"):
        spec = SimSpec(orientation="111", surface=surf, ml=90,
                       box_x=5, box_y=9)
        validate(spec)


def test_validate_113_surfaces():
    for surf in ("", "O"):
        spec = SimSpec(orientation="113", surface=surf, ml=108,
                       box_x=9, box_y=3)
        validate(spec)


def test_validate_species_Ar():
    spec = SimSpec(orientation="100", surface="1x1", species="Ar", ml=81)
    validate(spec)


def test_validate_species_O2():
    spec = SimSpec(orientation="100", surface="1x1", species="O2", ml=81)
    validate(spec)


# ─── SimSpec defaults ─────────────────────────────────────────────────────────

def test_simspec_defaults():
    s = SimSpec()
    assert s.orientation == "100"
    assert s.surface == "1x1"
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
    assert s.plot_interval_hours == 12


def test_simspec_custom_fields():
    s = SimSpec(
        orientation="113",
        surface="O",
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
    assert s.surface == "O"
    assert s.energy == 2.0
    assert s.ml == 108
    assert s.name == "test_job"
    assert s.account == "mygroup"
    assert s.email == "user@example.com"
