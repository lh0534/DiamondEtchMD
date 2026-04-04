"""
tests/test_ml.py — parametrized tests for the ML formula across all orientations and box sizes.

ML = ml_factor * box_x * box_y
  001: ml_factor = 1
  111: ml_factor = 2
  113: ml_factor = 4
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md.spec import compute_ml
from diamond_etch_md.orientations import ORIENT


# ─── parametrized cases: (orientation, box_x, box_y, expected_ml) ────────────

@pytest.mark.parametrize("orientation, box_x, box_y, expected", [
    # 001  ml_factor=1
    ("001", 9,  9,  81),
    ("001", 5,  5,  25),
    ("001", 10, 10, 100),
    ("001", 6,  8,  48),
    ("001", 1,  1,  1),
    # 111  ml_factor=2
    ("111", 5,  9,  90),
    ("111", 9,  9,  162),
    ("111", 3,  5,  30),
    ("111", 10, 10, 200),
    ("111", 1,  1,  2),
    # 113  ml_factor=4
    ("113", 9,  3,  108),
    ("113", 6,  3,  72),
    ("113", 9,  6,  216),
    ("113", 3,  3,  36),
    ("113", 1,  1,  4),
])
def test_compute_ml(orientation, box_x, box_y, expected):
    result = compute_ml(orientation, box_x, box_y)
    assert result == expected, (
        f"compute_ml({orientation!r}, {box_x}, {box_y}) = {result}, expected {expected}"
    )


def test_ml_factor_matches_orient_registry():
    """compute_ml must use exactly the ml_factor stored in the ORIENT registry."""
    for orientation, cfg in ORIENT.items():
        factor = cfg["ml_factor"]
        for box_x, box_y in [(9, 9), (5, 3), (1, 1)]:
            expected = factor * box_x * box_y
            assert compute_ml(orientation, box_x, box_y) == expected


def test_ml_commutative_except_for_anisotropy():
    """ML formula is commutative in box_x and box_y (they enter symmetrically)."""
    for orientation in ORIENT:
        assert compute_ml(orientation, 4, 7) == compute_ml(orientation, 7, 4)


def test_ml_scales_linearly_with_box():
    """Doubling box_x should double ML."""
    for orientation in ORIENT:
        ml_base = compute_ml(orientation, 5, 5)
        ml_2x   = compute_ml(orientation, 10, 5)
        assert ml_2x == 2 * ml_base


@pytest.mark.parametrize("orientation", list(ORIENT.keys()))
def test_ml_always_positive(orientation):
    """ML must be strictly positive for any positive box dimensions."""
    assert compute_ml(orientation, 1, 1) > 0
    assert compute_ml(orientation, 100, 100) > 0
