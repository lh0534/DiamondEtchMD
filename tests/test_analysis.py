"""
tests/test_analysis.py — tests for the analysis submodules (etch_products and ncarbon).
"""

import sys
import pytest
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md.analysis.etch_products import parse_etch_products, etch_yield
from diamond_etch_md.analysis.ncarbon import parse_ncarbon, etch_depth


# ─── fixtures ─────────────────────────────────────────────────────────────────

ETCH_PRODUCTS_CONTENT = textwrap.dedent("""\
    12 C 1 0 0  0.001  0.003  0.412
    47 C 2 1 0 -0.002  0.001  0.387
    47 O 0 0 1  0.000 -0.005  0.210
""")

NCARBON_CONTENT = textwrap.dedent("""\
    1  648  0  0
    2  647  0  0
    50 630  0  12
""")


@pytest.fixture
def etch_products_file(tmp_path):
    p = tmp_path / "etch_products.txt"
    p.write_text(ETCH_PRODUCTS_CONTENT)
    return p


@pytest.fixture
def ncarbon_file(tmp_path):
    p = tmp_path / "ncarbon.txt"
    p.write_text(NCARBON_CONTENT)
    return p


# ─── parse_etch_products ──────────────────────────────────────────────────────

def test_parse_etch_products_count(etch_products_file):
    records = parse_etch_products(etch_products_file)
    assert len(records) == 3


def test_parse_etch_products_first_record(etch_products_file):
    r = parse_etch_products(etch_products_file)[0]
    assert r["impact"] == 12
    assert r["atom_type"] == "C"
    assert r["n_C"] == 1
    assert r["n_H"] == 0
    assert r["n_O"] == 0
    assert r["vx"] == pytest.approx(0.001)
    assert r["vy"] == pytest.approx(0.003)
    assert r["vz"] == pytest.approx(0.412)


def test_parse_etch_products_second_record(etch_products_file):
    r = parse_etch_products(etch_products_file)[1]
    assert r["impact"] == 47
    assert r["n_C"] == 2
    assert r["n_H"] == 1
    assert r["n_O"] == 0
    assert r["vx"] == pytest.approx(-0.002)


def test_parse_etch_products_oxygen_cluster(etch_products_file):
    r = parse_etch_products(etch_products_file)[2]
    assert r["atom_type"] == "O"
    assert r["n_C"] == 0
    assert r["n_O"] == 1


def test_parse_etch_products_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert parse_etch_products(p) == []


def test_parse_etch_products_skips_comments(tmp_path):
    p = tmp_path / "ep.txt"
    p.write_text("# comment\n1 C 1 0 0 0.0 0.0 0.5\n")
    records = parse_etch_products(p)
    assert len(records) == 1
    assert records[0]["impact"] == 1


# ─── etch_yield ───────────────────────────────────────────────────────────────

def test_etch_yield_basic(etch_products_file):
    records = parse_etch_products(etch_products_file)
    # total C ejected = 1 + 2 + 0 = 3; last impact = 47
    y = etch_yield(records, ml=81)
    assert y == pytest.approx(3 / 47)


def test_etch_yield_empty():
    assert etch_yield([], ml=81) == 0.0


def test_etch_yield_single_record(tmp_path):
    p = tmp_path / "ep.txt"
    p.write_text("10 C 3 0 0 0.0 0.0 0.5\n")
    records = parse_etch_products(p)
    assert etch_yield(records, ml=81) == pytest.approx(3 / 10)


# ─── parse_ncarbon ────────────────────────────────────────────────────────────

def test_parse_ncarbon_count(ncarbon_file):
    records = parse_ncarbon(ncarbon_file)
    assert len(records) == 3


def test_parse_ncarbon_first_record(ncarbon_file):
    r = parse_ncarbon(ncarbon_file)[0]
    assert r["impact"] == 1
    assert r["n_carbon"] == 648
    assert r["n_hydrogen"] == 0
    assert r["n_oxygen"] == 0


def test_parse_ncarbon_last_record(ncarbon_file):
    r = parse_ncarbon(ncarbon_file)[-1]
    assert r["impact"] == 50
    assert r["n_carbon"] == 630
    assert r["n_hydrogen"] == 0
    assert r["n_oxygen"] == 12


def test_parse_ncarbon_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert parse_ncarbon(p) == []


def test_parse_ncarbon_skips_comments(tmp_path):
    p = tmp_path / "nc.txt"
    p.write_text("# header\n1 648 0 0\n")
    records = parse_ncarbon(p)
    assert len(records) == 1


# ─── etch_depth ───────────────────────────────────────────────────────────────

def test_etch_depth_basic(ncarbon_file):
    records = parse_ncarbon(ncarbon_file)
    # n0=648; impacts: 648, 647, 630; ML=81
    depths = etch_depth(records, ml=81, box_x=9, box_y=9, orientation="100")
    assert len(depths) == 3
    assert depths[0] == pytest.approx(0.0)
    assert depths[1] == pytest.approx(1 / 81)
    assert depths[2] == pytest.approx(18 / 81)


def test_etch_depth_empty():
    assert etch_depth([], ml=81, box_x=9, box_y=9, orientation="100") == []


def test_etch_depth_no_etching(tmp_path):
    p = tmp_path / "nc.txt"
    p.write_text("1 100 0 0\n2 100 0 0\n3 100 0 0\n")
    records = parse_ncarbon(p)
    depths = etch_depth(records, ml=10, box_x=5, box_y=2, orientation="100")
    assert all(d == pytest.approx(0.0) for d in depths)


def test_etch_depth_monotone_decrease(ncarbon_file):
    records = parse_ncarbon(ncarbon_file)
    depths = etch_depth(records, ml=81, box_x=9, box_y=9, orientation="100")
    assert depths == sorted(depths)
