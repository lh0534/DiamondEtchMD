"""
tests/test_integration.py — smoke test: build a simulation directory end-to-end.

All surface templates are bundled with the package, so no external directory
is needed.  Checks that make_sim writes the expected files and that their
contents match the SimSpec.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md import SimSpec, compute_ml, make_sim


def test_make_sim_creates_expected_files(tmp_path):
    spec = SimSpec(
        orientation="100",
        reconstruction="bare_1x1",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=compute_ml("100", 9, 9),
        box_x=9, box_y=9, box_depth=3,
        name="smoke_test",
    )
    outdir = tmp_path / "sim"
    make_sim(spec, outdir)

    assert (outdir / "config.lmp").exists()
    assert (outdir / "head.lmp").exists()
    assert (outdir / "make_surf.lmp").exists()
    assert (outdir / "submit").exists()
    for name in ("sweep.lmp", "thermalize.lmp", "addfix.lmp", "ffield.reax"):
        assert (outdir / name).exists()


def test_make_sim_config_matches_spec(tmp_path):
    spec = SimSpec(
        orientation="100",
        reconstruction="bare_1x1",
        species="O",
        energy=1.5,
        temperature=600.0,
        ml=compute_ml("100", 9, 9),
        box_x=9, box_y=9, box_depth=3,
        name="config_check",
    )
    make_sim(spec, tmp_path / "sim")
    cfg = (tmp_path / "sim" / "config.lmp").read_text()

    assert "variable    ML equal 81" in cfg
    assert "1.5" in cfg      # energy
    assert "600.0" in cfg    # temperature
    assert "incident_type_index equal 3" in cfg   # O is type 3


def test_make_sim_submit_matches_spec(tmp_path):
    spec = SimSpec(
        orientation="100",
        reconstruction="bare_1x1",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=81,
        wall_hours=48,
        name="submit_check",
    )
    make_sim(spec, tmp_path / "sim")
    sub = (tmp_path / "sim" / "submit").read_text()

    assert "#SBATCH --job-name=submit_check" in sub
    assert "#SBATCH --time=48:00:00" in sub
    assert "-in head.lmp" in sub


def test_make_sim_111_pandey(tmp_path):
    spec = SimSpec(
        orientation="111",
        reconstruction="bare_2x1_pandey",
        termination="bare",
        species="O",
        energy=1.0,
        temperature=300.0,
        ml=compute_ml("111", 5, 9),
        box_x=5, box_y=9, box_depth=3,
        name="111_pandey",
    )
    make_sim(spec, tmp_path / "sim")
    assert (tmp_path / "sim" / "make_surf.lmp").exists()
    surf = (tmp_path / "sim" / "make_surf.lmp").read_text()
    assert "Pandey" in surf


def test_make_sim_111_O_termination(tmp_path):
    spec = SimSpec(
        orientation="111",
        reconstruction="bare_2x1_pandey",
        termination="O_2x1_pandey",
        species="O",
        energy=1.0,
        temperature=300.0,
        ml=compute_ml("111", 5, 9),
        box_x=5, box_y=9, box_depth=3,
        name="111_pandey_O",
    )
    make_sim(spec, tmp_path / "sim")
    cfg = (tmp_path / "sim" / "config.lmp").read_text()
    assert "variable    O_terminate equal true" in cfg


def test_make_sim_Ar(tmp_path):
    spec = SimSpec(
        orientation="100",
        reconstruction="bare_1x1",
        species="Ar",
        energy=100.0,
        temperature=300.0,
        ml=compute_ml("100", 9, 9),
        box_x=9, box_y=9, box_depth=3,
        name="Ar_test",
    )
    outdir = tmp_path / "sim"
    make_sim(spec, outdir)

    head = (outdir / "head.lmp").read_text()
    assert "hybrid reaxff" in head
    assert "zbl" in head
    assert "delete_atoms group IonRemove" in head

    cfg = (outdir / "config.lmp").read_text()
    assert "incident_type_index equal 4" in cfg
    assert "M_Ar equal 39.948" in cfg

    # O2.molecule should NOT be present for Ar
    assert not (outdir / "O2.molecule").exists()


def test_make_sim_O2(tmp_path):
    spec = SimSpec(
        orientation="100",
        reconstruction="bare_1x1",
        species="O2",
        energy=100.0,
        temperature=300.0,
        ml=compute_ml("100", 9, 9),
        box_x=9, box_y=9, box_depth=3,
        name="O2_test",
    )
    outdir = tmp_path / "sim"
    make_sim(spec, outdir)

    head = (outdir / "head.lmp").read_text()
    assert "molecule O2 O2.molecule" in head
    assert "mol O2" in head
    # no ZBL for O2
    assert "zbl" not in head

    cfg = (outdir / "config.lmp").read_text()
    assert "energ equal 50.0" in cfg  # energy halved for O2

    # O2.molecule should be present
    assert (outdir / "O2.molecule").exists()


def test_make_sim_113(tmp_path):
    spec = SimSpec(
        orientation="113",
        reconstruction="bare",
        termination="O",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=compute_ml("113", 9, 3),
        box_x=9, box_y=3, box_depth=3,
        name="113_O",
    )
    make_sim(spec, tmp_path / "sim")
    assert (tmp_path / "sim" / "make_surf.lmp").exists()
    cfg = (tmp_path / "sim" / "config.lmp").read_text()
    assert "variable    O_terminate equal true" in cfg
