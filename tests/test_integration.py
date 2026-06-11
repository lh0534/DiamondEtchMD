"""
tests/test_integration.py — smoke test: build a simulation directory end-to-end.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md import SimSpec, CyclePhase, compute_ml, make_sim, make_ale, etch_mode


def test_make_sim_creates_expected_files(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="1x1",
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
    assert (outdir / "spec.json").exists()
    assert (outdir / "auto-plot.py").exists()
    for name in ("sweep.lmp", "thermalize.lmp", "addfix.lmp", "ffield.reax"):
        assert (outdir / name).exists()


def test_make_sim_config_matches_spec(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="1x1",
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
        surface="1x1",
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
        surface="2x1_pandey",
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


def test_make_sim_111_pandey_O(tmp_path):
    spec = SimSpec(
        orientation="111",
        surface="2x1_pandey_O",
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
        surface="1x1",
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
    assert "IonRemove" not in head

    cfg = (outdir / "config.lmp").read_text()
    assert "incident_type_index equal 4" in cfg
    assert "M_Ar equal 39.948" in cfg
    assert not (outdir / "O2.molecule").exists()


def test_make_sim_O2(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="1x1",
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
    assert "zbl" not in head

    cfg = (outdir / "config.lmp").read_text()
    assert "energ equal 50.0" in cfg
    assert (outdir / "O2.molecule").exists()


def test_make_sim_110(tmp_path):
    spec = SimSpec(
        orientation="110",
        surface="O",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=compute_ml("110", 4, 6),
        box_x=4, box_y=6, box_depth=5,
        name="110_O",
    )
    make_sim(spec, tmp_path / "sim")
    assert (tmp_path / "sim" / "make_surf.lmp").exists()
    surf = (tmp_path / "sim" / "make_surf.lmp").read_text()
    assert "orient z 1 1 0" in surf
    cfg = (tmp_path / "sim" / "config.lmp").read_text()
    assert "variable    O_terminate equal true" in cfg


def test_make_sim_113(tmp_path):
    spec = SimSpec(
        orientation="113",
        surface="",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=compute_ml("113", 9, 3),
        box_x=9, box_y=3, box_depth=3,
        name="113_bare",
    )
    make_sim(spec, tmp_path / "sim")
    assert (tmp_path / "sim" / "make_surf.lmp").exists()
    cfg = (tmp_path / "sim" / "config.lmp").read_text()
    assert "variable    O_terminate equal false" in cfg


def test_make_sim_113_O(tmp_path):
    spec = SimSpec(
        orientation="113",
        surface="O",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=compute_ml("113", 9, 3),
        box_x=9, box_y=3, box_depth=3,
        name="113_O",
    )
    make_sim(spec, tmp_path / "sim")
    cfg = (tmp_path / "sim" / "config.lmp").read_text()
    assert "variable    O_terminate equal true" in cfg


# ─── RIE-etch integration tests ──────────────────────────────────────────────

def test_make_sim_rie_etch_creates_expected_files(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="1x1",
        species="O",
        energy=20.0,
        temperature=300.0,
        ml=compute_ml("100", 9, 9),
        box_x=9, box_y=9, box_depth=5,
        flux_ratio=5,
        radical_energy=0.2,
        name="rie_smoke",
    )
    outdir = tmp_path / "rie_sim"
    make_sim(spec, outdir)

    assert (outdir / "config.lmp").exists()
    assert (outdir / "head.lmp").exists()
    assert (outdir / "submit").exists()
    assert (outdir / "spec.json").exists()


def test_make_sim_rie_etch_mode(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="1x1",
        species="O",
        energy=20.0,
        ml=compute_ml("100", 9, 9),
        flux_ratio=5,
        name="rie_mode",
    )
    assert etch_mode(spec) == "rie-etch"


def test_make_sim_rie_etch_config(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="1x1",
        species="O",
        energy=20.0,
        temperature=300.0,
        ml=compute_ml("100", 9, 9),
        box_x=9, box_y=9, box_depth=5,
        flux_ratio=5,
        radical_energy=0.2,
        name="rie_cfg",
    )
    make_sim(spec, tmp_path / "sim")
    cfg = (tmp_path / "sim" / "config.lmp").read_text()
    assert "variable    flux_ratio equal 5" in cfg
    assert "variable    radical_energy equal 0.2" in cfg


def test_make_sim_rie_etch_head(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="1x1",
        species="O",
        energy=20.0,
        temperature=300.0,
        ml=compute_ml("100", 9, 9),
        box_x=9, box_y=9, box_depth=5,
        flux_ratio=5,
        radical_energy=0.2,
        name="rie_head",
    )
    make_sim(spec, tmp_path / "sim")
    head = (tmp_path / "sim" / "head.lmp").read_text()
    assert "neutral_loop" in head
    assert "skip_radicals" in head
    assert "${c} 0 ${ncarbon}" in head


def test_make_sim_rie_etch_submit(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="1x1",
        species="O",
        energy=20.0,
        temperature=300.0,
        ml=compute_ml("100", 9, 9),
        flux_ratio=5,
        name="rie_submit",
    )
    make_sim(spec, tmp_path / "sim")
    sub = (tmp_path / "sim" / "submit").read_text()
    assert "neut_complete" in sub
    assert "cn_start" in sub


# ─── make_ale() integration tests ────────────────────────────────────────────

def test_make_ale_2phase_succeeds(tmp_path):
    spec = SimSpec(
        orientation="100",
        surface="O_ether",
        temperature=300.0,
        ml=compute_ml("100", 8, 8),
        box_x=8, box_y=8, box_depth=5,
        phases=[
            CyclePhase(species="Ar", energy=30.0, fluence_ml=5),
            CyclePhase(species="O2", energy=20.0, fluence_ml=5, flux_ratio=10),
        ],
        cycles=5,
        name="ale_test",
    )
    outdir = tmp_path / "ale_sim"
    make_ale(spec, outdir)
    assert (outdir / "head.lmp").exists()
    assert (outdir / "config.lmp").exists()


def test_make_ale_no_phases_fails(tmp_path):
    spec = SimSpec(orientation="100", surface="1x1", species="O", ml=81, name="bad_ale")
    with pytest.raises(SystemExit):
        make_ale(spec, tmp_path / "ale_bad")


def test_make_ale_3phase_fails(tmp_path):
    spec = SimSpec(
        orientation="100", surface="1x1", ml=81,
        phases=[
            CyclePhase(species="Ar", energy=30.0, fluence_ml=5),
            CyclePhase(species="O",  energy=1.0,  fluence_ml=3),
            CyclePhase(species="O2", energy=20.0, fluence_ml=5),
        ],
        name="bad_ale_3phase",
    )
    with pytest.raises(SystemExit):
        make_ale(spec, tmp_path / "ale_bad")
