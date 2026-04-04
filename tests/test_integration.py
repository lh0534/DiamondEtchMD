"""
tests/test_integration.py — smoke test: build a simulation directory end-to-end.

Uses a tmp_path as both outdir and a fake dfiles_root so no real filesystem
layout is required.  Checks that make_sim writes the expected files and that
their contents match the SimSpec.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md import SimSpec, compute_ml, make_sim


@pytest.fixture
def fake_dfiles(tmp_path):
    """Minimal dfiles/ tree with the files make_sim needs to copy/symlink."""
    d = tmp_path / "dfiles"
    rad = d / "radicals"
    rad.mkdir(parents=True)
    (rad / "make_surf.lmp").write_text("# fake make_surf\n")
    for name in ("sweep.lmp", "thermalize.lmp", "addfix.lmp"):
        (rad / name).write_text(f"# fake {name}\n")
    for name in ("ffield.reax", "lat_a.txt", "lmp_env.sh"):
        (d / name).write_text(f"# fake {name}\n")
    return d


def test_make_sim_creates_expected_files(tmp_path, fake_dfiles):
    spec = SimSpec(
        orientation="001",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=compute_ml("001", 9, 9),
        box_x=9, box_y=9, box_depth=3,
        name="smoke_test",
    )
    outdir = tmp_path / "sim"
    make_sim(spec, outdir, fake_dfiles)

    assert (outdir / "config.lmp").exists()
    assert (outdir / "head.lmp").exists()
    assert (outdir / "make_surf.lmp").exists()
    assert (outdir / "submit").exists()
    for name in ("sweep.lmp", "thermalize.lmp", "addfix.lmp", "ffield.reax"):
        assert (outdir / name).exists()


def test_make_sim_config_matches_spec(tmp_path, fake_dfiles):
    spec = SimSpec(
        orientation="001",
        species="H",
        energy=1.5,
        temperature=600.0,
        ml=compute_ml("001", 9, 9),
        box_x=9, box_y=9, box_depth=3,
        name="config_check",
    )
    make_sim(spec, tmp_path / "sim", fake_dfiles)
    cfg = (tmp_path / "sim" / "config.lmp").read_text()

    assert "variable    ML equal 81" in cfg
    assert "1.5" in cfg      # energy
    assert "600.0" in cfg    # temperature
    assert "incident_type_index equal 2" in cfg   # H is type 2


def test_make_sim_submit_matches_spec(tmp_path, fake_dfiles):
    spec = SimSpec(
        orientation="001",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=81,
        wall_hours=48,
        name="submit_check",
    )
    make_sim(spec, tmp_path / "sim", fake_dfiles)
    sub = (tmp_path / "sim" / "submit").read_text()

    assert "#SBATCH --job-name=submit_check" in sub
    assert "#SBATCH --time=48:00:00" in sub
    assert "-in head.lmp" in sub
