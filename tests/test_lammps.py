"""
tests/test_lammps.py — tests that generated LAMMPS input files contain expected content.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md.spec import SimSpec
from diamond_etch_md.lammps.config import get_config_lmp
from diamond_etch_md.lammps.head import get_head_lmp
from diamond_etch_md.lammps.submit import get_submit_script
from diamond_etch_md.orientations import ORIENT


# ─── fixtures ────────────────────────────────────────────────────────────────

def make_spec(orientation="001", energy=0.5, temperature=300.0, ml=81,
              box_x=9, box_y=9, box_depth=3, species="O",
              reconstruction="bare", termination="bare",
              name="test_job", **kw):
    return SimSpec(
        orientation=orientation,
        energy=energy,
        temperature=temperature,
        ml=ml,
        box_x=box_x,
        box_y=box_y,
        box_depth=box_depth,
        species=species,
        reconstruction=reconstruction,
        termination=termination,
        name=name,
        **kw,
    )


# ─── config.lmp tests ─────────────────────────────────────────────────────────

def test_config_contains_energy():
    spec = make_spec(energy=1.5)
    cfg = get_config_lmp(spec)
    assert "1.5" in cfg, "config.lmp should contain the energy value"


def test_config_contains_temperature():
    spec = make_spec(temperature=500.0)
    cfg = get_config_lmp(spec)
    assert "500.0" in cfg, "config.lmp should contain the temperature"


def test_config_contains_ml():
    spec = make_spec(ml=81)
    cfg = get_config_lmp(spec)
    assert "variable    ML equal 81" in cfg


def test_config_ml_113():
    spec = make_spec(orientation="113", ml=108, box_x=9, box_y=3)
    cfg = get_config_lmp(spec)
    assert "variable    ML equal 108" in cfg


def test_config_contains_box_x_and_y():
    spec = make_spec(box_x=6, box_y=4, ml=6)
    cfg = get_config_lmp(spec)
    assert "variable    x equal 6" in cfg
    assert "variable    y equal 4" in cfg


def test_config_recon_flag_001_2x1():
    spec = make_spec(orientation="001", reconstruction="2x1", ml=81)
    cfg = get_config_lmp(spec)
    assert "variable    reconstruct equal true" in cfg


def test_config_recon_flag_001_bare():
    spec = make_spec(orientation="001", reconstruction="bare", ml=81)
    cfg = get_config_lmp(spec)
    assert "variable    reconstruct equal false" in cfg


def test_config_recon_flag_111_always_false():
    """For 111, reconstruction is structural; flag is always false."""
    spec = make_spec(orientation="111", reconstruction="2x1_pandey", ml=90,
                     box_x=5, box_y=9)
    cfg = get_config_lmp(spec)
    assert "variable    reconstruct equal false" in cfg


def test_config_h_termination_flag():
    spec = make_spec(termination="H", ml=81)
    cfg = get_config_lmp(spec)
    assert "variable    H_terminate equal true" in cfg
    assert "variable    O_terminate equal false" in cfg
    assert "variable    O_ether_terminate equal false" in cfg


def test_config_o_termination_flag():
    spec = make_spec(termination="O", ml=81)
    cfg = get_config_lmp(spec)
    assert "variable    O_terminate equal true" in cfg
    assert "variable    H_terminate equal false" in cfg


def test_config_o_ether_termination_flag():
    spec = make_spec(termination="O_ether", ml=81)
    cfg = get_config_lmp(spec)
    assert "variable    O_ether_terminate equal true" in cfg


def test_config_species_O_type_index():
    spec = make_spec(species="O", ml=81)
    cfg = get_config_lmp(spec)
    assert "incident_type_index equal 3" in cfg


def test_config_species_H_type_index():
    spec = make_spec(species="H", ml=81)
    cfg = get_config_lmp(spec)
    assert "incident_type_index equal 2" in cfg


def test_config_orientation_comment():
    spec = make_spec(orientation="113", ml=108, box_x=9, box_y=3)
    cfg = get_config_lmp(spec)
    assert "orientation=113" in cfg


# ─── head.lmp tests ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("orientation", list(ORIENT.keys()))
def test_head_contains_lattice_cmd(orientation):
    """head.lmp must embed the orientation-specific lattice command."""
    ml_map = {"001": 81, "111": 90, "113": 108}
    bx_map = {"001": 9,  "111": 5,  "113": 9}
    by_map = {"001": 9,  "111": 9,  "113": 3}
    spec = make_spec(orientation=orientation,
                     ml=ml_map[orientation],
                     box_x=bx_map[orientation],
                     box_y=by_map[orientation])
    head = get_head_lmp(spec)
    expected_lattice = ORIENT[orientation]["lattice_cmd"]
    assert expected_lattice in head, (
        f"head.lmp for {orientation} should contain the orientation lattice command"
    )


@pytest.mark.parametrize("orientation", list(ORIENT.keys()))
def test_head_contains_bottom_expr(orientation):
    """head.lmp must embed the orientation-specific bottom expression."""
    ml_map = {"001": 81, "111": 90, "113": 108}
    bx_map = {"001": 9,  "111": 5,  "113": 9}
    by_map = {"001": 9,  "111": 9,  "113": 3}
    spec = make_spec(orientation=orientation,
                     ml=ml_map[orientation],
                     box_x=bx_map[orientation],
                     box_y=by_map[orientation])
    head = get_head_lmp(spec)
    expected_bottom = ORIENT[orientation]["bottom_expr"]
    assert expected_bottom in head, (
        f"head.lmp for {orientation} should contain the bottom expression"
    )


def test_head_001_lattice_specific_content():
    spec = make_spec(orientation="001", ml=81)
    head = get_head_lmp(spec)
    assert "orient z 0 0 1" in head
    assert "orient x 1 1 0" in head


def test_head_111_lattice_specific_content():
    spec = make_spec(orientation="111", ml=90, box_x=5, box_y=9)
    head = get_head_lmp(spec)
    assert "orient z 1 1 1" in head
    assert "orient x 2 -1 -1" in head


def test_head_113_lattice_specific_content():
    spec = make_spec(orientation="113", ml=108, box_x=9, box_y=3)
    head = get_head_lmp(spec)
    assert "orient z 1 1 3" in head
    assert "orient x -1 1 0" in head


def test_head_contains_energy_in_comment():
    spec = make_spec(energy=2.5, ml=81)
    head = get_head_lmp(spec)
    assert "2.5eV" in head


def test_head_contains_orientation_in_comment():
    spec = make_spec(orientation="113", ml=108, box_x=9, box_y=3)
    head = get_head_lmp(spec)
    assert "orientation=113" in head


def test_head_contains_loop_structure():
    spec = make_spec(ml=81)
    head = get_head_lmp(spec)
    assert "label\t\tloop" in head
    assert "jump\t\tSELF loop" in head
    assert "label\t\tcontinue_impact" in head


def test_head_contains_ncarbon_output():
    spec = make_spec(ml=81)
    head = get_head_lmp(spec)
    assert "ncarbon.txt" in head


# ─── submit script tests ─────────────────────────────────────────────────────

def test_submit_contains_job_name():
    spec = make_spec(name="my_diamond_sim")
    sub = get_submit_script(spec)
    assert "#SBATCH --job-name=my_diamond_sim" in sub


def test_submit_contains_wall_hours():
    spec = make_spec(wall_hours=48)
    sub = get_submit_script(spec)
    assert "#SBATCH --time=48:00:00" in sub


def test_submit_contains_gpu_request():
    spec = make_spec()
    sub = get_submit_script(spec)
    assert "#SBATCH --gres=gpu:1" in sub


def test_submit_contains_singleton_dependency():
    spec = make_spec()
    sub = get_submit_script(spec)
    assert "#SBATCH --dependency=singleton" in sub


def test_submit_contains_lammps_module():
    spec = make_spec()
    sub = get_submit_script(spec)
    assert "module load lammps/kokkos/gpu_della9_2022" in sub


def test_submit_contains_head_lmp_call():
    spec = make_spec()
    sub = get_submit_script(spec)
    assert "-in head.lmp" in sub


def test_submit_contains_make_surf_call():
    spec = make_spec()
    sub = get_submit_script(spec)
    assert "-in make_surf.lmp" in sub


def test_submit_creates_dirs():
    spec = make_spec()
    sub = get_submit_script(spec)
    assert "mkdir -p dumps data_files" in sub


def test_submit_custom_name():
    spec = make_spec(name="001_O_0.5eV_300K")
    sub = get_submit_script(spec)
    assert "#SBATCH --job-name=001_O_0.5eV_300K" in sub
