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

def make_spec(orientation="100", energy=0.5, temperature=300.0, ml=81,
              box_x=9, box_y=9, box_depth=3, species="O",
              reconstruction="bare_1x1", termination="bare",
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
    cfg = get_config_lmp(make_spec(energy=1.5))
    assert "1.5" in cfg


def test_config_contains_temperature():
    cfg = get_config_lmp(make_spec(temperature=500.0))
    assert "500.0" in cfg


def test_config_contains_ml():
    cfg = get_config_lmp(make_spec(ml=81))
    assert "variable    ML equal 81" in cfg


def test_config_ml_113():
    cfg = get_config_lmp(make_spec(orientation="113", reconstruction="bare",
                                   ml=108, box_x=9, box_y=3))
    assert "variable    ML equal 108" in cfg


def test_config_contains_box_x_and_y():
    cfg = get_config_lmp(make_spec(box_x=6, box_y=4, ml=6))
    assert "variable    x equal 6" in cfg
    assert "variable    y equal 4" in cfg


def test_config_recon_flag_100_bare_2x1():
    cfg = get_config_lmp(make_spec(orientation="100", reconstruction="bare_2x1", ml=81))
    assert "variable    reconstruct equal true" in cfg


def test_config_recon_flag_100_bare_1x1():
    cfg = get_config_lmp(make_spec(orientation="100", reconstruction="bare_1x1", ml=81))
    assert "variable    reconstruct equal false" in cfg


def test_config_recon_flag_111_always_false():
    """For 111, reconstruction is in the template; flag is always false."""
    cfg = get_config_lmp(make_spec(orientation="111", reconstruction="bare_2x1_pandey",
                                   ml=90, box_x=5, box_y=9))
    assert "variable    reconstruct equal false" in cfg


def test_config_o_termination_flag():
    cfg = get_config_lmp(make_spec(termination="O", ml=81))
    assert "variable    O_terminate equal true" in cfg
    assert "variable    O_ether_terminate equal false" in cfg


def test_config_o_ether_termination_flag():
    cfg = get_config_lmp(make_spec(termination="O_ether", ml=81))
    assert "variable    O_ether_terminate equal true" in cfg
    assert "variable    O_terminate equal false" in cfg


def test_config_o_1x1_termination_flag():
    """111 O_1x1 termination should set O_terminate=true."""
    cfg = get_config_lmp(make_spec(orientation="111", reconstruction="bare_1x1",
                                   termination="O_1x1", ml=90, box_x=5, box_y=9))
    assert "variable    O_terminate equal true" in cfg


def test_config_bare_termination_all_flags_false():
    cfg = get_config_lmp(make_spec(termination="bare", ml=81))
    assert "variable    O_terminate equal false" in cfg
    assert "variable    O_ether_terminate equal false" in cfg


def test_config_species_O_type_index():
    cfg = get_config_lmp(make_spec(species="O", ml=81))
    assert "incident_type_index equal 3" in cfg


def test_config_species_Ar_type_index():
    cfg = get_config_lmp(make_spec(species="Ar", ml=81))
    assert "incident_type_index equal 4" in cfg


def test_config_species_O2_energy_halving():
    cfg = get_config_lmp(make_spec(species="O2", energy=100.0, ml=81))
    assert "energ equal 50.0" in cfg


def test_config_species_O_energy_no_halving():
    cfg = get_config_lmp(make_spec(species="O", energy=100.0, ml=81))
    assert "energ equal 100.0" in cfg


def test_config_contains_M_Ar():
    cfg = get_config_lmp(make_spec(ml=81))
    assert "M_Ar equal 39.948" in cfg


def test_config_orientation_comment():
    cfg = get_config_lmp(make_spec(orientation="113", reconstruction="bare",
                                   ml=108, box_x=9, box_y=3))
    assert "orientation=113" in cfg


# ─── head.lmp tests ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("orientation,recon,ml,bx,by", [
    ("100", "bare_1x1", 81, 9, 9),
    ("111", "bare_1x1", 90, 5, 9),
    ("113", "bare",    108, 9, 3),
])
def test_head_contains_lattice_cmd(orientation, recon, ml, bx, by):
    spec = make_spec(orientation=orientation, reconstruction=recon,
                     ml=ml, box_x=bx, box_y=by)
    head = get_head_lmp(spec)
    assert ORIENT[orientation]["lattice_cmd"] in head


@pytest.mark.parametrize("orientation,recon,ml,bx,by", [
    ("100", "bare_1x1", 81, 9, 9),
    ("111", "bare_1x1", 90, 5, 9),
    ("113", "bare",    108, 9, 3),
])
def test_head_contains_bottom_expr(orientation, recon, ml, bx, by):
    spec = make_spec(orientation=orientation, reconstruction=recon,
                     ml=ml, box_x=bx, box_y=by)
    head = get_head_lmp(spec)
    assert ORIENT[orientation]["bottom_expr"] in head


def test_head_100_lattice_content():
    head = get_head_lmp(make_spec(orientation="100", ml=81))
    assert "orient z 0 0 1" in head
    assert "orient x 1 1 0" in head


def test_head_111_lattice_content():
    head = get_head_lmp(make_spec(orientation="111", reconstruction="bare_1x1",
                                  ml=90, box_x=5, box_y=9))
    assert "orient z 1 1 1" in head
    assert "orient x 2 -1 -1" in head


def test_head_113_lattice_content():
    head = get_head_lmp(make_spec(orientation="113", reconstruction="bare",
                                  ml=108, box_x=9, box_y=3))
    assert "orient z 1 1 3" in head
    assert "orient x -1 1 0" in head


def test_head_contains_energy_in_comment():
    head = get_head_lmp(make_spec(energy=2.5, ml=81))
    assert "2.5eV" in head


def test_head_contains_loop_structure():
    head = get_head_lmp(make_spec(ml=81))
    assert "label\t\tloop" in head
    assert "jump\t\tSELF loop" in head
    assert "label\t\tcontinue_impact" in head


def test_head_contains_ncarbon_output():
    head = get_head_lmp(make_spec(ml=81))
    assert "ncarbon.txt" in head


def test_head_Ar_hybrid_pair_style():
    head = get_head_lmp(make_spec(species="Ar", ml=81))
    assert "pair_style  hybrid reaxff" in head
    assert "zbl 5.0 6.0" in head


def test_head_Ar_zbl_pair_coeffs():
    head = get_head_lmp(make_spec(species="Ar", ml=81))
    assert "pair_coeff 1 4 zbl 6.0 18.0" in head
    assert "pair_coeff 4 4 zbl 18.0 18.0" in head


def test_head_Ar_qeq_nonargon():
    head = get_head_lmp(make_spec(species="Ar", ml=81))
    assert "fix reax_qeq nonargon" in head
    assert "fix reax_qeq all" not in head


def test_head_Ar_remove_after_impact():
    head = get_head_lmp(make_spec(species="Ar", ml=81))
    assert "delete_atoms group IonRemove" in head


def test_head_Ar_nonargon_regroup_in_loop():
    head = get_head_lmp(make_spec(species="Ar", ml=81))
    # nonargon group should appear after "group insert clear" in the loop
    assert "group nonargon type 1 2 3" in head


def test_head_Ar_four_masses():
    head = get_head_lmp(make_spec(species="Ar", ml=81))
    assert "mass        4 ${M_Ar}" in head


def test_head_O2_molecule_declaration():
    head = get_head_lmp(make_spec(species="O2", ml=81))
    assert "molecule O2 O2.molecule" in head


def test_head_O2_deposit_mol():
    head = get_head_lmp(make_spec(species="O2", ml=81))
    assert "mol O2" in head
    # type should be 0 for molecule injection
    assert "deposit 1 0 " in head


def test_head_O_no_zbl():
    head = get_head_lmp(make_spec(species="O", ml=81))
    assert "hybrid" not in head
    assert "zbl" not in head


def test_head_O_no_molecule():
    head = get_head_lmp(make_spec(species="O", ml=81))
    assert "molecule O2" not in head
    assert "mol O2" not in head


def test_head_O_qeq_all():
    head = get_head_lmp(make_spec(species="O", ml=81))
    assert "fix reax_qeq all" in head


def test_head_O_no_removal():
    head = get_head_lmp(make_spec(species="O", ml=81))
    assert "IonRemove" not in head


# ─── submit script tests ─────────────────────────────────────────────────────

def test_submit_contains_job_name():
    sub = get_submit_script(make_spec(name="my_diamond_sim"))
    assert "#SBATCH --job-name=my_diamond_sim" in sub


def test_submit_contains_wall_hours():
    sub = get_submit_script(make_spec(wall_hours=48))
    assert "#SBATCH --time=48:00:00" in sub


def test_submit_default_account():
    sub = get_submit_script(make_spec())
    assert "#SBATCH --account=dgraves" in sub


def test_submit_custom_account():
    sub = get_submit_script(make_spec(account="mygroup"))
    assert "#SBATCH --account=mygroup" in sub


def test_submit_no_email_by_default():
    sub = get_submit_script(make_spec())
    assert "--mail" not in sub


def test_submit_email_when_set():
    sub = get_submit_script(make_spec(email="user@example.com"))
    assert "#SBATCH --mail-type=END,FAIL" in sub
    assert "#SBATCH --mail-user=user@example.com" in sub


def test_submit_contains_gpu_request():
    sub = get_submit_script(make_spec())
    assert "#SBATCH --gres=gpu:1" in sub


def test_submit_contains_singleton_dependency():
    sub = get_submit_script(make_spec())
    assert "#SBATCH --dependency=singleton" in sub


def test_submit_contains_default_lammps_module():
    sub = get_submit_script(make_spec())
    assert "module load lammps/kokkos/gpu_della9_2022" in sub


def test_submit_custom_lammps_module():
    sub = get_submit_script(make_spec(lammps_module="lammps/kokkos/gpu_della10_2024"))
    assert "module load lammps/kokkos/gpu_della10_2024" in sub
    assert "gpu_della9_2022" not in sub


def test_submit_contains_head_lmp_call():
    sub = get_submit_script(make_spec())
    assert "-in head.lmp" in sub


def test_submit_contains_make_surf_call():
    sub = get_submit_script(make_spec())
    assert "-in make_surf.lmp" in sub


def test_submit_creates_dirs():
    sub = get_submit_script(make_spec())
    assert "mkdir -p dumps data_files" in sub


# ─── auto re-submit tests ───────────────────────────────────────────────────

def test_submit_captures_lmp_exit_code():
    sub = get_submit_script(make_spec())
    assert "lmp_exit=$?" in sub


def test_submit_skips_resubmit_on_lmp_failure():
    sub = get_submit_script(make_spec())
    assert "if [ $lmp_exit -ne 0 ]" in sub
    assert "not re-submitting" in sub


def test_submit_reads_end_c_from_config():
    sub = get_submit_script(make_spec())
    assert "ML=$(grep 'ML equal' config.lmp" in sub
    assert "end_fluence=$(grep 'end_fluence equal' config.lmp" in sub
    assert "end_c=$(( end_fluence * ML ))" in sub


def test_submit_resubmits_if_incomplete():
    sub = get_submit_script(make_spec())
    assert 'sbatch "$0"' in sub


def test_submit_skips_lammps_if_already_complete():
    sub = get_submit_script(make_spec())
    assert "Simulation already complete" in sub
    # The early-exit check should come before srun lmp
    already_idx = sub.index("Simulation already complete")
    srun_idx = sub.index("srun lmp -k on g 1")
    assert already_idx < srun_idx
