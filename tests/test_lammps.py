"""
tests/test_lammps.py — tests that generated LAMMPS input files contain expected content.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md.spec import SimSpec, IonComponent, CyclePhase
from diamond_etch_md.lammps.config import (get_config_lmp, get_config_lmp_multi_ion,
                                            get_config_lmp_single_impact,
                                            get_config_lmp_carbon_etch)
from diamond_etch_md.lammps.head import (get_head_lmp, get_head_lmp_multi_ion,
                                          get_head_lmp_carbon_etch)
from diamond_etch_md.lammps.head_cycling import get_head_lmp_cycle_etch
from diamond_etch_md.lammps.head_single_impact import (get_head_lmp_single_impact,
                                                        get_thermalize_surface_lmp)
from diamond_etch_md.lammps.submit import get_submit_script, get_submit_script_single_impact
from diamond_etch_md.orientations import ORIENT


def make_rie_spec(flux_ratio=5, radical_energy=0.2, **kw):
    """Helper: make a valid RIE-etch SimSpec."""
    defaults = dict(
        orientation="100", surface="1x1", species="O",
        energy=20.0, ml=81, name="rie_test",
    )
    defaults.update(kw)
    return SimSpec(flux_ratio=flux_ratio, radical_energy=radical_energy, **defaults)


# ─── fixtures ────────────────────────────────────────────────────────────────

def make_spec(orientation="100", energy=0.5, temperature=300.0, ml=81,
              box_x=9, box_y=9, box_depth=3, species="O",
              surface="1x1", name="test_job", **kw):
    return SimSpec(
        orientation=orientation,
        energy=energy,
        surface_temperature=temperature,
        ml=ml,
        box_x=box_x,
        box_y=box_y,
        box_depth=box_depth,
        species=species,
        surface=surface,
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
    cfg = get_config_lmp(make_spec(orientation="113", surface="",
                                   ml=108, box_x=9, box_y=3))
    assert "variable    ML equal 108" in cfg


def test_config_contains_box_x_and_y():
    cfg = get_config_lmp(make_spec(box_x=6, box_y=4, ml=6))
    assert "variable    x equal 6" in cfg
    assert "variable    y equal 4" in cfg


def test_config_recon_flag_100_2x1():
    cfg = get_config_lmp(make_spec(orientation="100", surface="2x1", ml=81))
    assert "variable    reconstruct equal true" in cfg


def test_config_recon_flag_100_1x1():
    cfg = get_config_lmp(make_spec(orientation="100", surface="1x1", ml=81))
    assert "variable    reconstruct equal false" in cfg


def test_config_recon_flag_111_always_false():
    """For 111, reconstruction is in the template; flag is always false."""
    cfg = get_config_lmp(make_spec(orientation="111", surface="2x1_pandey",
                                   ml=90, box_x=5, box_y=9))
    assert "variable    reconstruct equal false" in cfg


def test_config_o_terminate_flag():
    cfg = get_config_lmp(make_spec(surface="2x1_O", ml=81))
    assert "variable    O_terminate equal true" in cfg
    assert "variable    O_ether_terminate equal false" in cfg


def test_config_o_ether_flag():
    cfg = get_config_lmp(make_spec(surface="O_ether", ml=81))
    assert "variable    O_ether_terminate equal true" in cfg


def test_config_111_O_flag():
    """111 O-terminated surface should set O_terminate=true."""
    cfg = get_config_lmp(make_spec(orientation="111", surface="1x1_O",
                                   ml=90, box_x=5, box_y=9))
    assert "variable    O_terminate equal true" in cfg


def test_config_unterminated_all_flags_false():
    cfg = get_config_lmp(make_spec(surface="1x1", ml=81))
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
    # M_Ar is only emitted when species=Ar (ZBL species emit their own mass var)
    cfg = get_config_lmp(make_spec(species="Ar", ml=81))
    assert "M_Ar equal 39.948" in cfg


def test_config_no_unused_mass_vars():
    # Non-ZBL species (O) should not emit M_Ar or M_Er
    cfg = get_config_lmp(make_spec(species="O", ml=81))
    assert "M_Ar" not in cfg
    assert "M_Er" not in cfg


def test_config_orientation_comment():
    cfg = get_config_lmp(make_spec(orientation="113", surface="",
                                   ml=108, box_x=9, box_y=3))
    assert "orientation=113" in cfg


# ─── head.lmp tests ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("orientation,surface,ml,bx,by", [
    ("100", "1x1",      81, 9, 9),
    ("111", "1x1",      90, 5, 9),
    ("113", "",         108, 9, 3),
])
def test_head_contains_lattice_cmd(orientation, surface, ml, bx, by):
    spec = make_spec(orientation=orientation, surface=surface,
                     ml=ml, box_x=bx, box_y=by)
    head = get_head_lmp(spec)
    assert ORIENT[orientation]["lattice_cmd"] in head


@pytest.mark.parametrize("orientation,surface,ml,bx,by", [
    ("100", "1x1",      81, 9, 9),
    ("111", "1x1",      90, 5, 9),
    ("113", "",         108, 9, 3),
])
def test_head_contains_bottom_expr(orientation, surface, ml, bx, by):
    spec = make_spec(orientation=orientation, surface=surface,
                     ml=ml, box_x=bx, box_y=by)
    head = get_head_lmp(spec)
    assert ORIENT[orientation]["bottom_expr"] in head


def test_head_100_lattice_content():
    head = get_head_lmp(make_spec(orientation="100", ml=81))
    assert "orient z 0 0 1" in head
    assert "orient x 1 1 0" in head


def test_head_111_lattice_content():
    head = get_head_lmp(make_spec(orientation="111", surface="1x1",
                                  ml=90, box_x=5, box_y=9))
    assert "orient z 1 1 1" in head
    assert "orient x 2 -1 -1" in head


def test_head_113_lattice_content():
    head = get_head_lmp(make_spec(orientation="113", surface="",
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
    assert "pair_coeff 1 4 zbl 6.0 18" in head
    assert "pair_coeff 4 4 zbl 18 18" in head


def test_head_Ar_qeq_nonargon():
    head = get_head_lmp(make_spec(species="Ar", ml=81))
    assert "fix reax_qeq nonargon" in head
    assert "fix reax_qeq all" not in head


def test_head_Er_zbl_pair_coeffs():
    head = get_head_lmp(make_spec(species="Er", ml=81))
    assert "pair_coeff 1 4 zbl 6.0 68" in head
    assert "pair_coeff 4 4 zbl 68 68" in head


def test_head_Er_uses_M_Er_mass():
    head = get_head_lmp(make_spec(species="Er", ml=81))
    assert "mass        4 ${M_Er}" in head
    assert "mass        4 ${M_Ar}" not in head


def test_head_Er_incident_ion_remove():
    head = get_head_lmp(make_spec(species="Er", ml=81))
    assert "group       incident_ion type 4" in head
    assert "group       argon type 4" not in head


def test_head_Ar_no_explicit_removal():
    # Ar is now swept by sweep.lmp, not force-deleted post-impact
    head = get_head_lmp(make_spec(species="Ar", ml=81))
    assert "IonRemove" not in head


def test_head_Ar_nonargon_regroup_in_loop():
    head = get_head_lmp(make_spec(species="Ar", ml=81))
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
    assert "mkdir -p etch_event_trajs impact_snaps" in sub


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
    already_idx = sub.index("Simulation already complete")
    srun_idx = sub.index("srun lmp -k on g 1")
    assert already_idx < srun_idx


# ─── RIE-etch config.lmp tests ───────────────────────────────────────────────

def test_config_rie_contains_flux_ratio():
    cfg = get_config_lmp(make_rie_spec(flux_ratio=5))
    assert "variable    flux_ratio equal 5" in cfg


def test_config_rie_contains_radical_energy():
    cfg = get_config_lmp(make_rie_spec(radical_energy=0.3))
    assert "variable    radical_energy equal 0.3" in cfg


def test_config_rie_contains_radical_i_above():
    cfg = get_config_lmp(make_rie_spec())
    assert "variable    radical_i_above equal 6.0" in cfg


def test_config_rie_contains_inter_neutral_time():
    cfg = get_config_lmp(make_rie_spec(inter_neutral_time=800.0))
    assert "variable    inter_neutral_time equal 800.0" in cfg


def test_config_ion_etch_no_flux_ratio_variable():
    """ion-etch (flux_ratio==0) must NOT have flux_ratio LAMMPS variable."""
    cfg = get_config_lmp(make_spec(ml=81))
    assert "flux_ratio" not in cfg


# ─── RIE-etch head.lmp tests ─────────────────────────────────────────────────

def test_head_rie_contains_neutral_loop_label():
    head = get_head_lmp(make_rie_spec())
    assert "label       neutral_loop" in head


def test_head_rie_contains_skip_radicals_label():
    head = get_head_lmp(make_rie_spec())
    assert "label       skip_radicals" in head


def test_head_rie_contains_cn_start():
    head = get_head_lmp(make_rie_spec())
    assert "cn_start" in head


def test_head_rie_5col_ncarbon_format():
    """RIE-etch should write 5-col ncarbon.txt: c 0 ncarbon nhydrogen noxygen."""
    head = get_head_lmp(make_rie_spec())
    assert "${c} 0 ${ncarbon} ${nhydrogen} ${noxygen}" in head


def test_head_ion_etch_4col_ncarbon_format():
    """ion-etch should write 4-col ncarbon.txt: c ncarbon nhydrogen noxygen."""
    head = get_head_lmp(make_spec(ml=81))
    assert "${c} ${ncarbon} ${nhydrogen} ${noxygen}" in head
    assert "${c} 0 ${ncarbon}" not in head


def test_head_rie_data_file_with_cn_suffix():
    """RIE-etch ion snapshot goes to impact_snaps/${c}_0.data."""
    head = get_head_lmp(make_rie_spec())
    assert "impact_snaps/${c}_0.data" in head


def test_head_ion_etch_data_file_no_suffix():
    """ion-etch snapshot goes to impact_snaps/${c}.data (no _cn suffix)."""
    head = get_head_lmp(make_spec(ml=81))
    assert "impact_snaps/${c}.data" in head
    assert "impact_snaps/${c}_0.data" not in head


def test_head_rie_no_neutral_loop_when_zero_flux():
    """ion-etch must NOT contain the radical loop labels."""
    head = get_head_lmp(make_spec(ml=81))
    assert "neutral_loop" not in head
    assert "skip_radicals" not in head


# ─── RIE-etch submit script tests ────────────────────────────────────────────

def test_submit_rie_contains_neut_complete():
    sub = get_submit_script(make_rie_spec())
    assert "neut_complete" in sub


def test_submit_rie_reads_cn_start():
    sub = get_submit_script(make_rie_spec())
    assert "cn_start" in sub
    assert "NF>=5{print $2}" in sub


def test_submit_ion_etch_no_neut_complete():
    sub = get_submit_script(make_spec(ml=81))
    assert "neut_complete" not in sub
    assert "cn_start" not in sub


# ─── Multi-ion tests ──────────────────────────────────────────────────────────

def make_multi_ion_spec(flux_ratio=5, **kw):
    defaults = dict(
        orientation="100", surface="1x1", ml=81,
        flux_ratio=flux_ratio, radical_energy=0.2,
        ion_mix=[
            IonComponent(species="O",  fraction=0.5, energy=50.0),
            IonComponent(species="O2", fraction=0.5, energy=100.0),
        ],
    )
    defaults.update(kw)
    return SimSpec(**defaults)


def test_multi_ion_config_no_single_energ():
    cfg = get_config_lmp_multi_ion(make_multi_ion_spec())
    assert "variable    energ" not in cfg
    assert "incident_type_index" not in cfg


def test_multi_ion_config_has_i_above():
    cfg = get_config_lmp_multi_ion(make_multi_ion_spec())
    assert "i_above" in cfg


def test_multi_ion_config_has_mass_vars():
    # O+O2 mix: C/H/O mass vars needed. M_Ar is ALSO required even with no ZBL
    # species in the mix — head.lmp always allocates 4 atom types and falls
    # back to `mass 4 ${M_Ar}` as a placeholder for the unused 4th slot.
    cfg = get_config_lmp_multi_ion(make_multi_ion_spec())
    for var in ("M_C", "M_H", "M_O", "M_Ar"):
        assert var in cfg
    assert cfg.count("M_Ar equal") == 1  # defined exactly once, not duplicated


def test_multi_ion_config_rie_vars_present():
    cfg = get_config_lmp_multi_ion(make_multi_ion_spec(flux_ratio=5))
    assert "flux_ratio" in cfg
    assert "radical_energy" in cfg


def test_multi_ion_config_no_rie_vars_when_ion_etch():
    cfg = get_config_lmp_multi_ion(make_multi_ion_spec(flux_ratio=0))
    assert "flux_ratio" not in cfg


def test_multi_ion_head_has_ion_sel_labels():
    head = get_head_lmp_multi_ion(make_multi_ion_spec())
    assert "label       ion_sel_0" in head
    assert "label       ion_sel_1" in head
    assert "label       ion_sel_done" in head


def test_multi_ion_head_has_random_var():
    head = get_head_lmp_multi_ion(make_multi_ion_spec())
    assert "variable    r equal random" in head


def test_multi_ion_head_has_ion_impacts_print():
    head = get_head_lmp_multi_ion(make_multi_ion_spec())
    assert "ion_impacts.txt" in head
    assert "${cur_ion}" in head


def test_multi_ion_head_conditional_deposit():
    head = get_head_lmp_multi_ion(make_multi_ion_spec())
    assert "cur_ion_is_mol" in head
    assert "mol O2" in head


def test_multi_ion_head_no_zbl_for_o_o2_mix():
    head = get_head_lmp_multi_ion(make_multi_ion_spec())
    assert "zbl" not in head


def test_multi_ion_head_zbl_for_ar_mix():
    spec = SimSpec(
        ml=81,
        ion_mix=[
            IonComponent("Ar", 0.5, 100.0),
            IonComponent("O",  0.5, 50.0),
        ],
        flux_ratio=0,
    )
    head = get_head_lmp_multi_ion(spec)
    assert "zbl" in head
    assert "nonargon" in head


def test_multi_ion_head_no_explicit_ar_removal():
    # Ar is now swept by sweep.lmp; no IonRemove block should appear in head.lmp
    spec = SimSpec(
        ml=81,
        ion_mix=[
            IonComponent("Ar", 0.5, 100.0),
            IonComponent("O",  0.5, 50.0),
        ],
        flux_ratio=0,
    )
    head = get_head_lmp_multi_ion(spec)
    assert "IonRemove" not in head


def test_multi_ion_head_has_radical_loop_when_rie():
    head = get_head_lmp_multi_ion(make_multi_ion_spec(flux_ratio=5))
    assert "neutral_loop" in head


def test_multi_ion_head_no_radical_loop_when_ion_etch():
    head = get_head_lmp_multi_ion(make_multi_ion_spec(flux_ratio=0))
    assert "neutral_loop" not in head


def test_multi_ion_head_ncarbon_4col_when_ion_etch():
    head = get_head_lmp_multi_ion(make_multi_ion_spec(flux_ratio=0))
    assert '"${c} ${ncarbon} ${nhydrogen} ${noxygen}" append ncarbon.txt' in head


def test_multi_ion_head_ncarbon_5col_when_rie():
    head = get_head_lmp_multi_ion(make_multi_ion_spec(flux_ratio=5))
    assert '"${c} 0 ${ncarbon} ${nhydrogen} ${noxygen}" append ncarbon.txt' in head


def test_multi_ion_head_molecule_decl_for_o2():
    head = get_head_lmp_multi_ion(make_multi_ion_spec())
    assert "molecule O2 O2.molecule" in head


def test_multi_ion_head_no_molecule_decl_for_o_only():
    spec = SimSpec(
        ml=81,
        ion_mix=[
            IonComponent("O", 0.6, 50.0),
            IonComponent("O", 0.4, 100.0),
        ],
        flux_ratio=0,
    )
    head = get_head_lmp_multi_ion(spec)
    assert "molecule O2" not in head


def test_multi_ion_head_three_ions_all_labels():
    spec = SimSpec(
        ml=81,
        ion_mix=[
            IonComponent("O",  0.1, 20.0),
            IonComponent("O",  0.2, 10.0),
            IonComponent("O",  0.7, 5.0),
        ],
        flux_ratio=0,
    )
    head = get_head_lmp_multi_ion(spec)
    assert "label       ion_sel_0" in head
    assert "label       ion_sel_1" in head
    assert "label       ion_sel_2" in head
    # Two cumulative-threshold if-checks and one fallback jump
    assert head.count('if "${r} <') == 2
    assert "jump        SELF ion_sel_2" in head


# ─── cycle-etch head: potential switching ────────────────────────────────────

def _make_3phase_spec():
    return SimSpec(
        ml=64,
        box_x=8, box_y=8,
        phases=[
            CyclePhase(species="Ar", energy=50.0, fluence_ml=5),
            CyclePhase(species="O2", energy=20.0, fluence_ml=5, flux_ratio=2, radical_energy=0.2),
            CyclePhase(species="O",  energy=1.0,  fluence_ml=3),
        ],
        cycles=2,
    )


def test_cycle_head_zbl_to_plain_uses_C_not_NULL():
    # Regression: switching Ar→O2 must use "C H O C", not "C H O NULL"
    # "C H O NULL" is only valid in hybrid pair_style and causes
    # "All pair coeffs are not set" when plain reaxff is active.
    head = get_head_lmp_cycle_etch(_make_3phase_spec())
    # The ZBL→plain switch block
    assert '"pair_coeff * * ffield.reax C H O C"' in head
    # The plain→ZBL switch block still uses NULL (correct for hybrid)
    assert '"pair_coeff * * reaxff ffield.reax C H O NULL"' in head


def test_cycle_head_has_potential_switch_block():
    head = get_head_lmp_cycle_etch(_make_3phase_spec())
    assert "prev_needs_zbl" in head
    assert "current_needs_zbl" in head


# ─── cycle-etch head: nonargon group ordering ─────────────────────────────────

def test_cycle_head_nonargon_group_before_qeq():
    # Regression: "group nonargon type 1 2 3" must appear before
    # "fix reax_qeq nonargon" so the group exists when the fix is created.
    head = get_head_lmp_cycle_etch(_make_3phase_spec())
    group_pos = head.index("group nonargon")
    qeq_pos   = head.index("fix reax_qeq nonargon")
    assert group_pos < qeq_pos, (
        "group nonargon must be defined before fix reax_qeq nonargon"
    )


# ─── cycle-etch head: radical energy only when radicals exist ─────────────────

def test_cycle_head_no_radical_energy_var_when_all_ion_etch():
    # All flux_ratio=0 → no radicals → current_radical_energy must not appear.
    spec = SimSpec(
        ml=64,
        box_x=8, box_y=8,
        phases=[
            CyclePhase(species="Ar", energy=50.0, fluence_ml=5),
            CyclePhase(species="O",  energy=20.0, fluence_ml=5, flux_ratio=0),
        ],
        cycles=2,
    )
    head = get_head_lmp_cycle_etch(spec)
    assert "current_radical_energy" not in head, (
        "current_radical_energy must not be emitted when no phase has radicals"
    )


def test_cycle_head_radical_energy_var_present_when_any_rie():
    # At least one flux_ratio>0 → current_radical_energy must appear.
    spec = SimSpec(
        ml=64,
        box_x=8, box_y=8,
        phases=[
            CyclePhase(species="Ar", energy=50.0, fluence_ml=5),
            CyclePhase(species="O",  energy=20.0, fluence_ml=5, flux_ratio=3,
                       radical_energy=0.2),
        ],
        cycles=2,
    )
    head = get_head_lmp_cycle_etch(spec)
    assert "current_radical_energy" in head


# ─── single-impact head.lmp ───────────────────────────────────────────────────

def _make_si_spec(**kw):
    defaults = dict(
        species="O", energy=100.0, surface_temperature=300.0,
        single_impact=True, n_trials=200,
        orientation="100", surface="1x1", ml=81, box_x=9, box_y=9,
    )
    defaults.update(kw)
    return SimSpec(**defaults)


def test_single_impact_head_reads_thermalized_data():
    head = get_head_lmp_single_impact(_make_si_spec())
    # Old LAMMPS 2022 syntax: delete_atoms first, then read_data add 0
    assert "delete_atoms group all" in head
    assert "read_data   thermalized.data add 0" in head


def test_single_impact_head_regroups_after_read_data():
    # Groups must be re-declared after read_data to restore per-atom memberships.
    head = get_head_lmp_single_impact(_make_si_spec())
    # anchor must appear both in initial setup AND inside the trial loop
    occurrences = head.count("group       anchor region anchor")
    assert occurrences >= 2, "anchor group must be declared in setup and in each trial"


def test_single_impact_head_surf_z_immediate_eval():
    # surf_z_before must use $(…) immediate evaluation so it captures pre-impact z.
    head = get_head_lmp_single_impact(_make_si_spec())
    assert "variable    surf_z_before equal $(bound(carbon,zmax))" in head


def test_single_impact_head_cn_start_reset_per_trial():
    # cn_start must be reset to 0 in every trial (radical loop restart counter).
    head = get_head_lmp_single_impact(_make_si_spec())
    assert "variable    cn_start equal 0" in head


def test_single_impact_head_writes_ntrials_done():
    head = get_head_lmp_single_impact(_make_si_spec())
    assert "shell       echo ${trial} > ntrials_done.txt" in head


def test_single_impact_head_nonargon_before_qeq_for_ar():
    # For Ar ions (ZBL), the nonargon group must be declared before fix reax_qeq nonargon.
    head = get_head_lmp_single_impact(_make_si_spec(species="Ar", energy=50.0))
    # Find the in-loop declarations (the loop_reset section comes after label trial_start)
    trial_start = head.index("label\t\ttrial_start")
    loop_part   = head[trial_start:]
    group_pos   = loop_part.index("group       nonargon type 1 2 3")
    qeq_pos     = loop_part.index("fix         reax_qeq nonargon")
    assert group_pos < qeq_pos


def test_single_impact_head_no_nonargon_for_o():
    head = get_head_lmp_single_impact(_make_si_spec(species="O", energy=50.0))
    trial_start = head.index("label\t\ttrial_start")
    loop_part   = head[trial_start:]
    assert "group       nonargon" not in loop_part
    assert "fix         reax_qeq all" in loop_part


def test_single_impact_head_randomize_velocities_on():
    head = get_head_lmp_single_impact(_make_si_spec(randomize_velocities=True))
    assert "velocity    all create" in head


def test_single_impact_head_randomize_velocities_off():
    head = get_head_lmp_single_impact(_make_si_spec(randomize_velocities=False))
    assert "velocity    all create" not in head


# ─── single-impact config.lmp ────────────────────────────────────────────────

def test_single_impact_config_n_trials():
    cfg = get_config_lmp_single_impact(_make_si_spec(n_trials=777))
    assert "n_trials equal 777" in cfg


def test_single_impact_config_crystal_path_has_lat_a():
    cfg = get_config_lmp_single_impact(_make_si_spec())
    assert "lat_a file lat_a.txt" in cfg


def test_single_impact_config_carbon_path_has_sqrt2_lat_a():
    cfg = get_config_lmp_single_impact(SimSpec(
        species="O", energy=50.0, surface_temperature=300.0,
        single_impact=True, n_trials=100,
        initial_config_file="/fake/path.data", anchor_z_max=8.0,
    ))
    assert "lat_a equal 1.414" in cfg
    assert "anchor_z_max equal 8.0" in cfg


def test_single_impact_config_rie_adds_radical_block():
    cfg = get_config_lmp_single_impact(_make_si_spec(
        flux_ratio=5, radical_energy=0.3,
    ))
    assert "flux_ratio equal 5" in cfg
    assert "radical_energy equal 0.3" in cfg


# ─── single-impact submit script ─────────────────────────────────────────────

def test_single_impact_submit_reads_ntrials_done():
    sub = get_submit_script_single_impact(_make_si_spec(name="si_test"))
    assert "ntrials_done.txt" in sub
    assert "-var n_complete $n_complete" in sub


def test_single_impact_submit_crystal_uses_impact_snaps_0():
    sub = get_submit_script_single_impact(_make_si_spec())
    assert "data_file=impact_snaps/0.data" in sub
    assert "make_surf.lmp" in sub


def test_single_impact_submit_carbon_uses_initial_config():
    sub = get_submit_script_single_impact(SimSpec(
        species="O", energy=50.0, surface_temperature=300.0,
        single_impact=True, n_trials=100,
        initial_config_file="/fake/path.data", anchor_z_max=8.0,
        name="si_carbon",
    ))
    assert "data_file=initial_config.data" in sub
    assert "make_surf.lmp" not in sub


def test_single_impact_submit_n_trials_not_in_lmp_args():
    # n_trials is defined in config.lmp (equal style); passing it via -var would
    # conflict (LAMMPS can't redefine equal→index).  Submit reads it via grep only.
    sub = get_submit_script_single_impact(_make_si_spec(n_trials=500))
    assert "-var n_trials" not in sub
    assert "n_trials equal' config.lmp" in sub or "n_trials equal" in sub
    assert "n_events 0" in sub   # always 0; no mid-run carry-over


# ── thermalize_surface.lmp (carbon path) ─────────────────────────────────────

def _make_si_carbon_spec(**kw):
    defaults = dict(
        species="Ar", energy=50.0, surface_temperature=300.0,
        single_impact=True, n_trials=100,
        initial_config_file="/fake/graphullerene.data", anchor_z_max=7.0,
        initial_thermalization=True, initial_thermalization_steps=10_000_000,
        randomize_velocities=True,
    )
    defaults.update(kw)
    return SimSpec(**defaults)


def test_thermalize_surface_standalone_header():
    lmp = get_thermalize_surface_lmp(_make_si_carbon_spec())
    assert "package kokkos" in lmp
    assert "include     config.lmp" in lmp
    assert "read_data   ${data_file}" in lmp


def test_thermalize_surface_writes_thermalized_data():
    lmp = get_thermalize_surface_lmp(_make_si_carbon_spec())
    assert "write_data  thermalized.data nofix nocoeff" in lmp


def test_thermalize_surface_writes_impact_stats_header():
    lmp = get_thermalize_surface_lmp(_make_si_carbon_spec())
    assert "impact_stats.txt" in lmp
    assert "trial surf_z_before" in lmp


def test_thermalize_surface_nvt_steps():
    lmp = get_thermalize_surface_lmp(_make_si_carbon_spec(initial_thermalization_steps=5_000_000))
    assert "run         5000000" in lmp


def test_thermalize_surface_no_nvt_when_disabled():
    lmp = get_thermalize_surface_lmp(_make_si_carbon_spec(initial_thermalization=False))
    assert "run         0" in lmp


def test_thermalize_surface_no_stopclust():
    lmp = get_thermalize_surface_lmp(_make_si_carbon_spec())
    cmd_lines = [l for l in lmp.splitlines() if not l.strip().startswith("#")]
    assert not any("stopclust" in l for l in cmd_lines)
    assert not any("fix halt" in l for l in cmd_lines)


def test_thermalize_surface_has_minimize():
    lmp = get_thermalize_surface_lmp(_make_si_carbon_spec())
    assert "minimize" in lmp
    assert "minimfreeze" in lmp


def test_single_impact_head_carbon_no_therm_guard():
    # Carbon path: thermalization is external (thermalize_surface.lmp); head.lmp
    # must NOT contain the if-n_complete==0 guard.
    spec = _make_si_carbon_spec()
    head = get_head_lmp_single_impact(spec)
    assert "init_thermalization.lmp" not in head
    assert 'if "${n_complete} == 0" then "write_data thermalized.data' not in head


def test_single_impact_head_crystal_keeps_therm_guard():
    # Crystal path still handles thermalization inside head.lmp.
    head = get_head_lmp_single_impact(_make_si_spec(initial_thermalization=True))
    assert "init_thermalization.lmp" in head
    assert "write_data thermalized.data" in head


def test_single_impact_submit_carbon_calls_thermalize_surface():
    sub = get_submit_script_single_impact(_make_si_carbon_spec())
    assert "thermalize_surface.lmp" in sub
    assert "thermalized.data" in sub   # guards the call


def test_single_impact_submit_crystal_no_thermalize_surface():
    sub = get_submit_script_single_impact(_make_si_spec())
    assert "thermalize_surface.lmp" not in sub


def test_single_impact_stats_logged_before_remove_ar():
    # ion_in_box / ion_z_after must be recorded BEFORE remove_ar deletes the ion.
    spec = _make_si_carbon_spec(remove_ar=True)
    head = get_head_lmp_single_impact(spec)
    stats_pos   = head.index("ion_in_box equal count(insert)")
    remove_pos  = head.index("delete_atoms group incident_ion")
    assert stats_pos < remove_pos, (
        "stats_log must appear before remove_ar_block; "
        "otherwise count(insert) is always 0"
    )


# ─── Deposition mask ───────────────────────────────────────────────────────────
# mask_type / mask_width restrict ion/radical deposit + bzone bookkeeping to a
# sub-window (expose_zone) of the box. Not yet committed to git; no prior tests.

def _make_carbon_spec(**kw):
    defaults = dict(
        species="O", energy=20.0, surface_temperature=300.0, ml=64, fluence=50,
        initial_config_file="/fake/path.data", anchor_z_max=7.0,
    )
    defaults.update(kw)
    return SimSpec(**defaults)


# -- config.lmp: mask variable block, all four config generators --

@pytest.mark.parametrize("get_cfg,spec_kwargs", [
    (get_config_lmp,               dict()),
    (get_config_lmp_multi_ion,     dict(ion_mix=[
        IonComponent(species="O",  fraction=0.5, energy=50.0),
        IonComponent(species="O2", fraction=0.5, energy=100.0),
    ])),
    (get_config_lmp_single_impact, dict(single_impact=True, n_trials=50)),
])
def test_config_no_mask_by_default(get_cfg, spec_kwargs):
    spec = make_spec(**spec_kwargs)
    cfg = get_cfg(spec)
    assert "mask_width_x" not in cfg
    assert "mask_width_y" not in cfg


def test_config_carbon_etch_no_mask_by_default():
    cfg = get_config_lmp_carbon_etch(_make_carbon_spec())
    assert "mask_width_x" not in cfg
    assert "mask_width_y" not in cfg


@pytest.mark.parametrize("get_cfg,spec_kwargs", [
    (get_config_lmp,               dict()),
    (get_config_lmp_multi_ion,     dict(ion_mix=[
        IonComponent(species="O",  fraction=0.5, energy=50.0),
        IonComponent(species="O2", fraction=0.5, energy=100.0),
    ])),
    (get_config_lmp_single_impact, dict(single_impact=True, n_trials=50)),
])
def test_config_xymask_defines_both_axes(get_cfg, spec_kwargs):
    spec = make_spec(mask_type="xymask", mask_width=0.3, **spec_kwargs)
    cfg = get_cfg(spec)
    assert "variable    mask_width_x equal 0.3" in cfg
    assert "variable    mask_width_y equal 0.3" in cfg
    assert "mask_lo_x    equal xlo+(xhi-xlo)*v_mask_width_x" in cfg
    assert "mask_hi_x    equal xhi-(xhi-xlo)*v_mask_width_x" in cfg
    assert "mask_lo_y    equal ylo+(yhi-ylo)*v_mask_width_y" in cfg
    assert "mask_hi_y    equal yhi-(yhi-ylo)*v_mask_width_y" in cfg


@pytest.mark.parametrize("get_cfg,spec_kwargs", [
    (get_config_lmp,               dict()),
    (get_config_lmp_single_impact, dict(single_impact=True, n_trials=50)),
])
def test_config_xmask_defines_only_x(get_cfg, spec_kwargs):
    spec = make_spec(mask_type="xmask", mask_width=0.2, **spec_kwargs)
    cfg = get_cfg(spec)
    assert "mask_width_x equal 0.2" in cfg
    assert "mask_width_y" not in cfg


@pytest.mark.parametrize("get_cfg,spec_kwargs", [
    (get_config_lmp,               dict()),
    (get_config_lmp_single_impact, dict(single_impact=True, n_trials=50)),
])
def test_config_ymask_defines_only_y(get_cfg, spec_kwargs):
    spec = make_spec(mask_type="ymask", mask_width=0.25, **spec_kwargs)
    cfg = get_cfg(spec)
    assert "mask_width_y equal 0.25" in cfg
    assert "mask_width_x" not in cfg


def test_config_carbon_etch_xymask():
    cfg = get_config_lmp_carbon_etch(_make_carbon_spec(mask_type="xymask", mask_width=0.4))
    assert "mask_width_x equal 0.4" in cfg
    assert "mask_width_y equal 0.4" in cfg


# -- head.lmp: expose_zone region + deposit uses it instead of bbox --

def test_head_no_mask_deposit_uses_bbox():
    head = get_head_lmp(make_spec())
    assert "region expose_zone" not in head
    assert "region bbox units box" in head


def test_head_xymask_deposit_uses_expose_zone():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.3))
    assert "region      expose_zone block $(v_mask_lo_x) $(v_mask_hi_x) $(v_mask_lo_y) $(v_mask_hi_y) EDGE EDGE units box" in head
    assert "region expose_zone units box" in head
    assert "region bbox units box" not in head


def test_head_xmask_expose_zone_full_y_extent():
    head = get_head_lmp(make_spec(mask_type="xmask", mask_width=0.2))
    assert "region      expose_zone block $(v_mask_lo_x) $(v_mask_hi_x) EDGE EDGE EDGE EDGE units box" in head


def test_head_ymask_expose_zone_full_x_extent():
    head = get_head_lmp(make_spec(mask_type="ymask", mask_width=0.2))
    assert "region      expose_zone block EDGE EDGE $(v_mask_lo_y) $(v_mask_hi_y) EDGE EDGE units box" in head


def test_head_rie_radical_deposit_also_uses_expose_zone():
    # Radical pre-exposure deposit (not just the ion deposit) must respect the mask.
    spec = make_rie_spec(mask_type="xymask", mask_width=0.3)
    head = get_head_lmp(spec)
    assert "region expose_zone units box" in head
    # two deposit fixes should reference the mask: one for radicals, one for the ion
    assert head.count("region expose_zone units box") >= 2


def test_head_multi_ion_mask_uses_expose_zone():
    spec = make_multi_ion_spec(mask_type="xymask", mask_width=0.3)
    head = get_head_lmp_multi_ion(spec)
    assert "region expose_zone units box" in head


def test_head_carbon_etch_no_mask_uses_bbox():
    head = get_head_lmp_carbon_etch(_make_carbon_spec())
    assert "region expose_zone" not in head


def test_head_carbon_etch_mask_uses_expose_zone():
    head = get_head_lmp_carbon_etch(_make_carbon_spec(mask_type="xymask", mask_width=0.3))
    assert "region expose_zone" in head


def test_head_single_impact_no_mask_uses_bbox():
    head = get_head_lmp_single_impact(_make_si_spec())
    assert "region expose_zone" not in head
    assert "region bbox units box" in head


def test_head_single_impact_mask_uses_expose_zone():
    head = get_head_lmp_single_impact(_make_si_spec(mask_type="xymask", mask_width=0.3))
    assert "region expose_zone units box" in head


# ─── invert_mask: side out in expose_zone + subtract in burst bzone ──────────

def test_head_invert_mask_xymask_expose_zone_has_side_out():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2, invert_mask=True))
    assert "EDGE EDGE side out units box" in head


def test_head_normal_mask_expose_zone_no_side_out():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2))
    assert "side out" not in head


def test_head_invert_mask_burst_uses_intersect_frame():
    spec = make_rie_spec(mask_type="xymask", mask_width=0.2, invert_mask=True,
                         radical_burst=True, radical_burst_chunk=4)
    head = get_head_lmp(spec)
    assert "intersect 2" in head
    region_lines = [l for l in head.splitlines() if l.strip().startswith("region")]
    assert all("subtract" not in l for l in region_lines), "region subtract found in output"


def test_head_normal_mask_burst_is_plain_block():
    spec = make_rie_spec(mask_type="xymask", mask_width=0.2,
                         radical_burst=True, radical_burst_chunk=4)
    head = get_head_lmp(spec)
    assert "intersect 2" not in head


# ─── freeze_mask: top-layer anchor freeze ─────────────────────────────────────

def test_head_freeze_mask_absent_when_disabled():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2, freeze_mask=False))
    assert "freeze_mask_zone" not in head


def test_head_freeze_mask_normal_uses_intersect_frame():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2, freeze_mask=True))
    assert "intersect 2 freeze_mask_zone_full freeze_mask_zone_inner" in head
    region_lines = [l for l in head.splitlines() if l.strip().startswith("region")]
    assert all("subtract" not in l for l in region_lines), "region subtract found in output"


def test_head_freeze_mask_inverted_is_plain_block():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2,
                                  invert_mask=True, freeze_mask=True))
    assert "freeze_mask_zone_full" not in head
    assert "freeze_mask_zone" in head


def test_head_freeze_mask_anchor_union_present():
    head = get_head_lmp(make_spec(mask_type="xmask", mask_width=0.2, freeze_mask=True))
    assert "anchor union anchor freeze_mask_atoms" in head


def test_head_freeze_mask_uses_ncarbon_for_restart_detection():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2, freeze_mask=True))
    assert '"${n_complete} == 0"' in head or "${n_complete} == 0" in head


def test_head_freeze_mask_writes_file():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2, freeze_mask=True))
    assert "freeze_mask_z.txt" in head


def test_head_freeze_mask_depth_in_expression():
    head = get_head_lmp(make_spec(mask_type="xmask", mask_width=0.2,
                                  freeze_mask=True, freeze_mask_depth=1.5))
    assert "zmax) - 1.5" in head
    assert "v_z_freeze_lo+2.0" in head  # 1.5 + 0.5


def test_head_freeze_mask_default_depth():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2, freeze_mask=True))
    assert "zmax) - 2.0" in head
    assert "v_z_freeze_lo+2.5" in head  # 2.0 + 0.5


def test_head_freeze_mask_before_mobile_group():
    head = get_head_lmp(make_spec(mask_type="xymask", mask_width=0.2, freeze_mask=True))
    freeze_pos  = head.index("freeze_mask_zone")
    mobile_pos  = head.index("group \tmobile subtract all anchor")
    assert freeze_pos < mobile_pos, "freeze_mask_block must appear before group mobile"


def test_head_multi_ion_freeze_mask():
    spec = make_multi_ion_spec(mask_type="xymask", mask_width=0.2, freeze_mask=True)
    head = get_head_lmp_multi_ion(spec)
    assert "anchor union anchor freeze_mask_atoms" in head


def test_head_carbon_etch_freeze_mask():
    head = get_head_lmp_carbon_etch(_make_carbon_spec(mask_type="xmask", mask_width=0.2,
                                                      freeze_mask=True))
    assert "anchor union anchor freeze_mask_atoms" in head


def test_head_single_impact_mask_redefines_expose_zone_each_trial():
    # bbox is deleted/redefined at the top of every trial; expose_zone must be too,
    # otherwise it would still reference the previous trial's (deleted) bbox.
    head = get_head_lmp_single_impact(_make_si_spec(mask_type="xymask", mask_width=0.3))
    trial_start = head.index("label\t\ttrial_start")
    loop_part = head[trial_start:]
    assert "expose_zone delete" in loop_part


# ─── step_edge config variables ───────────────────────────────────────────────

def test_config_step_edge_disabled_by_default():
    cfg = get_config_lmp(make_rie_spec())
    assert "step_edge      equal 0" in cfg


def test_config_step_edge_enabled():
    cfg = get_config_lmp(make_rie_spec(step_edge=True))
    assert "step_edge      equal 1" in cfg
    assert "step_angle     equal 0.0" in cfg
    assert "step_position  equal 0.5" in cfg
    assert "step_invert    equal 0" in cfg


def test_config_step_edge_orientation_default_depth_100():
    cfg = get_config_lmp(make_rie_spec(orientation="100", step_edge=True))
    assert "step_depth_ang equal 2.0" in cfg


def test_config_step_edge_orientation_default_depth_111():
    cfg = get_config_lmp(make_rie_spec(orientation="111", surface="1x1", step_edge=True))
    assert "step_depth_ang equal 2.2" in cfg


def test_config_step_edge_custom_depth():
    cfg = get_config_lmp(make_rie_spec(step_edge=True, step_depth=1.8))
    assert "step_depth_ang equal 1.8" in cfg


def test_config_step_edge_angle_and_invert():
    cfg = get_config_lmp(make_rie_spec(step_edge=True, step_angle=45.0, step_invert=True))
    assert "step_angle     equal 45.0" in cfg
    assert "step_invert    equal 1" in cfg


def test_config_step_edge_multi_ion():
    spec = make_multi_ion_spec()
    spec = spec.__class__(**{**vars(spec), 'step_edge': True})
    cfg = get_config_lmp_multi_ion(spec)
    assert "step_edge      equal 1" in cfg


# ─── make_surf templates contain step block ───────────────────────────────────

TEMPLATE_DIR = Path(__file__).parents[1] / "diamond_etch_md" / "lammps" / "templates"


@pytest.mark.parametrize("fname", [
    "make_surf_100.lmp",
    "make_surf_110.lmp",
    "make_surf_111_1x1.lmp",
    "make_surf_111_2x1_pandey.lmp",
    "make_surf_111_2x1_single.lmp",
    "make_surf_113.lmp",
])
def test_make_surf_contains_step_block(fname):
    src = (TEMPLATE_DIR / fname).read_text()
    assert 'if "${step_edge} == 1"' in src
    assert "variable stp_sel atom" in src
    assert "group    step_rm  variable stp_sel" in src
    assert "delete_atoms group step_rm compress no" in src


@pytest.mark.parametrize("fname", [
    "make_surf_100.lmp",
    "make_surf_110.lmp",
    "make_surf_111_1x1.lmp",
    "make_surf_111_2x1_pandey.lmp",
    "make_surf_111_2x1_single.lmp",
    "make_surf_113.lmp",
])
def test_make_surf_step_block_before_final_minimize(fname):
    src = (TEMPLATE_DIR / fname).read_text()
    step_pos = src.index('if "${step_edge} == 1"')
    # find the last minimize (before write_data)
    last_min_pos = src.rindex("minimize 1.0e-5")
    assert step_pos < last_min_pos, "step block must come before the final minimize"
