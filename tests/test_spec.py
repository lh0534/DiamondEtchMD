"""
tests/test_spec.py — tests for SimSpec, compute_ml, and validate().
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md.spec import SimSpec, CyclePhase, IonComponent, compute_ml, validate, etch_mode, normalize_ion_mix
from diamond_etch_md.builder import multi_ion_dir_name


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
    spec = SimSpec(orientation="100", surface="1x1", species="Unobtainium", ml=81)
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
    assert s.surface_temperature == 300.0
    assert s.species == "O"
    assert s.energy == 0.5
    assert s.ion_angle == 0.0
    assert s.fluence == 50
    assert s.ml == 0
    assert s.box_x == 9
    assert s.box_y == 9
    assert s.box_depth == 3
    assert s.impact_time == 1000.0
    assert s.thermalization_time == 500.0
    assert s.wall_hours == 24
    assert s.name == ""
    assert s.account == "dgraves"
    assert s.email == ""
    assert s.lammps_module == "lammps/kokkos/gpu_della9_2022"
    assert s.plot_interval_hours == 12
    assert s.flux_ratio == 0
    assert s.radical_energy == 0.2


# ─── etch_mode() ──────────────────────────────────────────────────────────────

def test_etch_mode_ion_etch():
    s = SimSpec(orientation="100", surface="1x1", species="O", ml=81)
    assert etch_mode(s) == "ion-etch"


def test_etch_mode_rie_etch():
    s = SimSpec(orientation="100", surface="1x1", species="O", ml=81, flux_ratio=5)
    assert etch_mode(s) == "rie-etch"


def test_etch_mode_cycle_etch():
    s = SimSpec(
        orientation="100", surface="1x1", ml=81,
        phases=[
            CyclePhase(species="Ar", energy=30.0, fluence_ml=5),
            CyclePhase(species="O2", energy=20.0, fluence_ml=5),
        ],
    )
    assert etch_mode(s) == "cycle-etch"


# ─── RIE-etch validation ──────────────────────────────────────────────────────

def test_validate_rie_etch_Ar_passes():
    """Ar + O• radicals (RIE-etch) is a valid combination."""
    spec = SimSpec(orientation="100", surface="1x1", species="Ar", ml=81,
                   flux_ratio=5, radical_energy=0.2)
    validate(spec)  # should not raise


def test_validate_rie_etch_negative_flux_ratio():
    spec = SimSpec(orientation="100", surface="1x1", species="O", ml=81, flux_ratio=-1)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_rie_etch_zero_radical_energy():
    spec = SimSpec(orientation="100", surface="1x1", species="O", ml=81,
                   flux_ratio=5, radical_energy=0.0)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_rie_etch_valid():
    spec = SimSpec(orientation="100", surface="1x1", species="O", ml=81,
                   flux_ratio=5, radical_energy=0.2)
    validate(spec)  # should not raise


# ─── Deposition mask ──────────────────────────────────────────────────────────

def _make_mask_spec(mask_type, mask_width=0.3, **kw):
    defaults = dict(orientation="100", surface="1x1", species="O", ml=81)
    defaults.update(kw)
    return SimSpec(mask_type=mask_type, mask_width=mask_width, **defaults)


def test_validate_no_mask_by_default():
    spec = SimSpec(orientation="100", surface="1x1", ml=81)
    assert spec.mask_type is None
    validate(spec)  # should not raise


@pytest.mark.parametrize("mask_type", ["xymask", "xmask", "ymask"])
def test_validate_mask_type_valid(mask_type):
    validate(_make_mask_spec(mask_type))  # should not raise


def test_validate_mask_type_invalid_string():
    spec = _make_mask_spec("zmask")
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_mask_rejected_for_cycle_etch():
    spec = SimSpec(
        orientation="100", surface="1x1", ml=81,
        mask_type="xymask", mask_width=0.3,
        phases=[
            CyclePhase(species="Ar", energy=20.0, fluence_ml=1),
            CyclePhase(species="O2", energy=5.0, fluence_ml=1),
        ],
    )
    with pytest.raises(SystemExit):
        validate(spec)


def test_mask_width_xymask_float_normalized_to_tuple():
    # __post_init__ should upgrade a bare float to (w, w) for xymask.
    spec = _make_mask_spec("xymask", mask_width=0.25)
    assert spec.mask_width == (0.25, 0.25)
    validate(spec)  # should not raise post-normalization


def test_mask_width_xymask_explicit_tuple_preserved():
    spec = _make_mask_spec("xymask", mask_width=(0.2, 0.4))
    assert spec.mask_width == (0.2, 0.4)
    validate(spec)  # should not raise


@pytest.mark.parametrize("mask_type", ["xmask", "ymask"])
def test_mask_width_single_mask_stays_float(mask_type):
    # __post_init__ only normalizes xymask; x/ymask keep a bare float.
    spec = _make_mask_spec(mask_type, mask_width=0.2)
    assert spec.mask_width == 0.2
    validate(spec)  # should not raise


@pytest.mark.parametrize("mask_type", ["xymask", "xmask", "ymask"])
@pytest.mark.parametrize("bad_width", [0.0, 0.5, 0.6, -0.1])
def test_validate_mask_width_out_of_range(mask_type, bad_width):
    spec = _make_mask_spec(mask_type, mask_width=bad_width)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_mask_width_xymask_one_side_out_of_range():
    # (x_frac, y_frac) — even if only one side is out of range, should fail.
    spec = _make_mask_spec("xymask", mask_width=(0.3, 0.6))
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_mask_width_xymask_wrong_length_tuple():
    spec = _make_mask_spec("xymask", mask_width=(0.1, 0.2, 0.3))
    with pytest.raises(SystemExit):
        validate(spec)


@pytest.mark.parametrize("mask_type", ["xmask", "ymask"])
def test_validate_mask_width_wrong_type_for_single_mask(mask_type):
    # xmask/ymask require a single float, not a tuple.
    spec = _make_mask_spec(mask_type, mask_width=(0.2, 0.3))
    with pytest.raises(SystemExit):
        validate(spec)


# ─── invert_mask validation ───────────────────────────────────────────────────

def test_validate_invert_mask_requires_mask_type():
    spec = SimSpec(orientation="100", surface="1x1", ml=81, invert_mask=True)
    with pytest.raises(SystemExit) as exc:
        validate(spec)
    assert "invert_mask" in str(exc.value)


def test_validate_invert_mask_with_mask_type_ok():
    spec = _make_mask_spec("xymask", invert_mask=True)
    validate(spec)  # should not raise


# ─── freeze_mask validation ───────────────────────────────────────────────────

def test_validate_freeze_mask_requires_mask_type():
    spec = SimSpec(orientation="100", surface="1x1", ml=81, freeze_mask=True)
    with pytest.raises(SystemExit) as exc:
        validate(spec)
    assert "freeze_mask" in str(exc.value)


def test_validate_freeze_mask_depth_zero_invalid():
    spec = _make_mask_spec("xmask", freeze_mask=True, freeze_mask_depth=0.0)
    with pytest.raises(SystemExit):
        validate(spec)


def test_validate_freeze_mask_with_mask_type_ok():
    spec = _make_mask_spec("xmask", freeze_mask=True)
    validate(spec)  # should not raise


def test_validate_freeze_mask_custom_depth_ok():
    spec = _make_mask_spec("ymask", freeze_mask=True, freeze_mask_depth=1.0)
    validate(spec)  # should not raise


def test_freeze_mask_default_values():
    spec = SimSpec(orientation="100", surface="1x1", ml=81)
    assert spec.freeze_mask is False
    assert spec.freeze_mask_depth == 2.0


def test_invert_mask_default_false():
    spec = SimSpec(orientation="100", surface="1x1", ml=81)
    assert spec.invert_mask is False
    assert not hasattr(spec, "mask_invert")


def test_from_dict_backward_compat_mask_invert():
    spec = SimSpec.from_dict({
        "species": "O", "energy": 50.0, "ml": 81,
        "mask_type": "xmask", "mask_width": 0.2, "mask_invert": True,
    })
    assert spec.invert_mask is True
    assert not hasattr(spec, "mask_invert")


def test_etch_mode_unaffected_by_mask():
    # mask is an orthogonal deposition modifier, not a mode; etch_mode() should
    # return the same string whether or not a mask is applied.
    unmasked = SimSpec(orientation="100", surface="1x1", ml=81, flux_ratio=5,
                       radical_energy=0.2)
    masked = _make_mask_spec("xymask", flux_ratio=5, radical_energy=0.2)
    assert etch_mode(unmasked) == etch_mode(masked) == "rie-etch"


def test_validate_mask_with_carbon_etch_still_requires_anchor_z_max():
    # Carbon-etch validation should still fire even with a mask set.
    spec = SimSpec(
        species="O", energy=0.5, ml=64,
        initial_config_file="/fake/path.data",  # anchor_z_max intentionally omitted
        mask_type="xymask", mask_width=0.3,
    )
    with pytest.raises(SystemExit):
        validate(spec)


def test_simspec_custom_fields():
    s = SimSpec(
        orientation="113",
        surface="O",
        surface_temperature=500.0,
        species="O",
        energy=2.0,
        ion_angle=15.0,
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


# ─── Multi-ion mode ───────────────────────────────────────────────────────────

def _make_multi_ion_spec(flux_ratio=5, **kw):
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


def test_etch_mode_multi_ion_etch():
    s = _make_multi_ion_spec(flux_ratio=0)
    assert etch_mode(s) == "multi-ion-etch"


def test_etch_mode_multi_rie_etch():
    s = _make_multi_ion_spec(flux_ratio=5)
    assert etch_mode(s) == "multi-rie-etch"


def test_multi_ion_validate_passes():
    validate(_make_multi_ion_spec())


def test_multi_ion_validate_bad_species():
    with pytest.raises(SystemExit):
        validate(_make_multi_ion_spec(ion_mix=[
            IonComponent("O", 0.5, 50.0),
            IonComponent("Xenon", 0.5, 50.0),
        ]))


def test_multi_ion_validate_fractions_not_sum_to_1():
    with pytest.raises(SystemExit):
        validate(_make_multi_ion_spec(ion_mix=[
            IonComponent("O",  0.3, 50.0),
            IonComponent("O2", 0.3, 100.0),
        ]))


def test_multi_ion_validate_only_one_component():
    with pytest.raises(SystemExit):
        validate(_make_multi_ion_spec(ion_mix=[
            IonComponent("O", 1.0, 50.0),
        ]))


def test_multi_ion_validate_negative_fraction():
    with pytest.raises(SystemExit):
        validate(_make_multi_ion_spec(ion_mix=[
            IonComponent("O",  -0.1, 50.0),
            IonComponent("O2",  1.1, 100.0),
        ]))


def test_multi_ion_validate_zero_radical_energy_with_flux():
    with pytest.raises(SystemExit):
        validate(_make_multi_ion_spec(flux_ratio=5, radical_energy=0.0))


def test_normalize_ion_mix():
    raw = [IonComponent("O", 1.0, 50.0), IonComponent("O2", 3.0, 100.0)]
    normed = normalize_ion_mix(raw)
    assert abs(normed[0].fraction - 0.25) < 1e-9
    assert abs(normed[1].fraction - 0.75) < 1e-9


def test_multi_ion_dir_name_rie():
    s = _make_multi_ion_spec(flux_ratio=5)
    assert multi_ion_dir_name(s) == "RIE_O_50p_50eV_O2_50p_100eV_R5"


def test_multi_ion_dir_name_ion_etch():
    s = _make_multi_ion_spec(flux_ratio=0)
    assert multi_ion_dir_name(s) == "ION_O_50p_50eV_O2_50p_100eV"


def test_multi_ion_dir_name_three_component():
    s = SimSpec(
        ml=81,
        ion_mix=[
            IonComponent("O",  0.1, 20.0),
            IonComponent("O",  0.2, 10.0),
            IonComponent("O",  0.7, 5.0),
        ],
        flux_ratio=0,
    )
    assert multi_ion_dir_name(s) == "ION_O_10p_20eV_O_20p_10eV_O_70p_5eV"


# ─── step_edge validation ─────────────────────────────────────────────────────

def test_step_edge_default_fields():
    spec = SimSpec(orientation="100", surface="1x1", ml=81)
    assert spec.step_edge is False
    assert spec.step_angle == 0.0
    assert spec.step_position == 0.5
    assert spec.step_invert is False
    assert spec.step_depth is None


def test_validate_step_edge_defaults_ok():
    spec = SimSpec(orientation="100", surface="1x1", ml=81, step_edge=True)
    validate(spec)  # should not raise


def test_validate_step_edge_position_zero_invalid():
    spec = SimSpec(orientation="100", surface="1x1", ml=81, step_edge=True, step_position=0.0)
    with pytest.raises(SystemExit) as exc:
        validate(spec)
    assert "step_position" in str(exc.value)


def test_validate_step_edge_position_one_invalid():
    spec = SimSpec(orientation="100", surface="1x1", ml=81, step_edge=True, step_position=1.0)
    with pytest.raises(SystemExit) as exc:
        validate(spec)
    assert "step_position" in str(exc.value)


def test_validate_step_edge_depth_zero_invalid():
    spec = SimSpec(orientation="100", surface="1x1", ml=81, step_edge=True, step_depth=0.0)
    with pytest.raises(SystemExit) as exc:
        validate(spec)
    assert "step_depth" in str(exc.value)


def test_validate_step_edge_depth_negative_invalid():
    spec = SimSpec(orientation="100", surface="1x1", ml=81, step_edge=True, step_depth=-1.5)
    with pytest.raises(SystemExit) as exc:
        validate(spec)
    assert "step_depth" in str(exc.value)


def test_validate_step_edge_custom_depth_ok():
    spec = SimSpec(orientation="111", surface="1x1", ml=81, step_edge=True,
                   step_angle=30.0, step_depth=2.2)
    validate(spec)  # should not raise
