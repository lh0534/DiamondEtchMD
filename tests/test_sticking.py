"""Tests for diamond_etch_md.analysis.sticking."""

import json
import pytest
from pathlib import Path

from diamond_etch_md.analysis.sticking import check_radical_reflecting


def _make_sim_dir(tmp_path, ml, nc_rows, ep_rows, spec_ml=None):
    """Write minimal ncarbon.txt + etch_products.txt for testing."""
    nc_lines = []
    for impact, cn, nc_, nh, no in nc_rows:
        nc_lines.append(f"{impact} {cn} {nc_} {nh} {no}")
    (tmp_path / "ncarbon.txt").write_text("\n".join(nc_lines) + "\n")

    ep_lines = ["# impact cn n_C n_H n_O n_Ar"]
    for impact, cn, nc_, nh, no, nar in ep_rows:
        ep_lines.append(f"{impact} {cn} {nc_} {nh} {no} {nar}")
    (tmp_path / "etch_products.txt").write_text("\n".join(ep_lines) + "\n")

    if spec_ml is not None:
        (tmp_path / "spec.json").write_text(json.dumps({"ml": spec_ml}))

    return tmp_path


class TestCheckRadicalReflecting:

    def test_no_bad_bins_no_warning(self, tmp_path):
        """All radicals stick → no RADS_REFLECTING written."""
        ml = 5
        # 10 impacts, 2 radicals each: cn=1,2 for impacts 1-10
        nc_rows = []
        for impact in range(1, 11):
            nc_rows.append((impact, 1, 100, 0, 0))
            nc_rows.append((impact, 2, 100, 0, 0))
            nc_rows.append((impact, 0, 100, 0, 0))  # ion impact

        ep_rows = []
        # Only ion-ejected C clusters (cn=0): no reflections
        for impact in range(1, 11):
            ep_rows.append((impact, 0, 5, 0, 0, 0))

        _make_sim_dir(tmp_path, ml, nc_rows, ep_rows)

        result = check_radical_reflecting(tmp_path, ml=ml, threshold=0.95)
        assert not result["bad_bins"]
        assert not result["wrote_warning"]
        assert not (tmp_path / "RADS_REFLECTING").exists()

    def test_bad_bin_writes_warning(self, tmp_path):
        """High reflection rate → RADS_REFLECTING written."""
        ml = 5
        # 10 impacts with 2 radicals each; impacts 1-5 all reflect
        nc_rows = []
        for impact in range(1, 11):
            nc_rows.append((impact, 1, 100, 0, 5))
            nc_rows.append((impact, 2, 100, 0, 10))
            nc_rows.append((impact, 0, 100, 0, 0))

        ep_rows = []
        # Impacts 1-5: both radicals reflect (nO=1, nC=nH=nAr=0)
        for impact in range(1, 6):
            ep_rows.append((impact, 1, 0, 0, 1, 0))
            ep_rows.append((impact, 2, 0, 0, 1, 0))
        # Impacts 6-10: radicals stick (produce C clusters)
        for impact in range(6, 11):
            ep_rows.append((impact, 0, 3, 0, 0, 0))

        _make_sim_dir(tmp_path, ml, nc_rows, ep_rows)

        result = check_radical_reflecting(tmp_path, ml=ml, threshold=0.95)
        assert len(result["bad_bins"]) == 1
        assert result["bad_bins"][0]["ml_number"] == 1
        assert result["wrote_warning"]
        warning = (tmp_path / "RADS_REFLECTING").read_text()
        assert "100.0% reflecting" in warning
        assert "ML 1" in warning

    def test_warning_removed_when_cleared(self, tmp_path):
        """Pre-existing RADS_REFLECTING is removed when all bins are good."""
        ml = 5
        (tmp_path / "RADS_REFLECTING").write_text("stale warning\n")

        nc_rows = [(i, 1, 100, 0, 0) for i in range(1, 11)]
        nc_rows += [(i, 0, 100, 0, 0) for i in range(1, 11)]
        ep_rows = []  # no reflections

        _make_sim_dir(tmp_path, ml, nc_rows, ep_rows)

        result = check_radical_reflecting(tmp_path, ml=ml, threshold=0.95)
        assert not result["bad_bins"]
        assert not (tmp_path / "RADS_REFLECTING").exists()

    def test_ml_loaded_from_spec_json(self, tmp_path):
        """ml=0 should be loaded from spec.json."""
        ml = 5
        nc_rows = [(i, 1, 100, 0, 0) for i in range(1, 11)]
        nc_rows += [(i, 0, 100, 0, 0) for i in range(1, 11)]
        ep_rows = []

        _make_sim_dir(tmp_path, ml, nc_rows, ep_rows, spec_ml=ml)

        result = check_radical_reflecting(tmp_path, ml=0, threshold=0.95)
        assert "ml_stats" in result

    def test_missing_ncarbon_returns_empty(self, tmp_path):
        """Missing ncarbon.txt → empty result, no exception."""
        result = check_radical_reflecting(tmp_path, ml=5)
        assert result["ml_stats"] == []
        assert not result["wrote_warning"]

    def test_partial_bin_excluded(self, tmp_path):
        """The partial (in-progress) ML bin is not counted in stats."""
        ml = 5
        # Only 7 impacts → bin 0 complete (1-5), bin 1 partial (6-7)
        nc_rows = [(i, 1, 100, 0, 0) for i in range(1, 8)]
        nc_rows += [(i, 0, 100, 0, 0) for i in range(1, 8)]
        ep_rows = [(i, 1, 0, 0, 1, 0) for i in range(6, 8)]  # reflect in partial bin

        _make_sim_dir(tmp_path, ml, nc_rows, ep_rows)

        result = check_radical_reflecting(tmp_path, ml=ml, threshold=0.95)
        # Only bin 0 should appear; bin 1 (impacts 6-7) is partial
        assert len(result["ml_stats"]) == 1
        assert result["ml_stats"][0]["bin_idx"] == 0
