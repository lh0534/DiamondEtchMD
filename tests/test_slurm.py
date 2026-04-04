"""
tests/test_slurm.py — submit a real SLURM job and verify it was accepted.

Run with:
    pytest -m slurm -v

Skipped automatically in normal `pytest` runs.
"""

import sys
import re
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from diamond_etch_md import SimSpec, compute_ml, make_sim
from diamond_etch_md.spec import validate

pytestmark = pytest.mark.slurm


@pytest.fixture
def sim_dir(tmp_path):
    """Build a real simulation directory using bundled templates."""
    spec = SimSpec(
        orientation="100",
        reconstruction="bare_1x1",
        species="O",
        energy=0.5,
        temperature=300.0,
        ml=compute_ml("100", 9, 9),
        box_x=9, box_y=9, box_depth=3,
        fluence=1,
        wall_hours=1,
        name="demd-test",
    )
    validate(spec)
    outdir = tmp_path / "demd-test"
    make_sim(spec, outdir)
    return outdir


def test_sbatch_accepts_job(sim_dir):
    result = subprocess.run(
        ["sbatch", "--test-only", str(sim_dir / "submit")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"sbatch --test-only failed:\n{result.stderr}"
    )


def test_sbatch_submits_job(sim_dir):
    result = subprocess.run(
        ["sbatch", str(sim_dir / "submit")],
        capture_output=True, text=True,
        cwd=sim_dir,
    )
    assert result.returncode == 0, (
        f"sbatch failed:\n{result.stderr}"
    )
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    assert match, f"Unexpected sbatch output: {result.stdout!r}"
    job_id = match.group(1)
    print(f"\nSubmitted job {job_id}")

    # Cancel immediately — this is just a smoke test
    subprocess.run(["scancel", job_id], check=True)
