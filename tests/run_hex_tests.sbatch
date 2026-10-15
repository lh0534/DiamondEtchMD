#!/bin/bash
#SBATCH --job-name=hex_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --account=dgraves
#SBATCH --output=hex_tests_%j.out

# Hexagonal-diamond LAMMPS tests on one CPU core (myenv: LAMMPS 23 Jun 2022 with REAXFF):
# lattice + addfix replenishment (pair_style zero, ~1 min), then make_surf + four ion
# impacts through head.lmp: each orientation at ~10 Å with Ar+ and at 5-6 planes with O+.
cd /scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD
export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export OMPI_MCA_btl='^openib'
PY=/home/lh0534/.conda/envs/myenv/bin/python
TMP=/scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD/.hex_test_tmp
mkdir -p $TMP
# myenv's OpenMPI has no Slurm PMI support: clear the Slurm/PMI variables so MPI
# starts as a lone singleton (see run_burst_tests.sbatch)
NOSLURM="env $(env | grep -oE '^(SLURM|PMI|PMIX|OMPI_MCA_ess|OMPI_MCA_orte)[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')"
$NOSLURM $PY -m pytest tests/test_hex_lammps.py -m lammps -v --durations=0 --basetemp=$TMP/e2e
