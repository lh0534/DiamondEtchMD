#!/bin/bash
#SBATCH --job-name=cleanup_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --account=dgraves
#SBATCH --output=cleanup_tests_%j.out

# Post-ion product cleanup LAMMPS tests on one CPU core (myenv: LAMMPS 23 Jun 2022 with
# REAXFF): product_cleanup.lmp and the slab-only thermostat (pair_style zero, seconds),
# then an end-to-end CO2 attribution run and a short RIE run through head.lmp.
cd /scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD
export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export OMPI_MCA_btl='^openib'
PY=/home/lh0534/.conda/envs/myenv/bin/python
TMP=/scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD/.cleanup_test_tmp
mkdir -p $TMP
# myenv's OpenMPI has no Slurm PMI support: clear the Slurm/PMI variables so MPI
# starts as a lone singleton (see run_burst_tests.sbatch)
NOSLURM="env $(env | grep -oE '^(SLURM|PMI|PMIX|OMPI_MCA_ess|OMPI_MCA_orte)[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')"
$NOSLURM $PY -m pytest tests/test_product_cleanup_lammps.py -m lammps -v --durations=0 --basetemp=$TMP/e2e
