#!/bin/bash
#SBATCH --job-name=burst_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --account=dgraves
#SBATCH --output=burst_tests_%j.out

# Real-LAMMPS burst tests on one CPU core (myenv: LAMMPS 23 Jun 2022 with REAXFF).
cd /scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD
export OMP_NUM_THREADS=1
PY=/home/lh0534/.conda/envs/myenv/bin/python
TMP=/scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD/.burst_test_tmp
mkdir -p $TMP
export PYTHONFAULTHANDLER=1
export OMPI_MCA_btl='^openib'
# myenv's OpenMPI has no Slurm PMI support: under srun it refuses to start, and a
# plain launch inside the job sees SLURM_* and aborts in MPI_Init.  Clear the
# Slurm/PMI variables for the Python process so MPI starts as a lone singleton
# (as on the login/vis nodes).
NOSLURM="env $(env | grep -oE '^(SLURM|PMI|PMIX|OMPI_MCA_ess|OMPI_MCA_orte)[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')"
echo "== LAMMPS smoke test"
$NOSLURM $PY -c "from lammps import lammps; L = lammps(cmdargs=['-log', 'none', '-screen', 'none']); print('lammps ok', L.version(), 'REAXFF' in L.installed_packages)" || exit 1
$NOSLURM $PY -m pytest tests/test_burst_sampling.py -m lammps -v -k "not e2e" --basetemp=$TMP/fast
$NOSLURM $PY -m pytest tests/test_burst_sampling.py -m lammps -v -k e2e --durations=0 --basetemp=$TMP/e2e
