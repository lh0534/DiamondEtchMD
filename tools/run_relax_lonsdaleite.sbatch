#!/bin/bash
#SBATCH --job-name=relax_hex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --account=dgraves
#SBATCH --output=relax_lonsdaleite_%j.out

# ReaxFF zero-pressure lattice parameters of lonsdaleite and cubic diamond (one CPU core).
cd /scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD/tools
export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export OMPI_MCA_btl='^openib'
# myenv's OpenMPI has no Slurm PMI support: clear the Slurm/PMI variables so MPI
# starts as a lone singleton (see tests/run_burst_tests.sbatch)
NOSLURM="env $(env | grep -oE '^(SLURM|PMI|PMIX|OMPI_MCA_ess|OMPI_MCA_orte)[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')"
$NOSLURM /home/lh0534/.conda/envs/myenv/bin/python relax_lonsdaleite.py

