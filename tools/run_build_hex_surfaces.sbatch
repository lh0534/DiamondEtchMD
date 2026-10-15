#!/bin/bash
#SBATCH --job-name=hex_surfaces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --account=dgraves
#SBATCH --output=build_hex_surfaces_%A_%a.out

# Build the hexagonal-diamond surfaces for OVITO, one orientation per array task
# (make_surf.lmp on one CPU core, ~40 min for the ~1450-atom default boxes).
ORIENTS=(hex0001 hex1120 hex1010)
O=${ORIENTS[$SLURM_ARRAY_TASK_ID]}
cd /scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD/tools
export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export OMPI_MCA_btl='^openib'
# myenv's OpenMPI has no Slurm PMI support: clear the Slurm/PMI variables so MPI
# starts as a lone singleton (see tests/run_burst_tests.sbatch)
NOSLURM="env $(env | grep -oE '^(SLURM|PMI|PMIX|OMPI_MCA_ess|OMPI_MCA_orte)[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')"
cd /scratch/gpfs/DGRAVES/lh0534/jack_reax/DiamondEtchMD/tools
$NOSLURM /home/lh0534/.conda/envs/myenv/bin/python -u build_hex_surfaces.py /scratch/gpfs/DGRAVES/lh0534/jack_reax/dfiles/hex_diamond/surfaces $O
