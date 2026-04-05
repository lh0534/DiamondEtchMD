"""
lammps/submit.py — generator for the SLURM submit script.

The generated script:
  - loads the LAMMPS/Kokkos/GPU module for the Della cluster
  - runs make_surf.lmp to build the initial surface (skipped if data_files/0.data exists)
  - detects the resume state from ncarbon.txt and the latest restart snapshot
  - launches lmp with all required -var arguments
  - uses --dependency=singleton for automatic serialization of re-queued jobs
  - after LAMMPS exits, checks whether the target fluence has been reached and
    re-queues the same script if not done (auto re-submit)
"""

from ..spec import SimSpec


def get_submit_script(spec: SimSpec) -> str:
    """Generate the contents of the SLURM submit script for the given SimSpec."""
    mail_lines = (
        f"#SBATCH --mail-type=END,FAIL\n"
        f"#SBATCH --mail-user={spec.email}\n"
    ) if spec.email else ""

    return (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={spec.name}\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --mem=16G\n"
        f"#SBATCH --ntasks=1\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --gres=gpu:1\n"
        f"#SBATCH --time={spec.wall_hours}:00:00\n"
        f"#SBATCH --dependency=singleton\n"
        f"#SBATCH --nice=2\n"
        f"#SBATCH --account={spec.account}\n"
        f"{mail_lines}"
        f"\n"
        f"module purge\n"
        f"module load lammps/kokkos/gpu_della9_2022\n"
        f"\n"
        f"mkdir -p dumps data_files\n"
        f"\n"
        f"export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        f"export OMP_PROC_BIND=spread\n"
        f"export OMP_PLACES=threads\n"
        f"\n"
        f"if [ ! -f data_files/0.data ]; then\n"
        f"    srun lmp -log log_make_surf.lammps -k on g 1 -sf kk -in make_surf.lmp\n"
        f"fi\n"
        f"\n"
        f"n_lat_0=$(grep ' atoms' data_files/0.data | awk '{{print $1}}')\n"
        f"n_complete=$(tail -1 ncarbon.txt 2>/dev/null | awk '{{print $1}}')\n"
        f"n_complete=${{n_complete:-0}}\n"
        f"data_file=$(ls -t data_files/*.data | head -1)\n"
        f"log_file=log$(echo \"$(ls | grep -c .lammps)+1\" | bc).lammps\n"
        f"event_count=$(ls dumps/event_dump_*.dump 2>/dev/null | wc -l)\n"
        f"\n"
        f"# Check if simulation is already complete\n"
        f"ML=$(grep 'ML equal' config.lmp | awk '{{print $4}}')\n"
        f"end_fluence=$(grep 'end_fluence equal' config.lmp | awk '{{print $4}}')\n"
        f"end_c=$(( end_fluence * ML ))\n"
        f"if [ \"$n_complete\" -ge \"$end_c\" ]; then\n"
        f"    echo \"Simulation already complete: $n_complete / $end_c impacts done.\"\n"
        f"    exit 0\n"
        f"fi\n"
        f"\n"
        f"srun lmp -k on g 1 -sf kk \\\n"
        f"    -var data_file $data_file \\\n"
        f"    -var n_complete $n_complete \\\n"
        f"    -var log_file $log_file \\\n"
        f"    -var n_events $event_count \\\n"
        f"    -var n_lat_0 $n_lat_0 \\\n"
        f"    -log $log_file \\\n"
        f"    -screen none \\\n"
        f"    -nocite \\\n"
        f"    -in head.lmp\n"
        f"lmp_exit=$?\n"
        f"\n"
        f"# ── Auto re-submit ──────────────────────────────────────────────────\n"
        f"# Re-queue this script if LAMMPS exited cleanly and the target fluence\n"
        f"# has not been reached.  Uses the same submit script ($0) so the next\n"
        f"# job picks up from the latest ncarbon.txt / data_files snapshot.\n"
        f"if [ $lmp_exit -ne 0 ]; then\n"
        f"    echo \"LAMMPS exited with code $lmp_exit — not re-submitting.\"\n"
        f"    exit $lmp_exit\n"
        f"fi\n"
        f"\n"
        f"n_complete=$(tail -1 ncarbon.txt 2>/dev/null | awk '{{print $1}}')\n"
        f"n_complete=${{n_complete:-0}}\n"
        f"if [ \"$n_complete\" -lt \"$end_c\" ]; then\n"
        f"    echo \"Progress: $n_complete / $end_c impacts. Re-submitting...\"\n"
        f"    sbatch \"$0\"\n"
        f"else\n"
        f"    echo \"Simulation complete: $n_complete / $end_c impacts.\"\n"
        f"fi\n"
    )
