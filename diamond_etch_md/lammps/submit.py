"""
lammps/submit.py — generator for the SLURM submit script.

The generated script:
  - loads the LAMMPS/Kokkos/GPU module for the Della cluster
  - runs make_surf.lmp to build the initial surface (skipped if impact_snaps/0.data exists)
  - detects the resume state from ncarbon.txt and the latest restart snapshot
  - launches lmp with all required -var arguments
  - uses --dependency=singleton for automatic serialization of re-queued jobs
  - after LAMMPS exits, checks whether the target fluence has been reached and
    re-queues the same script if not done (auto re-submit)
"""

from ..spec import SimSpec, CyclePhase


def _plot_loop_block(interval_hours: int) -> str:
    """Return the bash snippet that starts/stops the background auto-plot loop."""
    if interval_hours <= 0:
        return (
            "_PLOT_PID=\"\"\n"
        )
    interval_s = interval_hours * 3600
    return (
        f"# Auto-plot every {interval_hours} h (--no-cna for speed during live run)\n"
        f"_PLOT_PID=\"\"\n"
        f"( while true; do\n"
        f"      sleep {interval_s}\n"
        f"      python3 auto-plot.py --no-cna 2>>plot.log || true\n"
        f"  done ) &\n"
        f"_PLOT_PID=$!\n"
    )


def _plot_kill_block() -> str:
    return "[ -n \"$_PLOT_PID\" ] && kill \"$_PLOT_PID\" 2>/dev/null; wait \"$_PLOT_PID\" 2>/dev/null || true\n"


def _final_plot_block(ml_var: str = "$ML") -> str:
    """Run a final plot with CNA strided to 1-per-ML after normal completion."""
    return (
        f"echo \"Running final analysis plots ...\"\n"
        f"python3 auto-plot.py --cna-stride {ml_var} 2>>plot.log || true\n"
    )


def get_submit_script(spec: SimSpec) -> str:
    """Generate the contents of the SLURM submit script for the given SimSpec.

    Handles both theory-etch (flux_ratio == 0) and RIE-etch (flux_ratio > 0).
    For RIE-etch, reads cn_start (col 2 of last ncarbon.txt line) and passes
    -var neut_complete $cn_start to LAMMPS for mid-radical-loop restarts.
    """
    mail_lines = (
        f"#SBATCH --mail-type=END,FAIL\n"
        f"#SBATCH --mail-user={spec.email}\n"
    ) if spec.email else ""

    plot_loop = _plot_loop_block(spec.plot_interval_hours)
    plot_kill = _plot_kill_block()
    final_plot = _final_plot_block()

    is_rie = spec.flux_ratio > 0

    # RIE-etch: read cn_start from col 2 of last ncarbon.txt line
    if is_rie:
        cn_start_lines = (
            f"cn_start=$(tail -1 ncarbon.txt 2>/dev/null | awk 'NF>=5{{print $2}} NF<5{{print 0}}')\n"
            f"cn_start=${{cn_start:-0}}\n"
        )
        neut_complete_var = f"    -var neut_complete $cn_start \\\n"
    else:
        cn_start_lines = ""
        neut_complete_var = ""

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
        f"#SBATCH --signal=B:USR1@120\n"
        f"#SBATCH --nice=2\n"
        f"#SBATCH --account={spec.account}\n"
        f"{mail_lines}"
        f"\n"
        f"module purge\n"
        f"module load {spec.lammps_module}\n"
        f"\n"
        f"mkdir -p etch_event_trajs impact_snaps\n"
        f"\n"
        f"export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        f"export OMP_PROC_BIND=spread\n"
        f"export OMP_PLACES=threads\n"
        f"\n"
        f"# Resubmit on SLURM time-limit signal (fired 120 s before wall-time).\n"
        f"# srun runs in the background so 'wait' is interruptible by the signal.\n"
        f"_resubmitted=0\n"
        f"_resubmit() {{\n"
        f"    local nc ml ef ec\n"
        f"    nc=$(tail -1 ncarbon.txt 2>/dev/null | awk '{{print $1}}'); nc=${{nc:-0}}\n"
        f"    ml=$(grep 'ML equal' config.lmp | awk '{{print $4}}')\n"
        f"    ef=$(grep 'end_fluence equal' config.lmp | awk '{{print $4}}')\n"
        f"    ec=$(( ef * ml ))\n"
        f"    if [ \"$nc\" -lt \"$ec\" ]; then\n"
        f"        echo \"Wall-time signal: $nc / $ec impacts done — re-submitting.\"\n"
        f"        {plot_kill}"
        f"        sbatch \"$0\"\n"
        f"        _resubmitted=1\n"
        f"    fi\n"
        f"}}\n"
        f"trap '_resubmit' USR1\n"
        f"\n"
        f"if [ ! -f impact_snaps/0.data ]; then\n"
        f"    srun lmp -log log_make_surf.lammps -k on g 1 -sf kk -in make_surf.lmp\n"
        f"fi\n"
        f"\n"
        f"n_lat_0=$(grep ' atoms' impact_snaps/0.data | awk '{{print $1}}')\n"
        f"n_complete=$(tail -1 ncarbon.txt 2>/dev/null | awk '{{print $1}}')\n"
        f"n_complete=${{n_complete:-0}}\n"
        f"{cn_start_lines}"
        f"data_file=$(ls -t impact_snaps/*.data | head -1)\n"
        f"log_file=log$(echo \"$(ls | grep -c .lammps)+1\" | bc).lammps\n"
        f"event_count=$(ls etch_event_trajs/event_dump_*.dump 2>/dev/null | wc -l)\n"
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
        f"# Run LAMMPS in background so the USR1 trap can fire during 'wait'\n"
        f"srun lmp -k on g 1 -sf kk \\\n"
        f"    -var data_file $data_file \\\n"
        f"    -var n_complete $n_complete \\\n"
        f"{neut_complete_var}"
        f"    -var log_file $log_file \\\n"
        f"    -var n_events $event_count \\\n"
        f"    -var n_lat_0 $n_lat_0 \\\n"
        f"    -log $log_file \\\n"
        f"    -screen none \\\n"
        f"    -nocite \\\n"
        f"    -in head.lmp &\n"
        f"SRUN_PID=$!\n"
        f"{plot_loop}"
        f"wait $SRUN_PID\n"
        f"lmp_exit=$?\n"
        f"{plot_kill}"
        f"\n"
        f"# If the time-limit signal fired, resubmit was already handled; exit cleanly\n"
        f"[ $_resubmitted -eq 1 ] && exit 0\n"
        f"\n"
        f"# Normal exit: resubmit if LAMMPS finished cleanly but fluence not reached\n"
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
        f"    {final_plot}"
        f"fi\n"
    )


def get_submit_script_cycle_etch(spec: SimSpec) -> str:
    """Generate the SLURM submit script for a cycle-etch SimSpec.

    Differences from the single-species version:
      - Reads cn_start (col 2 of last ncarbon.txt line) for mid-radical-loop restarts.
      - Passes -var neut_complete $cn_start to LAMMPS.
      - ncarbon.txt col 1 still tracks total completed ion impacts.
    """
    mail_lines = (
        f"#SBATCH --mail-type=END,FAIL\n"
        f"#SBATCH --mail-user={spec.email}\n"
    ) if spec.email else ""

    plot_loop = _plot_loop_block(spec.plot_interval_hours)
    plot_kill = _plot_kill_block()
    final_plot = _final_plot_block()

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
        f"#SBATCH --signal=B:USR1@120\n"
        f"#SBATCH --nice=2\n"
        f"#SBATCH --account={spec.account}\n"
        f"{mail_lines}"
        f"\n"
        f"module purge\n"
        f"module load {spec.lammps_module}\n"
        f"\n"
        f"mkdir -p etch_event_trajs impact_snaps\n"
        f"\n"
        f"export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        f"export OMP_PROC_BIND=spread\n"
        f"export OMP_PLACES=threads\n"
        f"\n"
        f"# Resubmit on SLURM time-limit signal (fired 120 s before wall-time).\n"
        f"_resubmitted=0\n"
        f"_resubmit() {{\n"
        f"    local nc ml ef ec\n"
        f"    nc=$(tail -1 ncarbon.txt 2>/dev/null | awk '{{print $1}}'); nc=${{nc:-0}}\n"
        f"    ml=$(grep 'ML equal' config.lmp | awk '{{print $4}}')\n"
        f"    ef=$(grep 'end_fluence equal' config.lmp | awk '{{print $4}}')\n"
        f"    ec=$(( ef * ml ))\n"
        f"    if [ \"$nc\" -lt \"$ec\" ]; then\n"
        f"        echo \"Wall-time signal: $nc / $ec impacts done — re-submitting.\"\n"
        f"        {plot_kill}"
        f"        sbatch \"$0\"\n"
        f"        _resubmitted=1\n"
        f"    fi\n"
        f"}}\n"
        f"trap '_resubmit' USR1\n"
        f"\n"
        f"if [ ! -f impact_snaps/0.data ]; then\n"
        f"    srun lmp -log log_make_surf.lammps -k on g 1 -sf kk -in make_surf.lmp\n"
        f"fi\n"
        f"\n"
        f"n_lat_0=$(grep ' atoms' impact_snaps/0.data | awk '{{print $1}}')\n"
        f"n_complete=$(tail -1 ncarbon.txt 2>/dev/null | awk '{{print $1}}')\n"
        f"n_complete=${{n_complete:-0}}\n"
        f"cn_start=$(tail -1 ncarbon.txt 2>/dev/null | awk '{{print $2}}')\n"
        f"cn_start=${{cn_start:-0}}\n"
        f"data_file=$(ls -t impact_snaps/*.data | head -1)\n"
        f"log_file=log$(echo \"$(ls | grep -c .lammps)+1\" | bc).lammps\n"
        f"event_count=$(ls etch_event_trajs/event_dump_*.dump 2>/dev/null | wc -l)\n"
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
        f"# Run LAMMPS in background so the USR1 trap can fire during 'wait'\n"
        f"srun lmp -k on g 1 -sf kk \\\n"
        f"    -var data_file $data_file \\\n"
        f"    -var n_complete $n_complete \\\n"
        f"    -var neut_complete $cn_start \\\n"
        f"    -var log_file $log_file \\\n"
        f"    -var n_events $event_count \\\n"
        f"    -var n_lat_0 $n_lat_0 \\\n"
        f"    -log $log_file \\\n"
        f"    -screen none \\\n"
        f"    -nocite \\\n"
        f"    -in head.lmp &\n"
        f"SRUN_PID=$!\n"
        f"{plot_loop}"
        f"wait $SRUN_PID\n"
        f"lmp_exit=$?\n"
        f"{plot_kill}"
        f"\n"
        f"[ $_resubmitted -eq 1 ] && exit 0\n"
        f"\n"
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
        f"    {final_plot}"
        f"fi\n"
    )
