#!/usr/bin/env python3
"""Regenerate analysis plots and summary.txt for this simulation directory.

Run from (or with a path to) the simulation directory:

    python autoplot.py                      # fast refresh, no CNA
    python autoplot.py --cna-stride 10      # amorphous C every 10th impact
    python autoplot.py --no-cna             # explicitly skip CNA
    python autoplot.py --help
"""
import sys
from pathlib import Path

# Resolve sim_dir from the script location so this works whether called as
# "python autoplot.py" (cwd) or "python /path/to/sim/autoplot.py" (any cwd).
sim_dir = str(Path(sys.argv[0]).resolve().parent)
sys.argv.insert(1, sim_dir)

from diamond_etch_md.cli import plot_main
plot_main()
