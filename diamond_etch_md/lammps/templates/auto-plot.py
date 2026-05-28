#!/usr/bin/env python3
"""Regenerate analysis plots and summary.txt for this simulation directory.

Run from (or with a path to) the simulation directory:

    python autoplot.py                      # prompts for CNA, then plots
    python autoplot.py --cna-stride 10      # amorphous C every 10th impact
    python autoplot.py --no-cna             # skip CNA without prompting
    python autoplot.py --help
"""
import sys
from pathlib import Path

# Use .absolute() (not .resolve()) so that symlinks are not followed — the
# symlink lives in the sim directory, and its parent is the sim directory.
sim_dir = str(Path(sys.argv[0]).absolute().parent)
sys.argv.insert(1, sim_dir)

# Ask about CNA unless the user already passed --no-cna or --cna-stride.
_cna_flags = {'--no-cna', '--cna-stride', '--help', '-h'}
if not any(f in sys.argv for f in _cna_flags):
    try:
        ans = input("Run CNA analysis? (requires jaxmd env; slow for large runs) [y/N] ").strip().lower()
    except EOFError:
        ans = ''
    if ans not in ('y', 'yes'):
        sys.argv.append('--no-cna')

from diamond_etch_md.cli import plot_main
plot_main()
