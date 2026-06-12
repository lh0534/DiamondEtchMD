#!/usr/bin/env python3
"""Regenerate analysis plots and summary.txt for this simulation directory.

Run from (or with a path to) the simulation directory:

    python autoplot.py                      # prompts: plot CNA? run new CNA?
    python autoplot.py --cna                # plot from existing cache, no compute
    python autoplot.py --cna-run            # update cache with new impacts, then plot
    python autoplot.py --cna-stride 10      # compute from scratch every 10th impact
    python autoplot.py --no-cna             # skip CNA entirely
    python autoplot.py --help
"""
import sys
from pathlib import Path

# Use .absolute() (not .resolve()) so that symlinks are not followed — the
# symlink lives in the sim directory, and its parent is the sim directory.
sim_dir = str(Path(sys.argv[0]).absolute().parent)
sys.argv.insert(1, sim_dir)

# Ensure diamond_etch_md is importable even when compute-node python3 differs
# from the one used to pip-install the package.  __file__ follows the symlink
# (via realpath) to the actual template file inside the source tree, so the
# package root (DiamondEtchMD/) is always 4 directories up.
try:
    import diamond_etch_md  # noqa: F401 — already on sys.path, nothing to do
except ModuleNotFoundError:
    _src_root = Path(__file__).resolve().parents[3]  # …/DiamondEtchMD/
    sys.path.insert(0, str(_src_root))

# Ask about CNA unless the user already passed a CNA flag.
_cna_flags = {'--no-cna', '--cna', '--cna-run', '--cna-stride', '--help', '-h'}
if not any(f in sys.argv for f in _cna_flags):
    try:
        ans1 = input("Plot CNA data? [y/N] ").strip().lower()
    except EOFError:
        ans1 = ''
    if ans1 not in ('y', 'yes'):
        sys.argv.append('--no-cna')
    else:
        try:
            ans2 = input("Run new CNA computation (slow)? [y/N] ").strip().lower()
        except EOFError:
            ans2 = ''
        if ans2 in ('y', 'yes'):
            try:
                stride_s = input("CNA stride (0 = 1 per ML, or enter N to analyze every N-th impact): ").strip()
                stride = int(stride_s) if stride_s else 0
            except (EOFError, ValueError):
                stride = 0
            if stride > 0:
                sys.argv += ['--cna-stride', str(stride)]
            else:
                sys.argv.append('--cna-run')
        else:
            sys.argv.append('--cna')

from diamond_etch_md.cli import plot_main
plot_main()
