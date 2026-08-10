"""Standalone analysis and maintenance CLIs.

Each module here is a script meant to be run directly (see its own
docstring for `python tools/<name>.py ...` usage), not a library imported
by backtest/ or research/. They are grouped in this package only so their
own cross-imports (e.g. objective_sweep.py importing ensemble_model.py and
refit_stability.py) resolve the same way whether run directly or imported
from tests.
"""
