"""Bayesian switching-Koopman experiments (see ../README.md)."""
import pathlib as _pathlib
import sys as _sys

# make the shared zara2_lab data layer (zlab) importable from anywhere in here
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from .koopman import (  # noqa: E402,F401
    TOKENS,
    err,
    fit_lstsq,
    fit_switching,
    psi,
    tokenize,
    transition_matrix,
)

OUT = _pathlib.Path(__file__).resolve().parents[1] / "out"
