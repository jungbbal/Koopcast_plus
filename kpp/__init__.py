"""
kpp -- the self-contained core of koopcast_plusplus.

Depends on nothing outside this repository (no CANVAS checkout, no sibling
baseline repos). Data layer follows OpenTraj's canonical ``TrajDataset`` schema:
heterogeneous raw files are read by per-dataset loaders and converge to ONE
in-memory table, after which windowing / prediction / evaluation are all
dataset-agnostic.

    raw files ──[loaders]──► TrajDataset ──[windows]──► numpy ──[predict]──► ADE/FDE
"""
__version__ = "0.0.1"
