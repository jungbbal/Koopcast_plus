"""
Self-contained data layer (OpenTraj-style).

    raw files ──[loaders]──► TrajDataset (canonical table) ──[windows]──► numpy

Example::

    from kpp.data import load, agent_windows, stack_windows
    ds = load("zara1")
    obs, pred = stack_windows(agent_windows(ds, obs_len=8, pred_len=12))
"""
from .trajdataset import TrajDataset, from_long_dataframe, CRITICAL_COLUMNS
from .loaders import (load, list_datasets, DATASETS, DatasetSpec,
                      load_obsmat, load_xyf_txt, load_taa_npy,
                      load_ethucy, ETHUCY_SCENES, ETHUCY_SPLITS)
from .loaders_opentraj import (load_gcs, load_town_center, load_edinburgh,
                               load_pets, load_wildtrack)
from .windows import (AgentWindow, agent_windows, stack_windows,
                      SceneWindow, scene_windows)

__all__ = [
    "TrajDataset", "from_long_dataframe", "CRITICAL_COLUMNS",
    "load", "list_datasets", "DATASETS", "DatasetSpec",
    "load_obsmat", "load_xyf_txt", "load_taa_npy",
    "load_ethucy", "ETHUCY_SCENES", "ETHUCY_SPLITS",
    "load_gcs", "load_town_center", "load_edinburgh", "load_pets", "load_wildtrack",
    "AgentWindow", "agent_windows", "stack_windows",
    "SceneWindow", "scene_windows",
]
