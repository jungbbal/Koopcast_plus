"""
kpp.baselines -- external trajectory-prediction baselines, vendored.

Third-party predictors (Social-STGCNN, SocialVAE, EigenTrajectory, Trajectron++,
CANVAS Linear/GP) live under ``vendor/`` byte-for-byte unmodified. Thin adapters
in ``adapters.py`` wrap them in the kpp ``Predictor`` interface so they drop
straight into ``kpp.eval.evaluate_scene`` alongside ConstantVelocity / KoopCast++.

Usage::

    from kpp.baselines import make_baseline
    from kpp.data import load_ethucy
    from kpp.eval import evaluate_scene

    model = make_baseline("stgcnn", "zara1")
    print(evaluate_scene(model, load_ethucy("zara1", "test")))

See ``README.md`` for provenance and per-baseline status.
"""
from .adapters import (
    BASELINES,
    WORKING,
    make_baseline,
    LinearBaseline,
    STGCNNBaseline,
    SocialVAEBaseline,
    EigenBaseline,
    GPBaseline,
    TrajectronBaseline,
)

__all__ = [
    "BASELINES", "WORKING", "make_baseline",
    "LinearBaseline", "STGCNNBaseline", "SocialVAEBaseline",
    "EigenBaseline", "GPBaseline", "TrajectronBaseline",
]
