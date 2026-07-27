"""
Predictors -- one common ``Predictor`` interface, in-process (no subprocess).

Shipped:
  * ConstantVelocity -- trivial baseline / pipeline sanity check.
  * KoopCast++        -- neighbour-aware Koopman predictor (ours, retrained).

To add your own: subclass ``Predictor`` and implement ``predict(obs) -> pred``.
Nothing else in the eval/control loops needs to change.
"""
from .base import Predictor
from .constant_velocity import ConstantVelocity
from .koopcastpp import KoopCastPP

__all__ = ["Predictor", "ConstantVelocity", "KoopCastPP"]
