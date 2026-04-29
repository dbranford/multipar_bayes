"""
Package for bounds in Bayesian quantum estimation.
See F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, [arXiv:2511.16645](https://arxiv.org/abs/2511.16645) for more details.
"""

from .convex import HolevoNagaokaBound, NagaokaHayashiBound
from .pgm import PGMBound
from .lower import RPMBound, SPMBound, SqPMBound


__all__ = [
    "bounds",
    "measurements",
    "HolevoNagaokaBound",
    "NagaokaHayashiBound",
    "RPMBound",
    "SPMBound",
    "SqPMBound",
    "PGMBound",
]


__docformat__ = "numpy"
