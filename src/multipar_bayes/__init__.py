"""
Package for bounds in Bayesian quantum estimation.
See F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, arXiv:2511.XXXXX for more details.
"""

from .convex import nh_fun, hn_fun
from .pgm import pgm_fun
from .lower import spm_fun, rpm_fun, sqpm_fun


__all__ = [nh_fun, hn_fun, spm_fun, pgm_fun, rpm_fun, sqpm_fun, "measurements"]


__docformat__ = "numpy"
