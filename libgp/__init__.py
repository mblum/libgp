"""
libgp - Python bindings for Gaussian Process Regression Library
"""

from .libgp_cpp import CovFactory, GaussianProcess  # type: ignore

__all__ = ['GaussianProcess', 'CovFactory']
