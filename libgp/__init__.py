"""
libgp - Python bindings for Gaussian Process Regression Library
"""

from .libgp_cpp import CovFactory, GaussianProcess

__all__ = ['GaussianProcess', 'CovFactory']
