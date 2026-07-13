"""Isolated NumPy-2 materials kinetics runtime.

This package deliberately lives outside :mod:`ultra_deepagents.materials` so
importing the shared NumPy-1.26 sandbox cannot import Kawin accidentally.
"""

from .runner import execute_request, runtime_support

__all__ = ["execute_request", "runtime_support"]
