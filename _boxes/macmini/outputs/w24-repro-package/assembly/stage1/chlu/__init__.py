"""
CHLU - Causal Hamiltonian Learning Unit

A JAX/Equinox-based implementation of a dynamical system grounded in
symplectic mechanics with a relativistic Hamiltonian.
"""
from importlib.metadata import PackageNotFoundError, version as _pkg_version
try:
    # Single source of truth: the version declared in pyproject.toml.
    __version__ = _pkg_version("chlu")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.2.4"
