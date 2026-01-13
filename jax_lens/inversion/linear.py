"""
Linear solvers for pixelized source reconstruction.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


def solve_linear_system(
    A: jnp.ndarray,
    b: jnp.ndarray,
    solver: str = "dense",
    solver_kwargs: Optional[dict] = None,
) -> jnp.ndarray:
    """
    Solve A x = b using the chosen solver.
    """
    if solver_kwargs is None:
        solver_kwargs = {}

    if solver == "dense":
        return jnp.linalg.solve(A, b)

    if solver == "cg":
        x, _ = cg(A, b, **solver_kwargs)
        return x

    raise ValueError(f"Unknown solver: {solver}")


__all__ = ["solve_linear_system"]
