"""
Regularization utilities for Voronoi pixelizations.
"""

from __future__ import annotations

import jax.numpy as jnp


def laplacian_from_neighbors(
    neighbor_pairs: jnp.ndarray,
    n_pixels: int,
    weight: float = 1.0,
) -> jnp.ndarray:
    """
    Build a graph Laplacian regularization matrix from neighbor pairs.
    """
    reg = jnp.zeros((n_pixels, n_pixels))

    idx_i = neighbor_pairs[:, 0]
    idx_j = neighbor_pairs[:, 1]

    reg = reg.at[idx_i, idx_i].add(weight)
    reg = reg.at[idx_j, idx_j].add(weight)
    reg = reg.at[idx_i, idx_j].add(-weight)
    reg = reg.at[idx_j, idx_i].add(-weight)

    return reg


__all__ = ["laplacian_from_neighbors"]
