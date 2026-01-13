"""
Voronoi pixelization mapping utilities (Delaunay-based).

Topology is computed via SciPy Delaunay on NumPy arrays. Geometry (weights)
is computed in JAX for differentiability within a fixed topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as onp
import jax
import jax.numpy as jnp
from scipy.spatial import Delaunay


@jax.tree_util.register_pytree_node_class
@dataclass
class PixelizationCache:
    """
    Cache for Delaunay-based mapping.

    point_vertices: (N_points, 3) vertex indices for each point's simplex.
    valid_mask: (N_points,) True for points inside the convex hull.
    neighbor_pairs: (E, 2) unique unordered edges between neighboring seeds.
    """

    point_vertices: jnp.ndarray
    valid_mask: jnp.ndarray
    neighbor_pairs: jnp.ndarray

    def tree_flatten(self):
        children = (
            jnp.asarray(self.point_vertices),
            jnp.asarray(self.valid_mask),
            jnp.asarray(self.neighbor_pairs),
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        point_vertices, valid_mask, neighbor_pairs = children
        return cls(
            point_vertices=point_vertices,
            valid_mask=valid_mask,
            neighbor_pairs=neighbor_pairs,
        )


def build_pixelization_cache(
    seeds: onp.ndarray, points: onp.ndarray
) -> PixelizationCache:
    """
    Build a Delaunay topology cache for mapping points to Voronoi seeds.

    This function is non-JAX and should be called outside jit/grad.
    """
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError("seeds must have shape (N, 2)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")

    delaunay = Delaunay(seeds)
    simplex = delaunay.find_simplex(points)
    valid_mask = simplex >= 0

    point_vertices = onp.full((points.shape[0], 3), -1, dtype=onp.int32)
    point_vertices[valid_mask] = delaunay.simplices[simplex[valid_mask]]

    neighbor_pairs = _neighbor_pairs_from_simplices(delaunay.simplices)

    return PixelizationCache(
        point_vertices=jnp.asarray(point_vertices),
        valid_mask=jnp.asarray(valid_mask.astype(bool)),
        neighbor_pairs=jnp.asarray(neighbor_pairs),
    )


def _neighbor_pairs_from_simplices(simplices: onp.ndarray) -> onp.ndarray:
    edges = onp.concatenate(
        [
            simplices[:, [0, 1]],
            simplices[:, [1, 2]],
            simplices[:, [2, 0]],
        ],
        axis=0,
    )
    edges = onp.sort(edges, axis=1)
    edges = onp.unique(edges, axis=0)
    return edges.astype(onp.int32)


def barycentric_weights(
    points: jnp.ndarray,
    tri_vertices: jnp.ndarray,
    eps: float = 1e-9,
) -> jnp.ndarray:
    """
    Compute barycentric weights for points inside triangles.

    Args:
        points: (N, 2) array of point positions.
        tri_vertices: (N, 3, 2) triangle vertices per point.
        eps: Small diagonal jitter for near-degenerate triangles.
    """
    v0 = tri_vertices[:, 1] - tri_vertices[:, 0]
    v1 = tri_vertices[:, 2] - tri_vertices[:, 0]
    v2 = points - tri_vertices[:, 0]

    mat = jnp.stack([v0, v1], axis=-1)  # (N, 2, 2)
    eye = jnp.eye(2, dtype=mat.dtype)
    mat = mat + eps * eye

    uv = jnp.linalg.solve(mat, v2[..., None])[..., 0]
    w0 = 1.0 - uv[:, 0] - uv[:, 1]
    w1 = uv[:, 0]
    w2 = uv[:, 1]

    return jnp.stack([w0, w1, w2], axis=-1)


def mapping_weights(
    points: jnp.ndarray,
    seeds: jnp.ndarray,
    cache: PixelizationCache,
    eps: float = 1e-9,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute mapping weights for each point given a cached topology.

    Returns:
        weights: (N_points, 3) barycentric weights.
        safe_indices: (N_points, 3) vertex indices (0 for invalid points).
        valid_mask: (N_points,) boolean mask.
    """
    point_vertices = jnp.asarray(cache.point_vertices)
    valid_mask = jnp.asarray(cache.valid_mask)

    safe_indices = jnp.where(valid_mask[:, None], point_vertices, 0)
    tri_vertices = seeds[safe_indices]

    weights = barycentric_weights(points=points, tri_vertices=tri_vertices, eps=eps)
    weights = jnp.where(valid_mask[:, None], weights, 0.0)

    return weights, safe_indices, valid_mask


def apply_mapping(
    weights: jnp.ndarray,
    indices: jnp.ndarray,
    source_values: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply a sparse mapping (3 weights per point) to source values.
    """
    gathered = source_values[indices]
    return jnp.sum(weights * gathered, axis=-1)


def mapping_matrix_dense(
    weights: jnp.ndarray,
    indices: jnp.ndarray,
    n_pixels: int,
) -> jnp.ndarray:
    """
    Build a dense mapping matrix from sparse barycentric weights.
    """
    n_points = weights.shape[0]
    row_idx = jnp.repeat(jnp.arange(n_points)[:, None], 3, axis=1)
    mapping = jnp.zeros((n_points, n_pixels))
    mapping = mapping.at[row_idx, indices].add(weights)
    return mapping


__all__ = [
    "PixelizationCache",
    "build_pixelization_cache",
    "barycentric_weights",
    "mapping_weights",
    "apply_mapping",
    "mapping_matrix_dense",
]
