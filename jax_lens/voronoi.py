"""
Auto-differentiable Voronoi tessellation in JAX.

Topology is extracted with SciPy's Voronoi on host NumPy (non-JAX, non-jittable),
then geometry (vertex positions) is recomputed differentiably in JAX by solving
for triangle circumcenters. Only finite vertices are handled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as onp
import jax.numpy as jnp
from scipy.spatial import Voronoi


def circumcenter(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, eps: float = 1e-9
) -> jnp.ndarray:
    """
    Compute circumcenters for batches of triangles using JAX ops.

    Args:
        a, b, c: Arrays of shape (..., 2) representing triangle vertices.
        eps: Small diagonal jitter to avoid singular systems for near-colinear points.

    Returns:
        Array of shape (..., 2) with circumcenter coordinates.
    """
    ab = b - a
    ac = c - a

    ab_rhs = 0.5 * (jnp.sum(b * b, axis=-1) - jnp.sum(a * a, axis=-1))
    ac_rhs = 0.5 * (jnp.sum(c * c, axis=-1) - jnp.sum(a * a, axis=-1))

    mat = jnp.stack([ab, ac], axis=-2)  # (..., 2, 2)
    rhs = jnp.stack([ab_rhs, ac_rhs], axis=-1)  # (..., 2)

    eye = jnp.eye(2, dtype=mat.dtype)
    mat_reg = mat + eps * eye

    sol = jnp.linalg.solve(mat_reg, rhs[..., None])[..., 0]
    return sol


@dataclass
class AutoDiffVoronoi:
    """
    Compute a Voronoi tessellation with differentiable vertex geometry in JAX.

    Note: Topology extraction uses SciPy and is not JIT-compilable. Geometry
    recomputation is pure JAX and supports autodiff.
    """

    def __call__(self, seeds: jnp.ndarray) -> Tuple[jnp.ndarray, List[List[int]]]:
        if seeds.ndim != 2 or seeds.shape[-1] != 2:
            raise ValueError("seeds must have shape (N, 2)")

        topology = self._compute_topology(seeds)
        vertices = self._recompute_vertices(topology.vertex_seeds)
        faces = self._build_faces(topology, vertices.shape[0])
        return vertices, faces

    @dataclass
    class _Topology:
        vertex_to_index: Dict[int, int]
        vertex_seeds: jnp.ndarray  # (V, 3, 2)
        regions: List[List[int]]

    def _compute_topology(self, seeds: jnp.ndarray) -> _Topology:
        # Host-side SciPy Voronoi on detached NumPy copy.
        pts = onp.asarray(seeds)
        vor = Voronoi(pts)

        vertex_to_seed_ids: Dict[int, List[int]] = {}
        for ridge_pts, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
            for v_idx in ridge_vertices:
                if v_idx < 0:
                    continue  # Skip unbounded / infinite vertices
                vertex_to_seed_ids.setdefault(v_idx, []).extend(ridge_pts.tolist())

        vertex_ids: List[int] = []
        seed_triplets: List[Tuple[int, int, int]] = []
        for v_idx, seeds_for_v in vertex_to_seed_ids.items():
            uniq = list(dict.fromkeys(seeds_for_v))  # Preserve order, remove duplicates
            if len(uniq) < 3:
                continue
            triplet = tuple(uniq[:3])
            vertex_ids.append(v_idx)
            seed_triplets.append(triplet)

        if not seed_triplets:
            raise RuntimeError("No finite Voronoi vertices found.")

        vertex_to_index = {v_id: i for i, v_id in enumerate(vertex_ids)}

        seed_triplets_arr = jnp.asarray(seed_triplets, dtype=jnp.int32)
        gathered_seeds = seeds[seed_triplets_arr]  # (V, 3, 2)

        regions: List[List[int]] = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if not region or -1 in region:
                regions.append([])  # Unbounded region
                continue
            finite_vertices = [vertex_to_index[v] for v in region if v in vertex_to_index]
            regions.append(finite_vertices)

        return self._Topology(
            vertex_to_index=vertex_to_index,
            vertex_seeds=gathered_seeds,
            regions=regions,
        )

    def _recompute_vertices(self, vertex_seeds: jnp.ndarray) -> jnp.ndarray:
        a = vertex_seeds[:, 0]
        b = vertex_seeds[:, 1]
        c = vertex_seeds[:, 2]
        return circumcenter(a, b, c)

    def _build_faces(self, topology: _Topology, num_vertices: int) -> List[List[int]]:
        faces: List[List[int]] = []
        for region_vertices in topology.regions:
            if not region_vertices:
                faces.append([])
                continue
            filtered = [v for v in region_vertices if 0 <= v < num_vertices]
            faces.append(filtered)
        return faces


__all__ = ["AutoDiffVoronoi", "circumcenter"]
