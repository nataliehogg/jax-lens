"""
Pixelized source reconstruction via linear inversion.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import vmap

from jax_lens.inversion.linear import solve_linear_system
from jax_lens.fitting.imaging import convolve_image
from jax_lens.pixelization.mapping import (
    PixelizationCache,
    apply_mapping,
    mapping_matrix_dense,
    mapping_weights,
)
from jax_lens.pixelization.regularization import laplacian_from_neighbors


def normal_equations(
    weights: jnp.ndarray,
    indices: jnp.ndarray,
    data: jnp.ndarray,
    inv_noise2: jnp.ndarray,
    n_pixels: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Assemble normal equations for linear inversion: A s = b.
    """
    b = jnp.zeros((n_pixels,))
    b = b.at[indices].add(weights * data[:, None] * inv_noise2[:, None])

    row_idx = indices[:, :, None]
    col_idx = indices[:, None, :]
    contrib = inv_noise2[:, None, None] * weights[:, :, None] * weights[:, None, :]

    A = jnp.zeros((n_pixels, n_pixels))
    A = A.at[row_idx, col_idx].add(contrib)

    return A, b


def blurred_mapping_matrix(
    mapping: jnp.ndarray,
    image_shape: Tuple[int, int],
    psf: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convolve each column of the mapping matrix with the PSF.
    """
    n_image, n_pixels = mapping.shape
    if n_image != image_shape[0] * image_shape[1]:
        raise ValueError("image_shape does not match mapping size.")

    columns = mapping.T.reshape((n_pixels, image_shape[0], image_shape[1]))
    convolved = vmap(lambda img: convolve_image(img, psf))(columns)
    return convolved.reshape((n_pixels, n_image)).T


def pixelized_source_reconstruction(
    points: jnp.ndarray,
    seeds: jnp.ndarray,
    cache: PixelizationCache,
    data: jnp.ndarray,
    noise_map: jnp.ndarray,
    regularization_weight: float = 1.0,
    solver: str = "dense",
    solver_kwargs: Optional[dict] = None,
    mask: Optional[jnp.ndarray] = None,
    psf: Optional[jnp.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reconstruct a pixelized source and return model image + source values.
    """
    if solver_kwargs is None:
        solver_kwargs = {}

    weights, indices, valid_mask = mapping_weights(points, seeds, cache)

    inv_noise2 = 1.0 / (noise_map**2)
    if mask is not None:
        if mask.shape != inv_noise2.shape:
            mask = mask.reshape(inv_noise2.shape)
        inv_noise2 = inv_noise2 * mask.astype(inv_noise2.dtype)

    inv_noise2 = inv_noise2 * valid_mask.astype(inv_noise2.dtype)

    n_pixels = seeds.shape[0]

    if psf is None:
        A, b = normal_equations(weights, indices, data, inv_noise2, n_pixels)
        mapping = None
    else:
        if image_shape is None:
            raise ValueError("image_shape is required when psf is provided.")
        mapping = mapping_matrix_dense(weights, indices, n_pixels)
        mapping = blurred_mapping_matrix(mapping, image_shape, psf)
        weighted = mapping * inv_noise2[:, None]
        A = mapping.T @ weighted
        b = mapping.T @ (data * inv_noise2)

    if regularization_weight > 0.0:
        neighbor_pairs = jnp.asarray(cache.neighbor_pairs)
        reg = laplacian_from_neighbors(
            neighbor_pairs=neighbor_pairs,
            n_pixels=n_pixels,
            weight=regularization_weight,
        )
        A = A + reg

    source_values = solve_linear_system(A, b, solver=solver, solver_kwargs=solver_kwargs)
    if mapping is None:
        model = apply_mapping(weights, indices, source_values)
    else:
        model = mapping @ source_values

    return model, source_values


__all__ = [
    "pixelized_source_reconstruction",
    "normal_equations",
    "blurred_mapping_matrix",
]
