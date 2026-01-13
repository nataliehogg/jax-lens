"""
vmap-compatible likelihood pipeline for batched MCMC sampling.

This module provides the high-level API for using jax_lens with batched
parameter evaluation, enabling efficient use with samplers like BlackJAX,
NumPyro, or custom MCMC implementations.

The key insight is that we separate:
1. Static configuration (tracer structure, data) - fixed at compile time
2. Dynamic parameters (profile parameters) - vary at runtime, can be batched

Example usage with vmap:
    import jax
    import jax.numpy as jnp
    import jax_lens as jl

    # Create likelihood function with static data
    log_likelihood_fn = jl.create_likelihood_fn(
        config=config,
        grid=grid,
        data=data,
        noise_map=noise_map,
        psf=psf,
    )

    # Single evaluation
    log_like = log_likelihood_fn(params)

    # Batched evaluation (N parameter sets)
    batched_fn = jax.vmap(log_likelihood_fn)
    log_likes = batched_fn(batched_params)  # shape (N,)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import math
from typing import Callable, Optional, Dict, Any, Tuple

from jax_lens.lens.tracer import (
    TracerConfig,
    PlaneConfig,
    compute_plane_image,
    traced_grid_list,
    tracer_image,
)
from jax_lens.fitting.imaging import (
    log_likelihood,
    log_likelihood_chi2_only,
    convolve_image,
)
from jax_lens.cosmology.distances import (
    precompute_distances,
    scaling_factor_from_precomputed,
    Cosmology,
    PLANCK15,
)
from jax_lens.pixelization import PixelizationConfig, PixelizationCache, pixelized_source_reconstruction


def flatten_params(params: dict) -> Tuple[jnp.ndarray, dict]:
    """
    Flatten a nested parameter dictionary to a 1D array.

    This is useful for interfacing with samplers that expect flat parameter vectors.

    Parameters
    ----------
    params : dict
        Nested parameter dictionary

    Returns
    -------
    Tuple[jnp.ndarray, dict]
        - Flattened parameter array
        - Structure information for unflattening
    """
    flat_params = []
    structure = {"planes": []}

    for plane_idx, plane_params in enumerate(params["planes"]):
        plane_struct = {"light": [], "mass": []}

        for light_params in plane_params.get("light", []):
            profile_struct = {}
            for key, value in light_params.items():
                if isinstance(value, jnp.ndarray):
                    profile_struct[key] = ("array", value.shape, len(flat_params))
                    flat_params.extend(value.flatten().tolist())
                else:
                    profile_struct[key] = ("scalar", None, len(flat_params))
                    flat_params.append(float(value))
            plane_struct["light"].append(profile_struct)

        for mass_params in plane_params.get("mass", []):
            profile_struct = {}
            for key, value in mass_params.items():
                if isinstance(value, jnp.ndarray):
                    profile_struct[key] = ("array", value.shape, len(flat_params))
                    flat_params.extend(value.flatten().tolist())
                else:
                    profile_struct[key] = ("scalar", None, len(flat_params))
                    flat_params.append(float(value))
            plane_struct["mass"].append(profile_struct)

        structure["planes"].append(plane_struct)

    return jnp.array(flat_params), structure


def unflatten_params(flat_params: jnp.ndarray, structure: dict) -> dict:
    """
    Unflatten a 1D parameter array back to nested dictionary.

    Parameters
    ----------
    flat_params : jnp.ndarray
        Flattened parameter array
    structure : dict
        Structure information from flatten_params

    Returns
    -------
    dict
        Nested parameter dictionary
    """
    params = {"planes": []}

    for plane_struct in structure["planes"]:
        plane_params = {"light": [], "mass": []}

        for profile_struct in plane_struct["light"]:
            profile_params = {}
            for key, (ptype, shape, idx) in profile_struct.items():
                if ptype == "array":
                    size = int(math.prod(shape))
                    profile_params[key] = flat_params[idx : idx + size].reshape(shape)
                else:
                    profile_params[key] = flat_params[idx]
            plane_params["light"].append(profile_params)

        for profile_struct in plane_struct["mass"]:
            profile_params = {}
            for key, (ptype, shape, idx) in profile_struct.items():
                if ptype == "array":
                    size = int(math.prod(shape))
                    profile_params[key] = flat_params[idx : idx + size].reshape(shape)
                else:
                    profile_params[key] = flat_params[idx]
            plane_params["mass"].append(profile_params)

        params["planes"].append(plane_params)

    return params


def create_likelihood_fn(
    config: TracerConfig,
    grid: jnp.ndarray,
    data: jnp.ndarray,
    noise_map: jnp.ndarray,
    psf: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    include_noise_norm: bool = True,
    precomputed_distances: Optional[dict] = None,
    source_mode: str = "analytic",
    pixelization: Optional[PixelizationConfig] = None,
) -> Callable[[dict], float]:
    """
    Create a likelihood function with static data baked in.

    This returns a pure function that takes only the model parameters,
    making it suitable for use with jax.vmap.

    Parameters
    ----------
    config : TracerConfig
        Static tracer configuration (defines lens system structure)
    grid : jnp.ndarray
        Image-plane grid of (y, x) coordinates
    data : jnp.ndarray
        Observed image data
    noise_map : jnp.ndarray
        Noise map
    psf : jnp.ndarray, optional
        PSF kernel for convolution
    mask : jnp.ndarray, optional
        Boolean mask
    image_shape : Tuple[int, int], optional
        Shape for 2D image (required if psf is provided)
    include_noise_norm : bool
        Whether to include noise normalization in likelihood
    precomputed_distances : dict, optional
        Pre-computed cosmological distances
    source_mode : str
        "analytic" (default) or "pixelization" for Voronoi reconstruction.
    pixelization : PixelizationConfig, optional
        Settings for pixelized source reconstruction (required if source_mode="pixelization").

    Returns
    -------
    Callable
        If source_mode="analytic", maps params -> log_likelihood.
        If source_mode="pixelization", maps (params, PixelizationCache) -> log_likelihood.
    """
    if source_mode not in ("analytic", "pixelization"):
        raise ValueError("source_mode must be 'analytic' or 'pixelization'")

    # Pre-compute scaling factors if not provided
    if precomputed_distances is None and len(config.planes) > 1:
        redshifts = jnp.array([p.redshift for p in config.planes])
        precomputed_distances = precompute_distances(redshifts, config.cosmology)

    # Compute scaling factor matrix
    n_planes = len(config.planes)
    if precomputed_distances is not None:
        scaling_matrix = jnp.zeros((n_planes, n_planes))
        for i in range(n_planes):
            for j in range(i + 1, n_planes):
                scaling_matrix = scaling_matrix.at[i, j].set(
                    scaling_factor_from_precomputed(i, j, n_planes - 1, precomputed_distances)
                )
    else:
        scaling_matrix = None

    # Choose likelihood function
    if include_noise_norm:
        likelihood_fn = log_likelihood
    else:
        likelihood_fn = log_likelihood_chi2_only

    if source_mode == "analytic":
        def _likelihood(params: dict) -> float:
            # Compute model image
            model = tracer_image(grid, config, params, scaling_matrix)

            # Handle PSF convolution if needed
            if psf is not None and image_shape is not None:
                model_2d = model.reshape(image_shape)
                model_2d = convolve_image(model_2d, psf)
                model = model_2d.flatten()

            return likelihood_fn(data, model, noise_map, mask)

        return _likelihood

    if pixelization is None:
        raise ValueError("pixelization config is required for source_mode='pixelization'")
    if psf is not None and image_shape is None:
        raise ValueError("image_shape is required when psf is provided.")

    source_plane_index = pixelization.source_plane_index
    if source_plane_index < 0:
        source_plane_index = len(config.planes) + source_plane_index

    def _pixelized_likelihood(
        params: dict,
        pixelization_cache: PixelizationCache,
    ) -> float:
        traced_grids = traced_grid_list(grid, config, params, scaling_matrix)

        # Compute analytic light in non-source planes.
        total_image = jnp.zeros(grid.shape[:-1])
        for plane_idx, plane_config in enumerate(config.planes):
            if plane_idx == source_plane_index:
                if len(plane_config.light_profile_types) > 0:
                    raise ValueError(
                        "Source plane light profiles are not supported with pixelization."
                    )
                continue
            light_types = plane_config.light_profile_types
            light_params = params["planes"][plane_idx].get("light", [])
            if len(light_types) > 0:
                plane_image = compute_plane_image(
                    traced_grids[plane_idx], light_types, light_params
                )
                total_image = total_image + plane_image

        if psf is not None:
            total_image_2d = total_image.reshape(image_shape)
            total_image = convolve_image(total_image_2d, psf).reshape(-1)

        seeds = None
        if "pixelization" in params and "seeds" in params["pixelization"]:
            seeds = params["pixelization"]["seeds"]
        elif pixelization.seeds is not None:
            seeds = pixelization.seeds
        if seeds is None:
            raise ValueError("Pixelization seeds not provided.")

        data_residual = data - total_image
        source_model, _ = pixelized_source_reconstruction(
            points=traced_grids[source_plane_index],
            seeds=seeds,
            cache=pixelization_cache,
            data=data_residual,
            noise_map=noise_map,
            regularization_weight=pixelization.regularization_weight,
            solver=pixelization.solver,
            solver_kwargs=pixelization.solver_kwargs,
            mask=mask,
            psf=psf,
            image_shape=image_shape,
        )

        model = total_image + source_model
        return likelihood_fn(data, model, noise_map, mask)

    return _pixelized_likelihood


def create_likelihood_fn_flat(
    config: TracerConfig,
    grid: jnp.ndarray,
    data: jnp.ndarray,
    noise_map: jnp.ndarray,
    param_structure: dict,
    psf: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    include_noise_norm: bool = True,
    source_mode: str = "analytic",
    pixelization: Optional[PixelizationConfig] = None,
) -> Callable[[jnp.ndarray], float]:
    """
    Create a likelihood function that takes a flat parameter vector.

    This is useful for interfacing with samplers that expect flat arrays.

    Parameters
    ----------
    config : TracerConfig
        Static tracer configuration
    grid : jnp.ndarray
        Image-plane grid
    data : jnp.ndarray
        Observed image data
    noise_map : jnp.ndarray
        Noise map
    param_structure : dict
        Structure for unflattening parameters (from flatten_params)
    psf : jnp.ndarray, optional
        PSF kernel
    mask : jnp.ndarray, optional
        Boolean mask
    image_shape : Tuple[int, int], optional
        Shape for 2D image
    include_noise_norm : bool
        Whether to include noise normalization

    Returns
    -------
    Callable[[jnp.ndarray], float]
        Function that maps flat_params -> log_likelihood
    """
    # Create the dictionary-based likelihood
    if source_mode != "analytic":
        raise NotImplementedError("Flat parameter interface is not supported for pixelization.")

    dict_likelihood = create_likelihood_fn(
        config=config,
        grid=grid,
        data=data,
        noise_map=noise_map,
        psf=psf,
        mask=mask,
        image_shape=image_shape,
        include_noise_norm=include_noise_norm,
        source_mode=source_mode,
        pixelization=pixelization,
    )

    def _likelihood_flat(flat_params: jnp.ndarray) -> float:
        params = unflatten_params(flat_params, param_structure)
        return dict_likelihood(params)

    return _likelihood_flat


def batched_likelihood(
    likelihood_fn: Callable[[dict], float],
    batched_params: dict,
) -> jnp.ndarray:
    """
    Evaluate likelihood for a batch of parameter sets.

    Parameters
    ----------
    likelihood_fn : Callable
        Likelihood function (from create_likelihood_fn)
    batched_params : dict
        Parameters with batch dimension on all leaves.
        Each leaf array should have shape (batch_size, ...)

    Returns
    -------
    jnp.ndarray
        Log-likelihood values with shape (batch_size,)
    """
    return vmap(likelihood_fn)(batched_params)


def jit_likelihood(
    likelihood_fn: Callable[[dict], float],
) -> Callable[[dict], float]:
    """
    JIT-compile a likelihood function for faster execution.

    Parameters
    ----------
    likelihood_fn : Callable
        Likelihood function to compile

    Returns
    -------
    Callable
        JIT-compiled likelihood function
    """
    return jit(likelihood_fn)


# =============================================================================
# Simplified API for Common Use Cases
# =============================================================================


def create_simple_lens_likelihood(
    grid: jnp.ndarray,
    data: jnp.ndarray,
    noise_map: jnp.ndarray,
    z_lens: float = 0.5,
    z_source: float = 1.0,
    lens_mass_type: str = "sie",
    source_light_type: str = "sersic",
    lens_light_type: Optional[str] = None,
    psf: Optional[jnp.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Callable[[dict], float]:
    """
    Create a likelihood function for a simple lens + source system.

    This is a convenience function for the most common use case.

    Parameters
    ----------
    grid : jnp.ndarray
        Image-plane grid
    data : jnp.ndarray
        Observed image data
    noise_map : jnp.ndarray
        Noise map
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift
    lens_mass_type : str
        Type of lens mass profile
    source_light_type : str
        Type of source light profile
    lens_light_type : str, optional
        Type of lens light profile (if any)
    psf : jnp.ndarray, optional
        PSF kernel
    image_shape : Tuple[int, int], optional
        Shape for 2D image

    Returns
    -------
    Callable[[dict], float]
        Likelihood function

    Example
    -------
    >>> log_like_fn = create_simple_lens_likelihood(grid, data, noise_map)
    >>> params = {
    ...     'lens_mass': {'centre': jnp.array([0., 0.]), 'einstein_radius': 1.0, 'axis_ratio': 0.8, 'angle': 0.5},
    ...     'source_light': {'centre': jnp.array([0.1, 0.1]), 'intensity': 1.0, 'effective_radius': 0.3, 'sersic_index': 1.0},
    ... }
    >>> log_like = log_like_fn(params)
    """
    # Build config
    if lens_light_type is not None:
        lens_plane = PlaneConfig(
            redshift=z_lens,
            light_profile_types=(lens_light_type,),
            mass_profile_types=(lens_mass_type,),
        )
    else:
        lens_plane = PlaneConfig(
            redshift=z_lens,
            light_profile_types=(),
            mass_profile_types=(lens_mass_type,),
        )

    source_plane = PlaneConfig(
        redshift=z_source,
        light_profile_types=(source_light_type,),
        mass_profile_types=(),
    )

    config = TracerConfig(planes=(lens_plane, source_plane))

    def _likelihood(simple_params: dict) -> float:
        # Convert simple params to full structure
        params = {
            "planes": [
                {
                    "mass": [simple_params["lens_mass"]],
                    "light": [simple_params["lens_light"]] if "lens_light" in simple_params else [],
                },
                {
                    "mass": [],
                    "light": [simple_params["source_light"]],
                },
            ]
        }

        # Compute model
        model = tracer_image(grid, config, params)

        # Handle PSF
        if psf is not None and image_shape is not None:
            model_2d = model.reshape(image_shape)
            model_2d = convolve_image(model_2d, psf)
            model = model_2d.flatten()

        return log_likelihood(data, model, noise_map)

    return _likelihood


# =============================================================================
# Gradient and Hessian Functions
# =============================================================================


def likelihood_gradient(
    likelihood_fn: Callable[[dict], float],
    params: dict,
) -> dict:
    """
    Compute gradient of log-likelihood with respect to parameters.

    Parameters
    ----------
    likelihood_fn : Callable
        Likelihood function
    params : dict
        Parameter values

    Returns
    -------
    dict
        Gradient with same structure as params
    """
    grad_fn = jax.grad(likelihood_fn)
    return grad_fn(params)


def likelihood_value_and_gradient(
    likelihood_fn: Callable[[dict], float],
    params: dict,
) -> Tuple[float, dict]:
    """
    Compute log-likelihood and gradient simultaneously.

    More efficient than computing them separately.

    Parameters
    ----------
    likelihood_fn : Callable
        Likelihood function
    params : dict
        Parameter values

    Returns
    -------
    Tuple[float, dict]
        (log_likelihood, gradient)
    """
    return jax.value_and_grad(likelihood_fn)(params)
