import jax
import jax.numpy as jnp

from jax_lens.fitting.imaging import log_likelihood
from jax_lens.lens.tracer import PlaneConfig, TracerConfig, tracer_image
from jax_lens.pipeline import (
    create_likelihood_fn,
    create_likelihood_fn_flat,
    flatten_params,
)


def _grid(n: int = 10, extent: float = 1.0) -> jnp.ndarray:
    coords = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(-extent, extent, n),
            jnp.linspace(-extent, extent, n),
            indexing="ij",
        ),
        axis=-1,
    )
    return coords.reshape(-1, 2)


def test_flatten_unflatten_roundtrip():
    params = {
        "planes": [
            {
                "light": [
                    {
                        "centre": jnp.array([0.0, 0.0]),
                        "intensity": 1.0,
                        "effective_radius": 0.5,
                        "sersic_index": 2.0,
                    }
                ],
                "mass": [],
            },
            {"light": [], "mass": []},
        ]
    }

    flat, struct = flatten_params(params)
    likelihood_fn_flat = create_likelihood_fn_flat(
        config=TracerConfig(
            planes=(
                PlaneConfig(redshift=0.5, light_profile_types=("sersic",), mass_profile_types=()),
                PlaneConfig(redshift=1.0, light_profile_types=(), mass_profile_types=()),
            )
        ),
        grid=_grid(3),
        data=jnp.zeros((9,)),
        noise_map=jnp.ones((9,)),
        param_structure=struct,
        include_noise_norm=False,
    )

    # Just ensure unflattening works inside the wrapper (no error) and is stable under JIT.
    jax.jit(likelihood_fn_flat)(flat)


def test_create_likelihood_fn_matches_direct_log_likelihood():
    grid = _grid(12, extent=1.2)
    image_shape = (12, 12)

    config = TracerConfig(
        planes=(
            PlaneConfig(redshift=0.5, light_profile_types=(), mass_profile_types=("sie",)),
            PlaneConfig(redshift=1.0, light_profile_types=("sersic",), mass_profile_types=()),
        )
    )

    params = {
        "planes": [
            {
                "mass": [
                    {
                        "centre": jnp.array([0.0, 0.0]),
                        "einstein_radius": 1.1,
                        "axis_ratio": 0.8,
                        "angle": 0.4,
                    }
                ],
                "light": [],
            },
            {
                "mass": [],
                "light": [
                    {
                        "centre": jnp.array([0.05, 0.1]),
                        "intensity": 1.0,
                        "effective_radius": 0.3,
                        "sersic_index": 1.2,
                        "axis_ratio": 0.7,
                        "angle": 0.9,
                    }
                ],
            },
        ]
    }

    model = tracer_image(grid, config, params)
    noise_map = jnp.ones_like(model) * 0.05
    data = model + 0.01  # deterministic offset, avoids stochastic test failures

    direct = log_likelihood(data, model, noise_map)

    like_fn = create_likelihood_fn(
        config=config,
        grid=grid,
        data=data,
        noise_map=noise_map,
        psf=None,
        image_shape=image_shape,
    )
    via_pipeline = like_fn(params)

    assert jnp.allclose(via_pipeline, direct, rtol=1e-6, atol=1e-6)


def test_likelihood_grad_matches_finite_difference():
    grid = _grid(10, extent=1.0)

    config = TracerConfig(
        planes=(
            PlaneConfig(redshift=0.5, light_profile_types=(), mass_profile_types=("sie",)),
            PlaneConfig(redshift=1.0, light_profile_types=("sersic",), mass_profile_types=()),
        )
    )

    base_params = {
        "planes": [
            {
                "mass": [
                    {
                        "centre": jnp.array([0.0, 0.0]),
                        "einstein_radius": 1.0,
                        "axis_ratio": 0.85,
                        "angle": 0.3,
                    }
                ],
                "light": [],
            },
            {
                "mass": [],
                "light": [
                    {
                        "centre": jnp.array([0.0, 0.0]),
                        "intensity": 1.0,
                        "effective_radius": 0.25,
                        "sersic_index": 1.0,
                    }
                ],
            },
        ]
    }

    model = tracer_image(grid, config, base_params)
    noise_map = jnp.ones_like(model) * 0.1
    data = model + 0.02

    like_fn = create_likelihood_fn(
        config=config,
        grid=grid,
        data=data,
        noise_map=noise_map,
        include_noise_norm=False,
    )

    def fn(einstein_radius: float) -> jnp.ndarray:
        params = {
            "planes": [
                {
                    "mass": [
                        {
                            **base_params["planes"][0]["mass"][0],
                            "einstein_radius": einstein_radius,
                        }
                    ],
                    "light": [],
                },
                base_params["planes"][1],
            ]
        }
        return like_fn(params)

    grad_jax = jax.grad(fn)(1.0)

    eps = 1e-3
    fd = (fn(1.0 + eps) - fn(1.0 - eps)) / (2 * eps)

    assert jnp.isfinite(grad_jax)
    assert jnp.isfinite(fd)
    assert jnp.allclose(grad_jax, fd, rtol=1e-2, atol=1e-2)

