"""
Data loading utilities for real imaging datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as onp


@dataclass
class ImagingDataset:
    """
    Container for imaging data and associated metadata.
    """

    data: jnp.ndarray
    noise_map: jnp.ndarray
    psf: Optional[jnp.ndarray]
    mask: Optional[jnp.ndarray]
    pixel_scale: float
    grid: jnp.ndarray
    grid_flat: jnp.ndarray

    @property
    def data_flat(self) -> jnp.ndarray:
        return self.data.reshape(-1)

    @property
    def noise_flat(self) -> jnp.ndarray:
        return self.noise_map.reshape(-1)


def _load_fits(path: Path) -> Tuple[onp.ndarray, dict]:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("astropy is required to read FITS files") from exc

    with fits.open(path) as hdul:
        data = onp.array(hdul[0].data, dtype=onp.float64)
        header = dict(hdul[0].header)
    return data, header


def _infer_pixel_scale(header: dict, pixel_scale: Optional[float]) -> float:
    if pixel_scale is not None:
        return float(pixel_scale)

    for key in ("PIXSCALE", "PIXSCALE1", "PIXSCALE2"):
        if key in header:
            return float(header[key])

    if "CDELT1" in header:
        return abs(float(header["CDELT1"])) * 3600.0
    if "CD1_1" in header:
        return abs(float(header["CD1_1"])) * 3600.0

    raise ValueError("Pixel scale not found in FITS header; pass pixel_scale explicitly.")


def _create_grid(shape: Tuple[int, int], pixel_scale: float) -> jnp.ndarray:
    ny, nx = shape
    y = jnp.linspace(-(ny - 1) / 2 * pixel_scale, (ny - 1) / 2 * pixel_scale, ny)
    x = jnp.linspace(-(nx - 1) / 2 * pixel_scale, (nx - 1) / 2 * pixel_scale, nx)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    return jnp.stack([yy, xx], axis=-1)


def load_imaging_dataset(
    directory: str | Path,
    data_filename: str = "data.fits",
    noise_filename: str = "noise_map.fits",
    psf_filename: Optional[str] = "psf.fits",
    mask_filename: Optional[str] = None,
    pixel_scale: Optional[float] = None,
    mask_threshold: float = 0.5,
    mask_is_valid: bool = False,
    normalize_psf: bool = True,
) -> ImagingDataset:
    """
    Load an imaging dataset from a directory containing FITS files.

    mask_is_valid controls interpretation of the mask file:
      - True: mask values > threshold are kept (True).
      - False: mask values > threshold are excluded (True becomes keep = False).
    """
    directory = Path(directory)

    data, header = _load_fits(directory / data_filename)
    noise_map, _ = _load_fits(directory / noise_filename)

    psf = None
    if psf_filename is not None:
        psf, _ = _load_fits(directory / psf_filename)
        if normalize_psf and psf.sum() != 0.0:
            psf = psf / psf.sum()

    mask = None
    if mask_filename is not None:
        mask_data, _ = _load_fits(directory / mask_filename)
        mask_bool = mask_data > mask_threshold
        if mask_is_valid:
            mask = mask_bool
        else:
            mask = ~mask_bool

    pix = _infer_pixel_scale(header, pixel_scale)
    grid = _create_grid(data.shape, pix)

    return ImagingDataset(
        data=jnp.asarray(data),
        noise_map=jnp.asarray(noise_map),
        psf=jnp.asarray(psf) if psf is not None else None,
        mask=jnp.asarray(mask) if mask is not None else None,
        pixel_scale=pix,
        grid=grid,
        grid_flat=grid.reshape(-1, 2),
    )


__all__ = ["ImagingDataset", "load_imaging_dataset"]
