"""
Configuration objects for pixelized source reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp


@dataclass
class PixelizationConfig:
    """
    Settings for pixelized source reconstruction.
    """

    regularization_weight: float = 1.0
    solver: str = "dense"
    solver_kwargs: Optional[Dict[str, Any]] = None
    source_plane_index: int = -1
    seeds: Optional[jnp.ndarray] = None


__all__ = ["PixelizationConfig"]
