"""RefScanCSM converter package.

This package provides utilities for reading CPX files and converting
coil survey data to CSM format, along with functions for reading location
matrices and transforming coordinate systems.
"""

# CPX file reading
from .parse_cpx import read_cpx

# SIN file reading
from .parse_sin import (
    get_mps_to_xyz_transform,
    get_idx_to_mps_transform,
    get_matrix_size,
    get_voxel_sizes,
)

# Main workflow
from .get_csm import get_csm

from .espirit import espirit, fft3c

__version__ = "0.1.0"

__all__ = [
    "read_cpx",
    "get_mps_to_xyz_transform",
    "get_idx_to_mps_transform",
    "get_matrix_size",
    "get_voxel_sizes",
    "get_csm",
    "espirit",
    "fft3c",
]
