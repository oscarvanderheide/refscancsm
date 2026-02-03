"""Coilsurvey CPX to CSM converter package.

This package provides utilities for reading CPX files and converting
coil survey data to CSM format, along with functions for reading location
matrices and transforming coordinate systems.
"""

from .reader import (
    readCpx,
    oset,
    filename_extcase,
    read_location_matrix,
    read_voxel_sizes,
    read_matrix_size,
    transform_to_MPS_refscan,
    transform_to_MPS_target,
)

__version__ = "0.1.0"

__all__ = [
    "readCpx",
    "oset",
    "filename_extcase",
    "read_location_matrix",
    "read_voxel_sizes",
    "read_matrix_size",
    "transform_to_MPS_refscan",
    "transform_to_MPS_target",
]
