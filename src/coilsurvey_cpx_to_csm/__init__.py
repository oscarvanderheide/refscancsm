"""Coilsurvey CPX to CSM converter package.

This package provides utilities for reading CPX files and converting
coil survey data to CSM format, along with functions for reading location
matrices and transforming coordinate systems.
"""

# CPX file reading
from .read_cpx import read_cpx

# SIN file reading
from .read_sin import (
    read_location_matrix,
    read_voxel_sizes,
    read_matrix_size,
)

# Coordinate transformations
from .transforms import (
    transform_to_MPS_refscan,
    transform_to_MPS_target,
    create_mps_matrix,
)

# Main workflow
from .workflow import get_csm

__version__ = "0.1.0"

__all__ = [
    "read_cpx",
    "read_location_matrix",
    "read_voxel_sizes",
    "read_matrix_size",
    "transform_to_MPS_refscan",
    "transform_to_MPS_target",
    "create_mps_matrix",
    "get_csm",
]
