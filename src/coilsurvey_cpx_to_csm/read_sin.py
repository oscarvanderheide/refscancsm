"""Functions for reading SIN files from Philips MRI scanners."""

import numpy as np
import re


def read_location_matrix(sin_file_path, location_idx=1):
    """
    Reads a .sin file and extracts location_center_coordinates and location_matrices
    into a 4x3 matrix.

    Parameters
    ----------
    sin_file_path : str
        Path to the .sin file
    location_idx : int
        Location index to extract (default: 1, for location 01)

    Returns
    -------
    numpy.ndarray
        4x3 matrix where:
        - Row 0: location_center_coordinates
        - Rows 1-3: location_matrices (rows 1, 2, 3)
    """
    location_center = None
    location_matrices = []

    # Pattern to match location-specific lines
    loc_pattern = f" 01 00 {location_idx:02d}: location_center_coordinates"
    mat_patterns = [
        f" 01 {i:02d} {location_idx:02d}: location_matrices" for i in range(1, 4)
    ]

    with open(sin_file_path, "r") as f:
        for line in f:
            if loc_pattern in line:
                # Extract the three float values after the last colon
                values = re.findall(r"[-+]?\d*\.\d+", line.split(":")[-1])
                if len(values) == 3:
                    location_center = [float(v) for v in values]

            for mat_pattern in mat_patterns:
                if mat_pattern in line:
                    # Extract the three float values after the last colon
                    values = re.findall(r"[-+]?\d*\.\d+", line.split(":")[-1])
                    if len(values) == 3:
                        location_matrices.append([float(v) for v in values])

    # Combine into 4x3 matrix
    if location_center and len(location_matrices) == 3:
        matrix = np.array([location_center] + location_matrices)
        return matrix
    else:
        raise ValueError(
            f"Could not find all required location data in file for location {location_idx:02d}. "
            f"Found center: {location_center is not None}, matrices: {len(location_matrices)}"
        )


def read_voxel_sizes(sin_file_path):
    """
    Reads a .sin file and extracts voxel_sizes.

    Parameters
    ----------
    sin_file_path : str
        Path to the .sin file

    Returns
    -------
    numpy.ndarray
        1D array with 3 voxel sizes [x, y, z] in mm
    """
    with open(sin_file_path, "r") as f:
        for line in f:
            if "voxel_sizes" in line:
                # Extract the three float values after the last colon
                values = re.findall(r"[-+]?\d*\.\d+", line.split(":")[-1])
                if len(values) == 3:
                    return np.array([float(v) for v in values])

    raise ValueError("Could not find voxel_sizes in file")


def read_matrix_size(sin_file_path):
    """
    Reads a .sin file and extracts the matrix size (stored as scan_resolutions).

    Parameters
    ----------
    sin_file_path : str
        Path to the .sin file

    Returns
    -------
    numpy.ndarray
        1D array with 3 matrix size values [x, y, z]
    """
    with open(sin_file_path, "r") as f:
        for line in f:
            if "scan_resolutions" in line:
                # Extract integer or float values after the last colon
                values = re.findall(r"[-+]?\d+\.?\d*", line.split(":")[-1])
                if len(values) >= 3:
                    # Return first 3 values (ignore the 4th value which is always 1)
                    return np.array([float(v) for v in values[:3]])

    raise ValueError("Could not find scan_resolutions in file")
