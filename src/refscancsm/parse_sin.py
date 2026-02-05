"""Functions for reading information from .sin files from Philips MR systems."""

import numpy as np
import re

def get_mps_to_xyz_transform(
    sin_file_path: str, scan_type: str, location_idx: int = 1
) -> np.ndarray:
    """
    TODO
    """
    if scan_type not in ["source", "target"]:
        raise ValueError(f"scan_type must be 'source' or 'target', got '{scan_type}'")

    # Getting the translation and linear transformation components from the .sin file
    # is the same for both the source and target scans, but the way we build the final
    # affine transformation matrix differs based on the scan type (due to Philips-specific conventions).
    translation = _get_mps_to_xyz_translation_part(sin_file_path, location_idx)
    linear_part = _get_mps_to_xyz_linear_part(sin_file_path, location_idx)

    if scan_type == "source":

        translation = translation[[2, 0, 1]]
        linear_part = linear_part[:, [2, 0, 1]]
        linear_part[0, :] *= -1

    elif scan_type == "target":

        linear_part[0, :] *= -1
        linear_part[1, :] *= -1
        translation[2] *= -1
    
    # Build 4x4 matrix: 
    # [ rotation | translation]
    # [  0 0 0   |      1     ]
    mps_to_xyz = np.eye(4)
    mps_to_xyz[:3, :3] = linear_part
    mps_to_xyz[:3, 3] = translation

    return mps_to_xyz

def get_idx_to_mps_transform(sin_file_path: str
) -> np.ndarray:
    """
    Create 4x4 matrix that converts array indices (augmented with a 1) to coordinates in the
    MPS (Measurement, Phase, Slice) system of the scan.
    
    This matrix scales by voxel size and centers the coordinate system at the volume's
    isocenter.
    
    Parameters
    ----------
    sin_file_path : str
        Path to the .sin file
    
    Returns
    -------
    np.ndarray
        4x4 transformation matrix from array indices to MPS coordinates
    """

    # Get number of voxels and voxel size in each of the three dimensions
    voxel_sizes = get_voxel_sizes(sin_file_path)
    matrix_size = get_matrix_size(sin_file_path)

    # Create diagonal scaling matrix
    idx_to_mps = np.eye(4)
    idx_to_mps[0, 0] = voxel_sizes[0]
    idx_to_mps[1, 1] = voxel_sizes[1]
    idx_to_mps[2, 2] = voxel_sizes[2]
    
    # Add centering offset to place origin at isocenter
    # Convention: -(size/2 + 0.5) to match Philips/MATLAB indexing
    idx_to_mps[0, 3] = -(matrix_size[0] / 2 + 0.5) * voxel_sizes[0]
    idx_to_mps[1, 3] = -(matrix_size[1] / 2 + 0.5) * voxel_sizes[1]
    idx_to_mps[2, 3] = -(matrix_size[2] / 2 + 0.5) * voxel_sizes[2]
    
    return idx_to_mps


def get_voxel_sizes(sin_file_path: str) -> np.ndarray:
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


def get_matrix_size(sin_file_path: str) -> np.ndarray:
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


def _get_mps_to_xyz_linear_part(
    sin_file_path: str, location_idx: int
) -> np.ndarray:
    """
    Parse .sin file to extract the linear transformation part (scaling/rotation) of the
    affine transformation matrix that is used to map array indices to world coordinates.

    Parameters
    ----------
    sin_file_path : str
        Path to .sin file
    location_idx : int
        Location index to extract

    Returns
    -------
    linear_transformation : np.ndarray
        3x3 matrix representing linear transformation of the affine matrix
    """
    
    linear_transformation = []
    # Patterns to match location data in .sin file
    patterns = [
        f" 01 {i:02d} {location_idx:02d}: location_matrices" for i in range(1, 4)
    ]

    with open(sin_file_path, "r") as f:
        for line in f:
            # Extract linear transformation values (3 rows)
            for pattern in patterns:
                if pattern in line:
                    values = re.findall(r"[-+]?\d*\.\d+", line.split(":")[-1])
                    if len(values) == 3:
                        linear_transformation.append([float(v) for v in values])

    # Validate that we found all required data
    if len(linear_transformation) != 3:
        raise ValueError(
            f"Could not find complete linear transformation data in {sin_file_path} for location {location_idx:02d}. "
        )

    return np.array(linear_transformation)


def _get_mps_to_xyz_translation_part(
    sin_file_path: str, location_idx: int
) -> np.ndarray:
    """
    Parse .sin file to extract the translation part of the affine transformation matrix that is used to map
    array indices to world coordinates.

    Parameters
    ----------
    sin_file_path : str
        Path to .sin file
    location_idx : int
        Location index to extract

    Returns
    -------
    translation : np.ndarray
        3D translation vector [x, y, z] in mm
    """
    translation = None
    # Pattern to match location data in .sin file
    pattern = f" 01 00 {location_idx:02d}: location_center_coordinates"

    with open(sin_file_path, "r") as f:
        for line in f:
            # Extract translation coordinates
            if pattern in line:
                values = re.findall(r"[-+]?\d*\.\d+", line.split(":")[-1])
                if len(values) == 3:
                    translation = np.array([float(v) for v in values])


    # Validate that we found all required data
    if translation is None:
        raise ValueError(
            f"Could not find translation data in {sin_file_path} for location {location_idx:02d}. "
        )

    return translation