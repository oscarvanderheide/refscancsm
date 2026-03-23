"""Functions for reading information from .sin files from Philips MR systems."""

import re

import numpy as np


def get_mps_to_xyz_transform(sin_file_path: str, scan_type: str) -> np.ndarray:
    """
    TODO
    """
    if scan_type not in ["refscan", "target"]:
        raise ValueError(f"scan_type must be 'refscan' or 'target', got '{scan_type}'")

    # Getting the translation and linear transformation components from the .sin file
    # is the same for both the refscan and target scans, but the way we build the final
    # affine transformation matrix differs based on the scan type (due to Philips-specific conventions).
    translation = _get_mps_to_xyz_translation_part(sin_file_path)
    linear_part = _get_mps_to_xyz_linear_part(sin_file_path)

    # if scan_type == "refscan":
    # translation = translation[[2, 0, 1]]
    # linear_part = linear_part[:, [2, 0, 1]]

    # if scan_type == "target":
    #     linear_part[0, :] *= -1
    #     linear_part[1, :] *= -1
    #     translation[2] *= -1

    # Build 4x4 matrix:
    # [ rotation | translation]
    # [  0 0 0   |      1     ]
    mps_to_xyz = np.eye(4)
    mps_to_xyz[:3, :3] = linear_part
    mps_to_xyz[:3, 3] = translation

    # if scan_type == "refscan":
    #     # print("Inverting mps_to_xyz for refscan to get xyz_to_mps...")
    # print("Before:")
    # print(mps_to_xyz)
    # print("Actual of source_mps_to_xyz:")
    # print(mps_to_xyz)
    # print("Inverse of source_mps_to_xyz:")
    # print(np.linalg.inv(mps_to_xyz))
    # mps_to_xyz = np.linalg.inv(mps_to_xyz)

    return mps_to_xyz


def get_idx_to_mps_transform(
    sin_file_path: str, scan_type: str = "target"
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
    scan_type : str
        Either "refscan" or "target" (default: "target")

    Returns
    -------
    np.ndarray
        4x4 transformation matrix from array indices to MPS coordinates
    """

    # Get number of voxels and voxel size in each of the three dimensions
    voxel_sizes = get_voxel_sizes(sin_file_path, scan_type)
    matrix_size = get_matrix_size(sin_file_path, scan_type)

    # Create diagonal scaling matrix
    idx_to_mps = np.eye(4)
    idx_to_mps[0, 0] = voxel_sizes[0]
    idx_to_mps[1, 1] = voxel_sizes[1]
    idx_to_mps[2, 2] = voxel_sizes[2]

    # Add centering offset to place origin at isocenter
    # Convention: -(size/2 - 0.5) to match Philips/MATLAB indexing
    idx_to_mps[0, 3] = -(matrix_size[0] / 2 - 0.5) * voxel_sizes[0]
    idx_to_mps[1, 3] = -(matrix_size[1] / 2 - 0.5) * voxel_sizes[1]
    idx_to_mps[2, 3] = -(matrix_size[2] / 2 - 0.5) * voxel_sizes[2]

    return idx_to_mps


def get_voxel_sizes(sin_file_path: str, scan_type: str = "target") -> np.ndarray:
    """
    Reads a .sin file and extracts voxel_sizes.

    For refscan: returns voxel sizes as-is from the file.
    For target: multiplies voxel sizes by the ratio of recon_resolutions to scan_resolutions.

    Parameters
    ----------
    sin_file_path : str
        Path to the .sin file
    scan_type : str
        Either "refscan" or "target" (default: "target")

    Returns
    -------
    numpy.ndarray
        1D array with 3 voxel sizes [x, y, z] in mm
    """
    if scan_type not in ["refscan", "target"]:
        raise ValueError(f"scan_type must be 'refscan' or 'target', got '{scan_type}'")

    # Extract voxel sizes from file
    with open(sin_file_path, "r") as f:
        for line in f:
            if "voxel_sizes" in line:
                # Extract the three float values after the last colon
                values = re.findall(r"[-+]?\d*\.\d+", line.split(":")[-1])
                if len(values) == 3:
                    voxel_sizes = np.array([float(v) for v in values])
                    break
    # else:
    # raise ValueError("Could not find voxel_sizes in file")

    # For refscan, return as-is
    if scan_type == "refscan":
        return voxel_sizes

    # For target, multiply by ratio of recon_resolutions to scan_resolutions.
    # Only apply the ratio for the in-plane (x, y) dimensions; z is always 1.0
    # because for 3D scans recon_z == scan_z, and for multi-2D scans voxel_sizes[2]
    # from the file already encodes the correct inter-slice spacing.
    recon_resolutions = get_matrix_size(sin_file_path, "refscan")
    scan_resolutions = get_matrix_size(sin_file_path, "target")
    resolution_ratio = np.array([
        recon_resolutions[0] / scan_resolutions[0],
        recon_resolutions[1] / scan_resolutions[1],
        1.0,
    ])

    return voxel_sizes * resolution_ratio


def get_matrix_size(sin_file_path: str, scan_type: str) -> np.ndarray:
    """
    Reads a .sin file and extracts the matrix size.

    For refscan: looks for "recon_resolutions"
    For target: looks for "scan_resolutions"

    Parameters
    ----------
    sin_file_path : str
        Path to the .sin file
    scan_type : str
        Either "refscan" or "target"

    Returns
    -------
    numpy.ndarray
        1D array with 3 matrix size values [x, y, z]
    """
    if scan_type not in ["refscan", "target"]:
        raise ValueError(f"scan_type must be 'refscan' or 'target', got '{scan_type}'")

    search_key = "recon_resolutions" if scan_type == "refscan" else "scan_resolutions"
    # Create pattern to match exact parameter name (not as substring of another name)
    pattern = re.compile(rf"\b{search_key}\s+:")

    with open(sin_file_path, "r") as f:
        for line in f:
            if pattern.search(line):
                # Extract integer or float values after the last colon
                values = re.findall(r"[-+]?\d+\.?\d*", line.split(":")[-1])
                if len(values) >= 3:
                    # Return first 3 values (ignore the 4th value which is always 1)
                    result = np.array([float(v) for v in values[:3]])
                    # For multi-2D target scans, each slice is acquired as a separate
                    # 2D k-space, so scan_resolutions has z=1.  The actual number of
                    # slices in the reconstructed 3D volume equals nr_locations.
                    if scan_type == "target" and result[2] == 1:
                        nr_loc = _get_nr_locations(sin_file_path)
                        if nr_loc > 1:
                            result[2] = float(nr_loc)
                    return result

    raise ValueError(f"Could not find {search_key} in file {sin_file_path}")


def _get_nr_locations(sin_file_path: str) -> int:
    """Parse the number of slice locations from a .sin file (returns 1 if not found)."""
    pattern = re.compile(r"\bnr_locations\s+:")
    with open(sin_file_path, "r") as f:
        for line in f:
            if pattern.search(line):
                values = re.findall(r"\d+", line.split(":")[-1])
                if values:
                    return int(values[0])
    return 1


def _get_mps_to_xyz_linear_part(sin_file_path: str) -> np.ndarray:
    """
    Parse .sin file to extract the linear transformation part (scaling/rotation) of the
    affine transformation matrix that is used to map array indices to world coordinates.

    Parameters
    ----------
    sin_file_path : str
        Path to .sin file

    Returns
    -------
    linear_transformation : np.ndarray
        3x3 matrix representing linear transformation of the affine matrix
    """

    linear_transformation = []
    # Patterns to match location data in .sin file
    patterns = [f" 01 {i:02d} 01: location_matrices" for i in range(1, 4)]

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
            f"Could not find complete linear transformation data in {sin_file_path}. "
        )

    return np.array(linear_transformation)


def _get_mps_to_xyz_translation_part(sin_file_path: str) -> np.ndarray:
    """
    Parse .sin file to extract the translation part of the affine transformation matrix that is used to map
    array indices to world coordinates.

    If multiple locations (slices) are present, extracts the center slice's coordinates.

    Parameters
    ----------
    sin_file_path : str
        Path to .sin file

    Returns
    -------
    translation : np.ndarray
        3D translation vector [x, y, z] in mm
    """
    translations = {}
    # Pattern to match location data in .sin file
    # Format: XX XX YY: location_center_coordinates
    # where YY is the location (slice) index.
    pattern = re.compile(r" \d{2} \d{2} (\d{2}): location_center_coordinates")

    with open(sin_file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                loc_idx = int(match.group(1))
                values = re.findall(r"[-+]?\d*\.\d+", line.split(":")[-1])
                if len(values) == 3:
                    translations[loc_idx] = np.array([float(v) for v in values])

    if not translations:
        raise ValueError(f"Could not find translation data in {sin_file_path}. ")

    # If multiple slices are present, pick the middle one
    sorted_indices = sorted(translations.keys())
    n_slices = len(sorted_indices)
    
    if n_slices == 1:
        return translations[sorted_indices[0]]
    
    # For multi-slice, pick center
    # If even, this picks n/2 (e.g. for 40 slices, indices 1..40, it's index 20 or 21 depending on floor)
    # Using floor of (n-1)/2 index in the sorted list.
    center_idx = sorted_indices[(n_slices - 1) // 2]
    return translations[center_idx]
