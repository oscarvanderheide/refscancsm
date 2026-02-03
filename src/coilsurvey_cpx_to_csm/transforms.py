"""Coordinate transformation matrices for Philips MRI scan geometries."""

import numpy as np


def transform_to_MPS_refscan(matrix):
    """
    Transform refscan location matrix to 4x4 MPS transformation matrix.
    
    Applies the following operations:
    1. Reorder columns so third column becomes first (columns: [2, 0, 1])
    2. Multiply second row by -1
    3. Cut off top row and add it as fourth column
    4. Add new bottom row [0, 0, 0, 1]

    Note: For target scans, use transform_to_MPS_target() instead.

    Parameters
    ----------
    matrix : numpy.ndarray
        4x3 input matrix from SIN file (location_center_coordinates + location_matrices)

    Returns
    -------
    numpy.ndarray
        4x4 homogeneous transformation matrix in MPS coordinate system
    """
    # Make a copy to avoid modifying the input
    matrix = matrix.copy()
    
    # Step 1: Reorder columns (third column becomes first)
    matrix = matrix[:, [2, 0, 1]]

    # Step 2: Multiply second row by -1
    matrix[1, :] *= -1

    # Step 3: Cut off top row and use it as fourth column
    top_row = matrix[0, :].copy()
    matrix_3x3 = matrix[1:, :]

    # Add top row as fourth column
    matrix_3x4 = np.column_stack([matrix_3x3, top_row])

    # Step 4: Add bottom row [0, 0, 0, 1]
    bottom_row = np.array([[0, 0, 0, 1]])
    matrix_4x4 = np.vstack([matrix_3x4, bottom_row])

    return matrix_4x4


def transform_to_MPS_target(matrix):
    """
    Transform target scan location matrix to 4x4 MPS transformation matrix.
    
    Applies the following operations:
    1. Multiply the second and third rows by -1
    2. Cut off top row and add it as fourth column
    3. Add new bottom row [0, 0, 0, 1]
    4. Multiply the last entry of the third row by -1

    Note: For reference scans, use transform_to_MPS_refscan() instead.

    Parameters
    ----------
    matrix : numpy.ndarray
        4x3 input matrix from SIN file (location_center_coordinates + location_matrices)

    Returns
    -------
    numpy.ndarray
        4x4 homogeneous transformation matrix in MPS coordinate system
    """
    # Make a copy to avoid modifying the input
    matrix = matrix.copy()
    
    # Step 1: Multiply the second and third rows by -1
    matrix[1, :] *= -1
    matrix[2, :] *= -1

    # Step 2: Cut off top row and use it as fourth column
    top_row = matrix[0, :].copy()
    matrix_3x3 = matrix[1:, :]

    # Add top row as fourth column
    matrix_3x4 = np.column_stack([matrix_3x3, top_row])

    # Step 3: Add bottom row [0, 0, 0, 1]
    bottom_row = np.array([[0, 0, 0, 1]])
    matrix_4x4 = np.vstack([matrix_3x4, bottom_row])

    # Step 4: Multiply the last entry of the third row by -1
    matrix_4x4[2, 3] *= -1

    return matrix_4x4


def create_mps_matrix(matrix_size, voxel_sizes):
    """
    Create MPS coordinate system matrix for converting indices to mm coordinates.
    
    This creates a transformation matrix that converts array indices to physical
    coordinates in millimeters, centered at the isocenter of the scan.
    
    Parameters
    ----------
    matrix_size : numpy.ndarray
        Array size [nx, ny, nz]
    voxel_sizes : numpy.ndarray
        Voxel dimensions [dx, dy, dz] in mm
    
    Returns
    -------
    numpy.ndarray
        4x4 transformation matrix from indices to mm coordinates
    """
    # Create diagonal matrix with voxel sizes
    T = np.eye(4)
    T[0, 0] = voxel_sizes[0]
    T[1, 1] = voxel_sizes[1]
    T[2, 2] = voxel_sizes[2]
    
    # Add centering offset (MATLAB convention: -(size/2 + 0.5))
    T[0, 3] = -(matrix_size[0] / 2 + 0.5) * voxel_sizes[0]
    T[1, 3] = -(matrix_size[1] / 2 + 0.5) * voxel_sizes[1]
    T[2, 3] = -(matrix_size[2] / 2 + 0.5) * voxel_sizes[2]
    
    return T
