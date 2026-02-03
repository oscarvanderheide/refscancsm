import numpy as np
import re

def read_location_matrix(sin_file_path, location_idx=1):
    """
    Reads a .sin file and extracts location_center_coordinates and location_matrices
    into a 4x3 matrix.
    
    Parameters:
    -----------
    sin_file_path : str
        Path to the .sin file
    location_idx : int
        Location index to extract (default: 1, for location 01)
    
    Returns:
    --------
    numpy.ndarray
        4x3 matrix where:
        - Row 0: location_center_coordinates
        - Rows 1-3: location_matrices (rows 1, 2, 3)
    """
    location_center = None
    location_matrices = []
    
    # Pattern to match location-specific lines
    loc_pattern = f' 01 00 {location_idx:02d}: location_center_coordinates'
    mat_patterns = [f' 01 {i:02d} {location_idx:02d}: location_matrices' for i in range(1, 4)]
    
    with open(sin_file_path, 'r') as f:
        for line in f:
            if loc_pattern in line:
                # Extract the three float values after the last colon
                values = re.findall(r'[-+]?\d*\.\d+', line.split(':')[-1])
                if len(values) == 3:
                    location_center = [float(v) for v in values]
            
            for mat_pattern in mat_patterns:
                if mat_pattern in line:
                    # Extract the three float values after the last colon
                    values = re.findall(r'[-+]?\d*\.\d+', line.split(':')[-1])
                    if len(values) == 3:
                        location_matrices.append([float(v) for v in values])
    
    # Combine into 4x3 matrix
    if location_center and len(location_matrices) == 3:
        matrix = np.array([location_center] + location_matrices)
        return matrix
    else:
        raise ValueError(f"Could not find all required location data in file for location {location_idx:02d}. " 
                        f"Found center: {location_center is not None}, matrices: {len(location_matrices)}")

def read_voxel_sizes(sin_file_path):
    """
    Reads a .sin file and extracts voxel_sizes.
    
    Parameters:
    -----------
    sin_file_path : str
        Path to the .sin file
    
    Returns:
    --------
    numpy.ndarray
        1D array with 3 voxel sizes [x, y, z]
    """
    with open(sin_file_path, 'r') as f:
        for line in f:
            if 'voxel_sizes' in line:
                # Extract the three float values after the last colon
                values = re.findall(r'[-+]?\d*\.\d+', line.split(':')[-1])
                if len(values) == 3:
                    return np.array([float(v) for v in values])
    
    raise ValueError("Could not find voxel_sizes in file")

def read_matrix_size(sin_file_path):
    """
    Reads a .sin file and extracts the matrix size (stored as scan_resolutions)
    
    Parameters:
    -----------
    sin_file_path : str
        Path to the .sin file
    
    Returns:
    --------
    numpy.ndarray
        1D array with 3 matrix size values [x, y, z]
    """
    with open(sin_file_path, 'r') as f:
        for line in f:
            if 'scan_resolutions' in line:
                # Extract integer or float values after the last colon
                values = re.findall(r'[-+]?\d+\.?\d*', line.split(':')[-1])
                if len(values) >= 3:
                    # Return first 3 values (ignore the 4th value which is always 1)
                    return np.array([float(v) for v in values[:3]])
    
    raise ValueError("Could not find scan_resolutions in file")

def transform_to_MPS_refscan(matrix):
    """
    Transforms the 4x3 matrix with location information from the refscan sin file into a 4x4
    transformation matrix:
    1. Reorder columns so third column becomes first (columns: [2, 0, 1])
    2. Multiply second row by -1
    3. Cut off top row and add it as fourth column
    4. Add new bottom row [0, 0, 0, 1]

    Note that for the target scan different operations need to be applied.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        4x3 input matrix
    
    Returns:
    --------
    numpy.ndarray
        4x4 transformation matrix
    """
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
    Transforms the 4x3 matrix with location information from the sin file of the target scan into a 4x4 transformation matrix:
    1. Multiply the second and third rows by -1
    2. Cut off top row and add it as fourth column
    3. Add new bottom row [0, 0, 0, 1]
    4. Multiply the last entry of the third row by -1
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        4x3 input matrix
    
    Returns:
    --------
    numpy.ndarray
        4x4 transformation matrix
    """
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



# Example usage:
refscan_sin_file = 'patient47/test_nopatch/an_19112024_1620384_1000_2_senserefscanV4.sin'
matrix = read_location_matrix(refscan_sin_file)

print("Original Location Matrix (4x3):")
print(matrix)
print(f"Shape: {matrix.shape}\n")

T_MPS_refscan = transform_to_MPS_refscan(matrix)
print("Transformed Matrix (4x4):")
print(T_MPS_refscan)
print(f"Shape: {T_MPS_refscan.shape}")

target_sin_file = 'patient47/test_nopatch/an_19112024_1625290_3_2_tempo_cst1_gdV4.sin'
matrix = read_location_matrix(target_sin_file)

print("Original Location Matrix (4x3):")
print(matrix)
print(f"Shape: {matrix.shape}\n")

T_MPS_target = transform_to_MPS_target(matrix)
print("\nTransformed Target Matrix (4x4):")
print(T_MPS_target)
print(f"Shape: {T_MPS_target.shape}")

T_MPS_TargetToRefscan = T_MPS_refscan @ T_MPS_target

# Test voxel sizes extraction
voxel_sizes = read_voxel_sizes(refscan_sin_file)
print("\nVoxel sizes from refscan:")
print(voxel_sizes)
print(f"Shape: {voxel_sizes.shape}")

# Test matrix size extraction
matrix_size = read_matrix_size(refscan_sin_file)
print("\nMatrix size from refscan:")
print(matrix_size)
print(f"Shape: {matrix_size.shape}")