"""Core functions for reading CPX files and SIN files."""

import numpy as np
import os
import re
import struct
import sys


def oset(seq):
    """Order-preserving unique set."""
    seen = {}
    result = []
    for item in seq:
        if isinstance(item, (tuple, list)):
            marker = tuple(item)
        else:
            marker = item
        if marker not in seen:
            seen[marker] = 1
            result.append(item)
    return result


def filename_extcase(fn):
    """Find correct case-sensitive filename."""
    if os.path.exists(fn):
        return fn
    pn, ext = os.path.splitext(fn)
    bn = os.path.basename(pn)
    dir_path = os.path.dirname(fn) or "."
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            if f.lower() == (bn + ext).lower():
                return os.path.join(dir_path, f)
    return ""


def readCpx(filename):
    """
    Parse and read a .cpx file.
    """

    filename = filename_extcase(filename + ".cpx")

    # open the cpx file
    if type(filename) is not str:
        print("Input filename is not a string.")
        sys.exit(1)

    if os.path.splitext(filename)[1] not in [".cpx", ".CPX"]:
        print("input filename is not a .cpx file")
        sys.exit(1)

    # opens the file
    try:
        fil = open(filename, "rb")
    except IOError:
        print("cannot open .cpx file ", filename)
        sys.exit(1)

    hdr = dict()

    index = 0
    offset = 0
    # pre-allocate info array
    info = np.zeros(
        [10000, 128]
    )  # increased to store matrix_data_blocks and data_offset
    fmt = "<" + "15i2f111i"  # MATLAB format: 15 long + 2 float + 111 long = 512 bytes

    # Read first header to get matrix_data_blocks and data_offset
    fil.seek(0)
    line = fil.read(512)
    header_vals = struct.unpack("<15i2f111i", line)
    h1 = header_vals[0:15]
    h2 = header_vals[17:128]
    matrix_data_blocks = h1[12]  # h1(13) in MATLAB = index 12 in Python
    data_offset = (
        h2[25] if h2[25] != 0 else h1[9]
    )  # h2(26) in MATLAB = index 25 in Python, fallback to h1(10)

    # generate hdr table
    while True:
        fil.seek(offset)
        line = fil.read(512)

        # make sure line exists
        if len(line) == 512:
            h1_vals = struct.unpack("<15i", line[0:60])
            factor_vals = struct.unpack("<2f", line[60:68])
            h2_vals = struct.unpack("<111i", line[68:512])

            # Store in info array: h1 (15) + factors (2) + h2 (111) = 128 values
            info[index, 0:15] = h1_vals
            info[index, 15:17] = factor_vals
            info[index, 17:128] = h2_vals

            # Check if image exists (h1[8] = Complex Matrix Existence)
            if h1_vals[8] == 0:
                break
        else:
            break

        index += 1
        # Use MATLAB's offset calculation with matrix_data_blocks
        offset = (matrix_data_blocks * 512 + data_offset) * index

        # put info into dictionary
        key = "hdr_" + str(index)
        hdr[key] = info[index, :]
    hdr["headerType"] = "cpx"

    # truncate info array
    info = info[:index, :]

    num_images = index

    # pre-allocate data array
    mixes = oset(info[:, 0])  # h1[0] = mix
    locs = oset(info[:, 1])  # h1[1] = stack
    slices = oset(info[:, 2])  # h1[2] = slice
    echoes = oset(info[:, 4])  # h1[4] = echo
    phases = oset(info[:, 5])  # h1[5] = heart phase
    dynamics = oset(info[:, 6])  # h1[6] = dynamics
    rows = oset(info[:, 7])  # h1[7] = segments
    x_size = oset(info[:, 10])  # h1[10] = resolution x
    y_size = oset(info[:, 11])  # h1[11] = resolution y
    coils = oset(info[:, 18])  # h2[1] = coil (h2 starts at index 17)

    nmix = len(mixes)
    nloc = len(locs)
    nslice = len(slices)
    necho = len(echoes)
    ncard = len(phases)
    ndyn = len(dynamics)
    nrow = len(rows)
    nx = int(np.max(x_size))
    ny = int(np.max(y_size))
    nchan = len(coils)

    data_string = np.array(
        ["chan", "mix", "dyn", "card", "echo", "row", "loc", "slice", "y", "x"]
    )
    data = np.zeros(
        [nchan, nmix, ndyn, ncard, necho, nrow, nloc, nslice, ny, nx],
        dtype=np.complex64,
    )

    # read in the cpx file
    for index in range(num_images):
        # Calculate offset using MATLAB's method with matrix_data_blocks
        offset = (matrix_data_blocks * 512 + data_offset) * index + data_offset
        fil.seek(offset)

        mix = mixes.index(info[index, 0])
        loc = locs.index(info[index, 1])
        slice_idx = slices.index(info[index, 2])
        echo = echoes.index(info[index, 4])
        card = phases.index(info[index, 5])
        dyn = dynamics.index(info[index, 6])
        row = rows.index(info[index, 7])
        coil = coils.index(info[index, 18])  # h2[1] at index 18
        nx = int(info[index, 10])
        ny = int(info[index, 11])
        compression_factor = info[index, 13]
        size_bytes = int(8 * nx * ny // compression_factor)

        unparsed_data = fil.read(size_bytes)
        if compression_factor == 1:
            temp_data = np.frombuffer(unparsed_data, dtype=np.float32)
        elif compression_factor == 2:
            temp_data = np.frombuffer(unparsed_data, dtype=np.int16)
        elif compression_factor == 4:
            temp_data = np.frombuffer(unparsed_data, dtype=np.int8)
        temp_data.shape = [ny, nx, 2]
        complex_data = temp_data[:, :, 0] + 1j * temp_data[:, :, 1]
        data[coil, mix, dyn, card, echo, row, loc, slice_idx, :, :] = complex_data

    # setup the data labels
    data_labels = data_string[
        (np.array(data.shape[0 : len(data_string)]) > 1).nonzero()[0]
    ]

    return (data, hdr, data_labels)


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

    Parameters:
    -----------
    sin_file_path : str
        Path to the .sin file

    Returns:
    --------
    numpy.ndarray
        1D array with 3 voxel sizes [x, y, z]
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
    with open(sin_file_path, "r") as f:
        for line in f:
            if "scan_resolutions" in line:
                # Extract integer or float values after the last colon
                values = re.findall(r"[-+]?\d+\.?\d*", line.split(":")[-1])
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
