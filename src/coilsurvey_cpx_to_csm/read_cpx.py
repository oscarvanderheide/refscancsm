"""
CPX File Reader for Philips MRI Scanner Data.

This module provides functionality to read .cpx (Complex Matrix) files from
Philips MRI scanners. These files contain raw k-space or image-space data
from coil sensitivity scans.

File Format Overview:
--------------------
CPX files are binary files with a specific structure:
1. Multiple headers (512 bytes each) - one per image/coil/slice/dynamic
2. Raw data blocks following each header

Header Structure (512 bytes total):
- h1: 15 integers (60 bytes) - Main acquisition parameters
- factors: 2 floats (8 bytes) - Scaling factors
- h2: 111 integers (444 bytes) - Additional parameters

Key Header Fields (h1):
- h1[0]: Mix number
- h1[1]: Location/Stack
- h1[2]: Slice number
- h1[4]: Echo number
- h1[5]: Cardiac phase
- h1[6]: Dynamic scan number
- h1[7]: Segment/Row
- h1[8]: Complex matrix existence flag (0 = no more data)
- h1[9]: Data offset (backup)
- h1[10]: X resolution
- h1[11]: Y resolution
- h1[12]: Matrix data blocks
- h1[13]: Compression factor (1, 2, or 4)

Key Header Fields (h2):
- h2[1] (index 18): Coil number
- h2[25] (index 42): Data offset (primary)

Data Storage:
- Each image is stored as interleaved real/imaginary pairs
- Data type depends on compression factor:
  - Factor 1: float32 (no compression)
  - Factor 2: int16 (2x compression)
  - Factor 4: int8 (4x compression)
- Data is reshaped to [ny, nx, 2] where last dimension is [real, imag]
"""

import numpy as np
import os
import struct
from typing import Tuple, Dict, List


def _get_unique_ordered(values: np.ndarray) -> List[float]:
    """
    Get unique values from array while preserving order of first appearance.
    
    This replaces the custom 'oset' function with native Python functionality.
    Uses dict.fromkeys() which maintains insertion order (Python 3.7+).
    
    Parameters
    ----------
    values : np.ndarray
        Array of values (can contain duplicates)
        
    Returns
    -------
    List[float]
        List of unique values in order of first appearance
    """
    return list(dict.fromkeys(values))


def read_cpx(filepath: str) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Read and parse a Philips CPX (Complex Matrix) file.
    
    This function reads binary CPX files containing MRI coil sensitivity data
    or other complex-valued image data from Philips scanners. The file contains
    headers describing acquisition parameters and raw complex data.
    
    Workflow:
    ---------
    1. Read all 512-byte headers to extract acquisition parameters
    2. Determine array dimensions from unique parameter values
    3. Pre-allocate output array with shape [ncoils, nmix, ndyn, ncard, necho, nrow, nloc, nslice, ny, nx]
    4. Read each image's raw data and place in correct position
    5. Return organized data with metadata
    
    Parameters
    ----------
    filepath : str
        Path to the CPX file (with or without .cpx extension)
        
    Returns
    -------
    data : np.ndarray
        Complex data array with shape [ncoils, nmix, ndyn, ncard, necho, nrow, nloc, nslice, ny, nx]
        - ncoils: Number of receiver coils
        - nmix: Mix number
        - ndyn: Number of dynamics
        - ncard: Cardiac phases
        - necho: Echo number
        - nrow: Segments/rows
        - nloc: Locations/stacks
        - nslice: Slice number
        - ny, nx: Image dimensions
        
    headers : dict
        Dictionary containing all header information:
        - 'hdr_1', 'hdr_2', etc.: Individual image headers (128 values each)
        - 'headerType': Always 'cpx'
        
    data_labels : np.ndarray
        Array of strings indicating which dimensions are used (have size > 1)
        
    Raises
    ------
    ValueError
        If filepath is not a string or file doesn't have .cpx extension
    FileNotFoundError
        If the CPX file cannot be opened
        
    Examples
    --------
    >>> data, headers, labels = read_cpx("path/to/coilscan")
    >>> print(f"Data shape: {data.shape}")
    Data shape: (48, 1, 1, 1, 1, 1, 1, 1, 160, 116)
    >>> print(f"Active dimensions: {labels}")
    Active dimensions: ['chan' 'y' 'x']
    """
    # Validate input
    if not isinstance(filepath, str):
        raise ValueError("Input filepath must be a string")
    
    # Add .cpx extension if not present
    if not filepath.endswith(('.cpx', '.CPX')):
        filepath = filepath + ".cpx"
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CPX file not found: {filepath}")
    
    # Open binary file
    try:
        with open(filepath, "rb") as file:
            # Read first header to get global parameters
            file.seek(0)
            first_header = file.read(512)
            
            if len(first_header) != 512:
                raise ValueError("Invalid CPX file: header too short")
            
            # Unpack first header: 15 ints + 2 floats + 111 ints = 128 values
            header_vals = struct.unpack("<15i2f111i", first_header)
            h1_first = header_vals[0:15]
            h2_first = header_vals[17:128]
            
            # Extract global file parameters
            matrix_data_blocks = h1_first[12]  # Number of 512-byte blocks per image
            data_offset = h2_first[25] if h2_first[25] != 0 else h1_first[9]
            
            # Pre-allocate array for all headers (max 10000 images)
            max_images = 10000
            header_info = np.zeros([max_images, 128])
            
            # Read all headers
            image_count = 0
            file_offset = 0
            
            while image_count < max_images:
                file.seek(file_offset)
                header_bytes = file.read(512)
                
                # Check if we have a complete header
                if len(header_bytes) != 512:
                    break
                
                # Unpack header components
                h1_vals = struct.unpack("<15i", header_bytes[0:60])
                factor_vals = struct.unpack("<2f", header_bytes[60:68])
                h2_vals = struct.unpack("<111i", header_bytes[68:512])
                
                # Store all values in info array
                header_info[image_count, 0:15] = h1_vals
                header_info[image_count, 15:17] = factor_vals
                header_info[image_count, 17:128] = h2_vals
                
                # Check complex matrix existence flag (h1[8])
                # When 0, no more valid data
                if h1_vals[8] == 0:
                    break
                
                image_count += 1
                
                # Calculate next header offset
                # Each image section = (matrix_data_blocks * 512 + data_offset) bytes
                file_offset = (matrix_data_blocks * 512 + data_offset) * image_count
            
            # Truncate to actual number of images found
            header_info = header_info[:image_count, :]
            
            # Store headers in dictionary
            headers = {"headerType": "cpx"}
            for i in range(image_count):
                headers[f"hdr_{i+1}"] = header_info[i, :]
            
            # Extract unique dimension values (order-preserving)
            # These define the output array dimensions
            mixes = _get_unique_ordered(header_info[:, 0])       # h1[0]
            locs = _get_unique_ordered(header_info[:, 1])        # h1[1]
            slices = _get_unique_ordered(header_info[:, 2])      # h1[2]
            echoes = _get_unique_ordered(header_info[:, 4])      # h1[4]
            phases = _get_unique_ordered(header_info[:, 5])      # h1[5]
            dynamics = _get_unique_ordered(header_info[:, 6])    # h1[6]
            rows = _get_unique_ordered(header_info[:, 7])        # h1[7]
            x_sizes = _get_unique_ordered(header_info[:, 10])    # h1[10]
            y_sizes = _get_unique_ordered(header_info[:, 11])    # h1[11]
            coils = _get_unique_ordered(header_info[:, 18])      # h2[1] at index 18
            
            # Determine array dimensions
            n_mix = len(mixes)
            n_loc = len(locs)
            n_slice = len(slices)
            n_echo = len(echoes)
            n_card = len(phases)
            n_dyn = len(dynamics)
            n_row = len(rows)
            n_x = int(np.max(x_sizes))
            n_y = int(np.max(y_sizes))
            n_chan = len(coils)
            
            # Pre-allocate output array
            # Dimension order: [coils, mix, dyn, cardiac, echo, row, loc, slice, y, x]
            data = np.zeros(
                [n_chan, n_mix, n_dyn, n_card, n_echo, n_row, n_loc, n_slice, n_y, n_x],
                dtype=np.complex64,
            )
            
            # Read and organize all image data
            for img_idx in range(image_count):
                # Calculate file offset for this image's data
                file_offset = (matrix_data_blocks * 512 + data_offset) * img_idx + data_offset
                file.seek(file_offset)
                
                # Get parameters for this specific image
                mix_idx = mixes.index(header_info[img_idx, 0])
                loc_idx = locs.index(header_info[img_idx, 1])
                slice_idx = slices.index(header_info[img_idx, 2])
                echo_idx = echoes.index(header_info[img_idx, 4])
                card_idx = phases.index(header_info[img_idx, 5])
                dyn_idx = dynamics.index(header_info[img_idx, 6])
                row_idx = rows.index(header_info[img_idx, 7])
                coil_idx = coils.index(header_info[img_idx, 18])
                
                # Get image dimensions and compression
                img_nx = int(header_info[img_idx, 10])
                img_ny = int(header_info[img_idx, 11])
                compression = header_info[img_idx, 13]
                
                # Calculate data size in bytes
                # Complex data = 2 values (real, imag) per pixel
                # Factor of 8 = 2 (real+imag) * 4 (bytes per float32)
                data_size_bytes = int(8 * img_nx * img_ny // compression)
                
                # Read raw data bytes
                raw_data = file.read(data_size_bytes)
                
                # Parse based on compression factor
                if compression == 1:
                    # No compression: float32
                    temp_data = np.frombuffer(raw_data, dtype=np.float32)
                elif compression == 2:
                    # 2x compression: int16
                    temp_data = np.frombuffer(raw_data, dtype=np.int16)
                elif compression == 4:
                    # 4x compression: int8
                    temp_data = np.frombuffer(raw_data, dtype=np.int8)
                else:
                    raise ValueError(f"Unknown compression factor: {compression}")
                
                # Reshape to [ny, nx, 2] where last dim is [real, imag]
                temp_data = temp_data.reshape([img_ny, img_nx, 2])
                
                # Convert to complex numbers
                complex_data = temp_data[:, :, 0] + 1j * temp_data[:, :, 1]
                
                # Store in output array at correct position
                data[coil_idx, mix_idx, dyn_idx, card_idx, echo_idx, 
                     row_idx, loc_idx, slice_idx, :img_ny, :img_nx] = complex_data
            
            # Create labels for dimensions that are actually used (size > 1)
            dimension_names = np.array([
                "chan", "mix", "dyn", "card", "echo", "row", "loc", "slice", "y", "x"
            ])
            active_dims = np.array(data.shape[:len(dimension_names)]) > 1
            data_labels = dimension_names[active_dims]
            
            return data, headers, data_labels
            
    except IOError as e:
        raise FileNotFoundError(f"Cannot open CPX file: {filepath}") from e
