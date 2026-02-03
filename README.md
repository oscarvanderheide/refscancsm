# Coilsurvey CPX to CSM

A Python package for reading CPX files and converting coil survey data to CSM format. This tool provides both a command-line interface and a Python library for working with MRI coil survey data.

## Features

- Read CPX files with automatic case-insensitive filename handling
- Extract location matrices and transformation data from SIN files
- Transform coordinate systems (MPS refscan and target transformations)
- Read voxel sizes and matrix dimensions
- Command-line tool (`refscan2csm`) for easy file conversion
- Python API for programmatic access

## Installation

### From GitHub (recommended for development)

```bash
# Install with uv
uv add git+https://github.com/yourusername/coilsurvey-cpx-to-csm.git

# Or using pip
pip install git+https://github.com/yourusername/coilsurvey-cpx-to-csm.git
```

### Direct usage with uvx (no installation needed)

```bash
uvx --from git+https://github.com/yourusername/coilsurvey-cpx-to-csm.git refscan2csm <file.cpx>
```

### Local development installation

```bash
# Clone the repository
git clone https://github.com/yourusername/coilsurvey-cpx-to-csm.git
cd coilsurvey-cpx-to-csm

# Install in editable mode
uv pip install -e .
```

## Usage

### Command-line Interface

Read and process a CPX file:

```bash
# Basic usage - reads file and saves to default output
refscan2csm path/to/file.cpx

# Specify output file
refscan2csm path/to/file.cpx -o output.npy

# Show detailed information without saving
refscan2csm path/to/file.cpx --no-save --verbose

# Get help
refscan2csm --help
```

### Python API

```python
from coilsurvey_cpx_to_csm import readCpx, read_location_matrix, transform_to_MPS_refscan
import numpy as np

# Read a CPX file
data, hdr, labels = readCpx("path/to/file")
print(f"Data shape: {data.shape}")
print(f"Labels: {labels}")

# Read location matrix from SIN file
matrix = read_location_matrix("path/to/file.sin", location_idx=1)
print(f"Location matrix:\n{matrix}")

# Transform to MPS coordinate system
T_MPS = transform_to_MPS_refscan(matrix)
print(f"Transformation matrix:\n{T_MPS}")

# Read voxel sizes and matrix dimensions
from coilsurvey_cpx_to_csm import read_voxel_sizes, read_matrix_size

voxel_sizes = read_voxel_sizes("path/to/file.sin")
matrix_size = read_matrix_size("path/to/file.sin")
```

## API Reference

### CPX Reading Functions

- **`readCpx(filename)`**: Read and parse a CPX file
  - Returns: `(data, hdr, data_labels)` tuple
  - `data`: numpy array with shape `[nchan, nmix, ndyn, ncard, necho, nrow, nloc, nslice, ny, nx]`
  - `hdr`: Dictionary containing header information
  - `data_labels`: Array of dimension labels

- **`oset(seq)`**: Order-preserving unique set function

- **`filename_extcase(fn)`**: Find correct case-sensitive filename

### SIN File Reading Functions

- **`read_location_matrix(sin_file_path, location_idx=1)`**: Extract location coordinates and matrices
  - Returns: 4x3 numpy array

- **`read_voxel_sizes(sin_file_path)`**: Extract voxel sizes
  - Returns: 1D array `[x, y, z]`

- **`read_matrix_size(sin_file_path)`**: Extract matrix dimensions
  - Returns: 1D array `[x, y, z]`

### Transformation Functions

- **`transform_to_MPS_refscan(matrix)`**: Transform refscan location matrix to 4x4 MPS coordinate system

- **`transform_to_MPS_target(matrix)`**: Transform target scan location matrix to 4x4 MPS coordinate system

## File Format Support

- **CPX files**: MRI coil survey data files
- **SIN files**: Scanner information files containing geometry and acquisition parameters

## Requirements

- Python >= 3.12
- numpy >= 2.4.1

## Development

```bash
# Install development dependencies
uv sync

# Run the CLI locally
uv run refscan2csm path/to/file.cpx
```

## License

MIT License

## Notes

The code is functional but may benefit from future cleanup, especially the `readCpx` function. Contributions welcome!
