# Coilsurvey CPX to CSM

A Python package for reading CPX files from a Philips SENSE refscan and interpolating coil sensitivity maps onto target scan geometries. This tool provides both a command-line interface and a Python library for working with MRI coil survey data.

## Features

- Read CPX files with automatic case-insensitive filename handling
- Extract location matrices and transformation data from SIN files
- Transform coordinate systems (MPS refscan and target transformations)
- Read voxel sizes and matrix dimensions
- **Interpolate coil sensitivity maps from reference to target geometry**
- Handle different scan planning (rotation, translation, voxel sizes)
- Command-line tool (`refscan2csm`) for easy coil map interpolation
- Python API for programmatic access
- Output to NumPy (.npy) or MATLAB (.mat) formats

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

**Main use case: Interpolate coil maps from reference scan to target scan geometry**

```bash
# Basic usage - interpolate and save to .npy
refscan2csm refscan.cpx refscan.sin target.sin

# Specify output file
refscan2csm refscan.cpx refscan.sin target.sin -o my_coil_maps.npy

# Save as MATLAB file
refscan2csm refscan.cpx refscan.sin target.sin -o coil_maps.mat --matlab

# Use cubic interpolation (default is linear)
refscan2csm refscan.cpx refscan.sin target.sin --interp-order 3

# Show detailed information without saving
refscan2csm refscan.cpx refscan.sin target.sin --no-save --verbose

# Get help
refscan2csm --help
```

**Arguments:**
- `refscan_cpx`: Path to reference scan CPX file (coil survey data)
- `refscan_sin`: Path to reference scan SIN file (geometry info)
- `target_sin`: Path to target scan SIN file (target geometry)

**Options:**
- `-o, --output`: Output file path (default: coil_maps_interpolated.npy)
- `--matlab`: Save in MATLAB .mat format
- `--interp-order {0,1,3}`: Interpolation order (0=nearest, 1=linear, 3=cubic)
- `--location-idx`: Location index from SIN files (default: 1)
- `--no-save`: Don't save output, just show info
- `-v, --verbose`: Show detailed information

### Python API

**Full workflow: Interpolate coil maps to target geometry**

```python
from coilsurvey_cpx_to_csm import interpolate_coil_maps
import numpy as np

# Interpolate coil sensitivity maps from reference to target geometry
coil_maps, metadata = interpolate_coil_maps(
    refscan_cpx_path="path/to/refscan",  # without .cpx extension
    refscan_sin_path="path/to/refscan.sin",
    target_sin_path="path/to/target.sin",
    location_idx=1,
    interpolation_order=1,  # 0=nearest, 1=linear, 3=cubic
)

print(f"Interpolated coil maps shape: {coil_maps.shape}")  # [ncoils, nz, ny, nx]
print(f"Number of coils: {metadata['ncoils']}")
print(f"Target voxel sizes: {metadata['target_voxel_sizes']} mm")

# Save the result
np.save("coil_maps.npy", coil_maps)
```

**Lower-level API: Read individual files**

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

### Main Interpolation Function

- **`interpolate_coil_maps(refscan_cpx_path, refscan_sin_path, target_sin_path, location_idx=1, interpolation_order=1)`**
  - Complete workflow to interpolate coil maps from reference to target geometry
  - Returns: `(interpolated_coil_maps, metadata)` tuple
  - `interpolated_coil_maps`: numpy array [ncoils, nz, ny, nx] in target geometry
  - `metadata`: dict with geometry information

### CPX Reading Functions

- **`readCpx(filename)`**: Read and parse a CPX file
  - Returns: `(data, hdr, data_labels)` tuple
  - `data`: numpy array with shape `[nchan, nmix, ndyn, ncard, necho, nrow, nloc, nslice,
    ny, nx]`
  - `hdr`: Dictionary containing header information
  - `data_labels`: Array of dimension labels

- **`oset(seq)`**: Order-preserving unique set function

- **`filename_extcase(fn)`**: Find correct case-sensitive filename

### SIN File Reading Functions

- **`read_location_matrix(sin_file_path, location_idx=1)`**: Extract location coordinates
  and matrices
  - Returns: 4x3 numpy array

- **`read_voxel_sizes(sin_file_path)`**: Extract voxel sizes
  - Returns: 1D array `[x, y, z]`

- **`read_matrix_size(sin_file_path)`**: Extract matrix dimensions
  - Returns: 1D array `[x, y, z]`

### Transformation Functions

- **`transform_to_MPS_refscan(matrix)`**: Transform refscan location matrix to 4x4 MPS
  coordinate system

- **`transform_to_MPS_target(matrix)`**: Transform target scan location matrix to 4x4 MPS
  coordinate system

## File Format Support

- **CPX files**: MRI coil survey data files
- **SIN files**: Scanner information files containing geometry and acquisition parameters

## Requirements

- Python >= 3.12
- numpy >= 2.4.1
- scipy >= 1.11.0 (for interpolation and optional MATLAB output)

## Interpolation Method

The tool uses linear interpolation by default (order=1) to avoid overshoots at mask boundaries where coil sensitivities drop to zero. This is similar to MATLAB's "makima" approach:

- **Linear (order=1, default)**: Best for masked data, avoids oscillations at boundaries
- **Nearest (order=0)**: No interpolation, fastest but blocky results
- **Cubic (order=3)**: Smoothest but may overshoot at mask edges

The coordinate transformation accounts for:
- Different voxel sizes between scans
- Different matrix dimensions
- Different scan planning (rotation, translation, angulation)
- Proper centering conventions

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

The code is functional but may benefit from future cleanup, especially the `readCpx`
function. Contributions welcome!

# Philips MRI Coil Sensitivity Map Extraction from CPX Files

## Overview
This project extracts coil sensitivity maps from Philips MRI scanner data (CPX format) and interpolates them onto target scan geometries for parallel imaging reconstruction.

---

## Coil Map Interpolation: Detailed Walkthrough

**The Problem:**
You have coil sensitivity maps from a **reference scan** (160×116×150 voxels, 4.4mm resolution) and need them on a **target scan** (224×224×192 voxels, 1mm resolution). The two scans have different sizes, resolutions, AND different **scan planning** (different slice angles, tilting, positioning in the scanner).

**The Core Question:**
For every single voxel position in the target array, what coil sensitivity value should go there?

---

## The Solution: Trace Backwards Through Coordinate Systems

### Starting Point
You have an empty target array of size [224, 224, 192]. Pick any arbitrary voxel, say **target index [112, 112, 96]**. This is just an array position - it doesn't mean anything physical yet. You need to fill it with the correct coil sensitivity value.

### Step 1: Target array index → Physical coordinates in target scan space

**The question:** "Index [112, 112, 96] in my target array - where is that in the real world?"

**What you do:** Apply the target scan's voxel size and figure out where this voxel sits relative to the target scan's center (its isocenter).

```
TargetMPS matrix: converts indices → millimeters
- Uses target voxel size: [1mm, 1mm, 1mm]
- Centers the coordinate system: offset = -(224/2 + 0.5) = -112.5

Target index [112, 112, 96]:
  → x: 1mm × (112 - 112.5) = -0.5mm from center
  → y: 1mm × (112 - 112.5) = -0.5mm from center  
  → z: 1mm × (96 - 96.5) = -0.5mm from center

Result: Position [-0.5mm, -0.5mm, -0.5mm] in the target scan's own coordinate system
```

**Why this matters:** Now you know where this voxel is in physical space (millimeters), not just array indices.

---

### Step 2: Target scan coordinates → Reference scan coordinates

**The question:** "This physical position [-0.5, -0.5, -0.5]mm in my target scan - where is that same anatomical location in the reference scan's coordinate system?"

**The problem:** The two scans were planned differently:
- The target T1w might be tilted 5° for better brain coverage
- The reference coil scan might be straight axial slices
- They might have different centering in the scanner bore
- Different slice orientations, angulations, offsets

**What you do:** Apply the transformation matrix **T_MPS_TargetToSource** that encodes the rotation and translation between the two scan geometries.

```
T_MPS_TargetToSource matrix: 4×4 matrix from .sin files
- Off-diagonal terms = rotation (different slice angles)
- Last column = translation (different scan centers)

Position in target space: [-0.5, -0.5, -0.5]mm
  → Multiply by T_MPS_TargetToSource
  → Position in reference space: [10.2, -5.3, 8.7]mm

This is the SAME anatomical point, but expressed in the reference scan's coordinate system
```

**Why this matters:** The same piece of anatomy appears at different coordinates in each scan because they were planned differently. This step accounts for that mismatch.

---

### Step 3: Reference scan coordinates → Reference array index

**The question:** "Position [10.2, -5.3, 8.7]mm in the reference scan - which array index does that correspond to in the reference coil map data?"

**What you do:** Reverse the process from Step 1, but using the reference scan's voxel size and dimensions.

```
SourceMPS inverse: converts millimeters → indices
- Uses reference voxel size: [4.375mm, 4.375mm, 4.667mm]  
- Centers using reference dimensions: 160×116×150

Position [10.2, -5.3, 8.7]mm:
  → x: 10.2mm / 4.375mm = 2.33 voxels from center
      Add center offset (80): 2.33 + 80 = 82.33
  → y: -5.3mm / 4.375mm = -1.21 voxels from center
      Add center offset (58): -1.21 + 58 = 56.79
  → z: 8.7mm / 4.667mm = 1.86 voxels from center
      Add center offset (75): 1.86 + 75 = 76.86

Result: Fractional array index [82.33, 56.79, 76.86] in the reference data
```

**Why this matters:** You now know exactly where to look in the reference coil map array to get the value for your original target voxel.

---

### Step 4: Sample the reference coil map

**The question:** "Array index [82.33, 56.79, 76.86] - what's the coil sensitivity value there?"

**The problem:** The indices are fractional (not integers), so you can't directly index the array.

**What you do:** Use **3D interpolation** (makima method) to estimate the value at fractional positions by blending nearby integer positions.

```
For each of 44 coils:
  → Read reference coil map at fractional position [82.33, 56.79, 76.86]
  → Get interpolated complex value (e.g., 0.8 + 0.3i)
  → Store this in target array at original position [112, 112, 96]
```

**Why makima not spline:** Makima respects the masking (doesn't overshoot at boundaries where coil sensitivity drops to zero).

---

## Summary

1. **Start:** Empty target voxel [112, 112, 96] needs a value
2. **Step 1:** Convert to physical position in target scan → [-0.5, -0.5, -0.5]mm
3. **Step 2:** Map to same anatomy in reference scan → [10.2, -5.3, 8.7]mm  
4. **Step 3:** Convert to reference array index → [82.33, 56.79, 76.86]
5. **Step 4:** Interpolate reference coil map at that position → get coil sensitivity value
6. **Done:** Store value in target voxel [112, 112, 96]

### Coordinate System Conventions
- **MPS coordinates**: M=phase, P=readout, S=slice (Philips convention)
- **MATLAB order**: `[nx, ny, nz, ncoils]` (column-major)
- **Python order**: `[ncoils, nz, ny, nx]` (row-major)

