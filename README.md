# RefScanCSM

A Python package for reading CPX files from a Philips SENSE refscan and interpolating coil sensitivity maps onto target scan geometries. This tool provides both a command-line interface and a Python library for working with MRI coil survey data.

## Features

- Read CPX files with automatic case-insensitive filename handling
- Extract location matrices and transformation data from SIN files
- Transform coordinate systems (MPS refscan and target transformations)
- Read voxel sizes and matrix dimensions
- **Interpolate coil sensitivity maps from reference to target geometry**
- Handle different scan planning (rotation, translation, voxel sizes)
- Command-line tool (`get_csm`) for easy coil map interpolation
- Python API for programmatic access
- Output to NumPy (.npy) or MATLAB (.mat) formats

## Installation

### From GitHub (recommended for development)

```bash
# Install with uv
uv add git+https://github.com/yourusername/refscancsm.git

# Or using pip
pip install git+https://github.com/yourusername/refscancsm.git
```

### Direct usage with uvx (no installation needed)

```bash
uvx --from git+https://github.com/yourusername/refscancsm.git get_csm <file.cpx>
```

### Local development installation

```bash
# Clone the repository
git clone https://github.com/yourusername/refscancsm.git
cd refscancsm

# Install in editable mode
uv pip install -e .
```

## Usage

### Command-line Interface

**Main use case: Interpolate coil maps from reference scan to target scan geometry**

```bash
# Basic usage - interpolate and save to .npy
get_csm refscan.cpx refscan.sin target.sin

# Specify output file
get_csm refscan.cpx refscan.sin target.sin -o my_coil_maps.npy

# Save as MATLAB file
get_csm refscan.cpx refscan.sin target.sin -o coil_maps.mat

# Use cubic interpolation (default is linear)
get_csm refscan.cpx refscan.sin target.sin --interp-order 3

# Show detailed information
get_csm refscan.cpx refscan.sin target.sin --verbose

# Get help
get_csm --help
```

**Arguments:**
- `refscan_cpx`: Path to reference scan CPX file (coil survey data)
- `refscan_sin`: Path to reference scan SIN file (geometry info)
- `target_sin`: Path to target scan SIN file (target geometry)

**Options:**
- `-o, --output`: Output file path (default: csm.npy)
- `--interp-order {0,1,3}`: Interpolation order (0=nearest, 1=linear, 3=cubic)
- `--location-idx`: Location index from SIN files (default: 1)
- `-v, --verbose`: Show detailed information

### Python API

**Full workflow: Interpolate coil maps to target geometry**

```python
from refscancsm import get_csm
import numpy as np

# Interpolate coil sensitivity maps from reference to target geometry
coil_maps = get_csm(
    refscan_cpx_path="path/to/refscan",  # without .cpx extension
    sin_path_refscan="path/to/refscan.sin",
    sin_path_target="path/to/target.sin",
    location_idx=1,
    interpolation_order=1,  # 0=nearest, 1=linear, 3=cubic
    verbose=True,
)

print(f"Interpolated coil maps shape: {coil_maps.shape}")  # [ncoils, nz, ny, nx]
print(f"Number of coils: {coil_maps.shape[0]}")

# Save the result
np.save("coil_maps.npy", coil_maps)
```

**Lower-level API: Read individual files**

```python
from refscancsm import read_cpx, get_voxel_sizes, get_matrix_size
from refscancsm import get_mps_to_xyz_transform, get_idx_to_mps_transform
import numpy as np

# Read a CPX file
data, hdr, labels = read_cpx("path/to/file")
print(f"Data shape: {data.shape}")
print(f"Labels: {labels}")

# Read voxel sizes and matrix dimensions from SIN file
voxel_sizes = get_voxel_sizes("path/to/file.sin")
matrix_size = get_matrix_size("path/to/file.sin")
print(f"Voxel sizes: {voxel_sizes} mm")
print(f"Matrix size: {matrix_size}")

# Get transformation matrices
idx_to_mps = get_idx_to_mps_transform("path/to/file.sin")
mps_to_xyz = get_mps_to_xyz_transform("path/to/file.sin", "source", location_idx=1)
idx_to_xyz = mps_to_xyz @ idx_to_mps
print(f"Index to world transformation:\n{idx_to_xyz}")
```

## API Reference

### Main Interpolation Function

- **`get_csm(refscan_cpx_path, sin_path_refscan, sin_path_target, location_idx=1, interpolation_order=1, verbose=True)`**
  - Complete workflow to interpolate coil maps from reference to target geometry
  - Returns: numpy array [ncoils, nz, ny, nx] in target geometry

### CPX Reading Functions

- **`read_cpx(filepath)`**: Read and parse a CPX file
  - Returns: `(data, hdr, data_labels)` tuple
  - `data`: numpy array with shape `[nchan, nmix, ndyn, ncard, necho, nrow, nloc, nslice,
    ny, nx]`
  - `hdr`: Dictionary containing header information
  - `data_labels`: Array of dimension labels

### SIN File Reading Functions

- **`get_voxel_sizes(sin_file_path)`**: Extract voxel sizes
  - Returns: 1D array `[x, y, z]` in mm

- **`get_matrix_size(sin_file_path)`**: Extract matrix dimensions
  - Returns: 1D array `[x, y, z]`

### Transformation Functions

- **`get_idx_to_mps_transform(sin_file_path)`**: Create 4x4 matrix that converts array indices to MPS coordinates
  - Returns: 4x4 transformation matrix

- **`get_mps_to_xyz_transform(sin_file_path, scan_type, location_idx=1)`**: Get transformation from MPS to world coordinates
  - `scan_type`: Either "source" or "target"
  - Returns: 4x4 transformation matrix

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
uv run get_csm refscan.cpx refscan.sin target.sin
```

## License

MIT License

## Notes

The code is functional but may benefit from future cleanup, especially the `read_cpx`
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

