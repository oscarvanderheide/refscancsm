# RefScanCSM

A Python package for computing MRI coil sensitivity maps from Philips SENSE reference scans and interpolating them onto arbitrary target scan geometries using ESPIRiT. Handles different scan orientations, resolutions, and positions automatically.

## Features

- **Automatic refscan detection** — just provide the target .sin file
- GPU-accelerated interpolation and FFT (10-15x faster with CUDA)
- ESPIRiT-based coil sensitivity calibration
- Handle different scan planning (rotation, translation, voxel sizes)
- Command-line tool (`get_csm`) and Python API
- Output to NumPy (.npy) or MATLAB (.mat) formats

## Quick Start

### Command Line (no installation needed)

```bash
# Most common usage - auto-detects refscan files in same directory
uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm target.sin

# With custom output path
uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm target.sin -o coil_maps.npy
```

**Recommended**: Create an alias for convenience:
```bash
alias get_csm='uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm'
```

Then use it anywhere:
```bash
get_csm /path/to/target.sin
get_csm target.sin -o my_coil_maps.mat
```

### Python API

```python
from refscancsm import get_csm

# Simplest usage - auto-detects refscan files
csm = get_csm('target.sin')

# Result shape: (n_coils, nz, ny, nx), dtype: complex64
print(f"Coil sensitivity maps: {csm.shape}")
```

## Installation

### For Projects (with uv)

```bash
# Add to your project
uv add git+https://github.com/oscarvanderheide/refscancsm.git
```

### For Development

```bash
# Clone and install in editable mode
git clone https://github.com/oscarvanderheide/refscancsm.git
cd refscancsm
uv pip install -e .
```

## Usage

### Command-line Interface

**Basic usage** (auto-detects `*senserefscan*.cpx` and `*senserefscan*.sin` in same directory):
```bash
get_csm target.sin
```

**Explicit paths** (when multiple refscan files exist):
```bash
get_csm target.sin --refscan-cpx refscan.cpx --refscan-sin refscan.sin
```

**Options**:
```bash
get_csm target.sin \
  -o coil_maps.npy \          # Output path (.npy or .mat)
  --interp-order 3 \           # 0=nearest, 1=linear (default), 3=cubic
  --calib-size 24 \            # ESPIRiT calibration region size
  --kernel-size 6 \            # ESPIRiT kernel size
  --threshold 0.001 \          # Singular value threshold
  --force-cpu \                # Force CPU even when GPU available
  -v                           # Verbose timing output
```

**Get help**:
```bash
get_csm --help
```

### Python API

**Basic usage** (auto-detects refscan files):
```python
from refscancsm import get_csm
import numpy as np

# Auto-detect refscan files in same directory as target
csm = get_csm('target.sin')

# Result: (n_coils, nz, ny, nx), complex64
print(f"Shape: {csm.shape}")
np.save('coil_maps.npy', csm)
```

**Full control**:
```python
csm = get_csm(
    sin_path_target='target.sin',
    refscan_cpx_path='refscan.cpx',     # Optional - auto-detected if None
    sin_path_refscan='refscan.sin',     # Optional - auto-detected if None
    interpolation_order=1,               # 0=nearest, 1=linear, 3=cubic
    calib_size=24,                       # ESPIRiT calibration region
    kernel_size=6,                       # ESPIRiT kernel size
    threshold=0.001,                     # Singular value threshold
    force_cpu=False,                     # Force CPU even when GPU available
)
```

## Requirements

- Python ≥ 3.12
- numpy ≥ 2.4.1
- scipy ≥ 1.11.0
- tqdm ≥ 4.67.3
- cupy-cuda11x ≥ 13.6.0 (optional, for GPU acceleration)

## Performance

**GPU acceleration** (NVIDIA CUDA):
- Interpolation: 5-10x faster (5-8s vs 40-50s)
- FFT operations: 10-15x faster (1-2s vs 15-20s)
- Overall speedup: ~8-10x for typical datasets

**Typical runtime** (32 coils, 256³ volume):
- With GPU: ~15-20 seconds
- CPU only: ~2-3 minutes

**Note**: GPU acceleration is **highly recommended** for production use. CPU fallback is available but has not been extensively tested on large datasets.

To force CPU usage even when GPU is available:
```bash
get_csm target.sin --force-cpu
```

Or in Python:
```python
csm = get_csm('target.sin', force_cpu=True)
```

## Algorithm

This package uses **ESPIRiT** (Uecker et al., MRM 2014) for coil sensitivity estimation:

1. Extract calibration region from k-space center
2. Build calibration matrix from overlapping patches
3. Eigendecomposition of Gram matrix → signal-space kernels
4. Transform kernels to image domain
5. Compute per-voxel covariance matrices
6. Sinc-interpolate and extract dominant eigenvector
7. Phase-align and normalize sensitivity maps

The spatial interpolation handles:
- Different voxel sizes between scans
- Different matrix dimensions
- Different scan planning (rotation, translation, angulation)
- Proper centering conventions

**Interpolation methods**:
- **Linear (order=1, default)**: Best for masked data, avoids oscillations
- **Nearest (order=0)**: Fastest but blocky
- **Cubic (order=3)**: Smoothest but may overshoot at mask edges

## Troubleshooting

**GPU not detected**:
```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```
Verify CUDA toolkit matches CuPy version (cuda11x/cuda12x).

**Slow performance** (20+ seconds for FFT):
Check that data stays on GPU through FFT. See `.github/instructions/gpu-optimization.instructions.md` for details.

**Out of memory**:
- Use CPU fallback: `CUDA_VISIBLE_DEVICES="" get_csm target.sin`
- Reduce matrix size or use lower interpolation order


# Compare CPU vs GPU outputs
python test_cpu_gpu_comparison.py <target.sin>
```

See [TEST_GUIDE.md](TEST_GUIDE.md) for detailed testing instructions.Development

```bash
# Install with dev dependencies
git clone https://github.com/oscarvanderheide/refscancsm.git
cd refscancsm
uv pip install -e . --group dev

# Run tests (manual, no automated suite)
python test.py
```

**Development notes**:
- GPU optimization: Current implementation transfers data to CPU after FFT for ESPIRiT. Potential for further optimization by keeping more operations on GPU.
- CPU fallback: Available but not extensively tested on large datasets. Contributions welcome for validation.

## License

MIT License

## Contributing

Contributions welcome! Areas for improvement:
- [ ] Automated test suite
- [ ] More extensive CPU fallback testing
- [ ] Additional GPU optimization in ESPIRiT pipeline
- [ ] 2D scan support
- [ ] Batch processing for multiple targets

## References

- Uecker M, et al. ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA. *Magnetic Resonance in Medicine* 71.3 (2014): 990-1001.
- BART Toolbox: https://mrirecon.codeberg.page/
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

