# RefScanCSM — Workspace Instructions

Python package for computing MRI coil sensitivity maps from Philips SENSE reference scans and interpolating them onto arbitrary target scan geometries using ESPIRiT.

## Project Overview

**Purpose**: Transform Philips coil survey data (from `.cpx`/`.sin` files) into coil sensitivity maps aligned to target scan geometry, enabling parallel imaging reconstruction across different scan orientations/resolutions.

**Key Technology**: GPU-accelerated (CuPy) with CPU fallback, transparent array backend dispatch.

## Architecture

### Core Modules

```
src/refscancsm/
├── cli.py              # Command-line interface (get_csm)
├── get_csm.py          # Main workflow orchestration
├── parse_cpx.py        # Read Philips CPX binary files
├── parse_sin.py        # Extract geometry/transforms from SIN files
├── interp.py           # Spatial interpolation (GPU/CPU)
├── espirit.py          # ESPIRiT algorithm implementation
├── walsh.py            # Alternative Walsh method (unused)
└── utils.py            # Backend selection, FFT, GPU detection
```

### Data Pipeline

```
Read .cpx → Flip/Rotate → Interpolate → 3D FFT → ESPIRiT → CSM
  (CPU)       (CPU)      (GPU*/CPU)   (GPU*/CPU)  (Mixed)   (NumPy)
                                                    ↑
                                    GPU→CPU transfer point
```

**Critical GPU→CPU Transfer**: Occurs in `get_csm.py` after FFT before ESPIRiT.

### Module Dependencies

- `cli.py` → `get_csm.py`
  - → `parse_cpx.py` (read_cpx)
  - → `parse_sin.py` (geometry transforms)
  - → `interp.py` (spatial resampling)
  - → `utils.py` (FFT, GPU detection)
  - → `espirit.py` (coil sensitivity calibration)

## Development Setup

### Installation

```bash
# Development mode
git clone https://github.com/oscarvanderheide/refscancsm.git
cd refscancsm
uv pip install -e .

# With dev dependencies (arrayview, jupyter)
uv pip install -e . --group dev
```

### Dependencies

- **cupy-cuda11x** ≥13.6.0 — GPU acceleration (optional)
- **numpy** ≥2.4.1 — Array operations
- **scipy** ≥1.11.0 — Interpolation, linalg
- **tqdm** ≥4.67.3 — Progress bars

### Testing

**No formal test suite** — use manual testing:

```bash
# Import test
python -c "from refscancsm import get_csm; print('OK')"

# Functional test (requires real data)
python test.py  # Jupyter-style script with #%% cells
```

**Debugging**: See `debug/` directory for step-by-step validation scripts and reference `.npy` outputs.

## Guidelines

### GPU/CPU Backend Strategy

**Auto-detection pattern**:
```python
from .utils import gpu_available, cp

if gpu_available():
    # Use CuPy arrays
    xp = cp
else:
    # Use NumPy arrays
    xp = np
```

**Array-agnostic operations**: All FFT/array functions use dispatch:
```python
def _xp(x):
    """Return cp or np based on array type."""
    return cp.get_array_module(x) if cp is not None else np

def fft3c(x):
    xp = _xp(x)
    return xp.fft.fftshift(xp.fft.fftn(...), ...)
```

### Data Transfer Guidelines

**Keep data on GPU as long as possible**:

✅ **Good** — minimize transfers:
```python
# In interp.py
result = interpolate_on_gpu(...)
return result  # Keep on GPU

# In get_csm.py
interpolated = interpolate(...)  # GPU array
kspace = fft3c(interpolated)     # Stays on GPU
if hasattr(kspace, 'get'):
    kspace = kspace.get()         # Transfer once
csm = espirit(kspace)             # CPU
```

❌ **Bad** — premature transfer:
```python
# In interp.py
result = interpolate_on_gpu(...)
return cp.asnumpy(result)  # Transfers too early!

# In get_csm.py
interpolated = interpolate(...)  # Now NumPy
kspace = fft3c(interpolated)     # Runs on CPU (slow!)
```

**Performance impact**: FFT on CPU is ~10-15x slower than GPU for typical data sizes.

### Critical Transfer Point

**Location**: `src/refscancsm/get_csm.py:74-82`

```python
with timed("Converting coil images to k-space (3D FFT)"):
    kspace = fft3c(interpolated_coil_imgs)
    # Convert back to CPU if on GPU (espirit expects NumPy)
    if hasattr(kspace, 'get'):  # CuPy array has .get() method
        kspace = kspace.get()

csm = espirit(kspace, ...)  # Requires NumPy array
```

**Why**: ESPIRiT uses NumPy-only operations (scipy.linalg, ThreadPoolExecutor) except for one GPU-accelerated step internally.

### Complex Array Handling

**Interpolation workaround** (`interp.py`):
```python
# map_coordinates doesn't support complex dtype
result[coil] = map_fn(data.real, coords, ...) + \
               1j * map_fn(data.imag, coords, ...)
```

Always split real/imaginary when using scipy/cupyx interpolation.

### Coordinate System Conventions

1. **Array shapes**: `(n_coils, nz, ny, nx)` — Philips convention (x=freq, y=phase, z=slice)
2. **FFT axes**: `(-3, -2, -1)` = (z, y, x)
3. **Interpolation coords**: Must be in `[Z, Y, X]` order (reverse of shape)
4. **Transforms**: Chain `target_idx → MPS → xyz → MPS → refscan_idx`

### Common Patterns

#### Checking Array Device
```python
# Method 1: Duck typing
if hasattr(arr, 'get'):
    cpu_arr = arr.get()

# Method 2: Module check
if type(arr).__module__ == 'cupy':
    cpu_arr = arr.get()

# Method 3: get_array_module
xp = cp.get_array_module(arr) if cp else np
```

#### Moving Between Devices
```python
# NumPy → CuPy
gpu_arr = cp.asarray(np_arr)

# CuPy → NumPy
np_arr = gpu_arr.get()  # or cp.asnumpy(gpu_arr)
```

#### Centered FFTs
```python
# Always use fft3c/ifft3c from utils.py
kspace = fft3c(image)   # Applies fftshift for k-space center
image = ifft3c(kspace)  # Applies ifftshift
```

## ESPIRiT Algorithm Details

**Pipeline** (based on Uecker et al., MRM 2014):
1. Extract calibration region (ACS) from k-space center
2. Build Casorati matrix from overlapping patches
3. Eigendecomposition of Gram matrix → signal-space kernels
4. IFFT kernels to image domain (PSF-like basis)
5. Compute per-voxel covariance matrices (H^H H)
6. Sinc-interpolate covariance to full resolution
7. Extract dominant eigenvector per voxel → sensitivity map
8. Phase-align and normalize

**GPU acceleration points**:
- Gram matrix eigendecomposition (`espirit.py:229-232`)
- 3D sinc interpolation + eigenmap extraction (`espirit.py:550-631`)

**CPU fallback**: 2D ESPIRiT always uses CPU.

## File Formats

**Input**:
- `.cpx` — Philips complex coil images (binary, compressed)
- `.sin` — Philips scan metadata (text, geometry/transforms)

**Output**:
- `.npy` — NumPy array format (default)
- `.mat` — MATLAB v7.3 format (use `-o file.mat`)

**Shape**: `(n_coils, nz, ny, nx)` — complex64

## Performance Considerations

**Bottlenecks** (typical runtime %):
1. Interpolation: 60-70% (GPU speedup: 5-10x)
2. ESPIRiT: 20-30% (mixed GPU/CPU)
3. FFT: 5-10% (GPU speedup: 10-15x)
4. I/O: <5%

**Memory**:
- Typical: 2-4 GB GPU (32 coils, 256³ volume)
- Watch for OOM on 512³ targets

**Thread control**:
```python
from refscancsm.utils import set_num_threads
set_num_threads(8)  # Override auto-detection
```

## Usage Examples

### Command Line
```bash
# Auto-detect refscan files
get_csm target.sin

# Explicit paths
get_csm target.sin --refscan-cpx ref.cpx --refscan-sin ref.sin

# Cubic interpolation
get_csm target.sin --interp-order 3 -o csm.npy

# Verbose timing
get_csm target.sin -v
```

### Python API
```python
from refscancsm import get_csm
import numpy as np

# Basic usage
csm = get_csm('target.sin')

# Full control
csm = get_csm(
    sin_path_target='target.sin',
    refscan_cpx_path='ref.cpx',
    sin_path_refscan='ref.sin',
    interpolation_order=1,     # 0=nearest, 1=linear, 3=cubic
    calib_size=24,             # ESPIRiT calibration region
    kernel_size=6,             # ESPIRiT kernel size
    threshold=0.001            # Singular value threshold
)

# Shape: (n_coils, nz, ny, nx), dtype: complex64
np.save('csm.npy', csm)
```

## Anti-Patterns to Avoid

❌ **Converting to NumPy too early**
```python
# In interpolation
return cp.asnumpy(result)  # Bad! Loses GPU benefit for FFT
```

❌ **Manual FFT instead of centered FFT**
```python
kspace = np.fft.fftn(img)  # Bad! K-space center at corners
```

❌ **Ignoring device context**
```python
result = cp.zeros(...)
np_result = result + 1  # Bad! Implicit conversion error
```

❌ **Threading GPU operations**
```python
with ThreadPoolExecutor() as exec:
    exec.map(gpu_func, cupy_arrays)  # Bad! GPU context not thread-safe
```

## Related Projects

- **BART** (`bart/`) — Reference implementation in C
- **Julia** (`julia/`) — Alternative implementation for validation
- **arrayview** — Interactive 4D array visualization (dev dependency)

## Troubleshooting

**GPU not detected**:
- Check: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`
- Verify: CUDA toolkit matches CuPy version (cuda11x/cuda12x)

**OOM errors**:
- Reduce batch size or use CPU fallback
- Check `nvidia-smi` for memory leaks

**Slow FFT performance**:
- Verify data is on GPU: `type(arr).__module__ == 'cupy'`
- Check transfer point in `get_csm.py:74-82`

**Import errors**:
- Reinstall: `uv pip install -e .`
- Check Python ≥3.12

## Version Control

**Ignored files**: `.cpx`, `.sin`, `.npy`, `.mat`, `debug/`, `bart/` (external)

**Key branches**: Development on `main` branch
