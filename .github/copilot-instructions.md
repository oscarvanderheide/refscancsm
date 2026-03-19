# RefScanCSM — Workspace Instructions

Python package for computing MRI coil sensitivity maps from Philips SENSE reference scans and interpolating them onto arbitrary target scan geometries using ESPIRiT.

## Project Overview

**Purpose**: Transform Philips coil survey data (from `.cpx`/`.sin` files) into coil sensitivity maps aligned to target scan geometry, enabling parallel imaging reconstruction across different scan orientations/resolutions.

**Key Technology**: GPU-accelerated via **PyTorch** — single code path for CUDA, MPS (Apple Silicon), and CPU.  No CuPy dependency.  ESPIRiT is provided by the external [`espirit`](https://pypi.org/project/espirit/) PyPI package.

## Architecture

### Core Modules

```
src/refscancsm/
├── cli.py              # Command-line interface (get_csm)
├── get_csm.py          # Main workflow orchestration
├── parse_cpx.py        # Read Philips CPX binary files
├── parse_sin.py        # Extract geometry/transforms from SIN files
├── interp.py           # Spatial interpolation (torch.nn.functional.grid_sample)
├── walsh.py            # Alternative Walsh method (unused)
└── utils.py            # Device selection, FFT helpers (torch.fft)
```

### Data Pipeline

```
Read .cpx (numpy)
    ↓  torch.from_numpy()  +  .to(device)
Interpolate → target geometry   [torch.nn.functional.grid_sample, all devices]
    ↓
3D centered FFT                 [torch.fft, all devices]
    ↓
ESPIRiT (external package)      [all devices, passed via device= param]
    ↓  tensor.cpu().numpy()
Return numpy ndarray
```

### Module Dependencies

- `cli.py` → `get_csm.py`
  - → `parse_cpx.py` (read_cpx, returns numpy)
  - → `parse_sin.py` (geometry transforms, returns numpy)
  - → `interp.py` (spatial resampling, returns torch.Tensor)
  - → `utils.py` (fft3c, get_device)
  - → external `espirit` package (coil sensitivity calibration)

## Development Setup

### Installation

```bash
git clone https://github.com/oscarvanderheide/refscancsm.git
cd refscancsm
uv pip install -e .

# For CUDA (replace cu128 with your CUDA version)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# With dev dependencies
uv pip install -e . --group dev
```

### Dependencies

- **torch** ≥2.0.0 — All compute (CUDA/MPS/CPU)
- **espirit** ≥0.1.0 — ESPIRiT calibration (PyPI, PyTorch-based)
- **numpy** ≥2.4.1 — Array I/O, geometry transforms
- **scipy** ≥1.11.0 — Order-3 interpolation fallback, savemat
- **tqdm** ≥4.67.1 — Progress bars

### Testing

```bash
# Import test
python -c "from refscancsm import get_csm, get_device; print(get_device())"

# Device comparison test (requires real data)
python test_cpu_gpu_comparison.py target.sin

# Functional test (Jupyter-style)
python test.py
```

## Guidelines

### Device Strategy

**`get_device()` auto-detects the best device** — CUDA > MPS > CPU:

```python
from .utils import get_device

device = get_device()          # torch.device('cuda') | torch.device('mps') | torch.device('cpu')
tensor = data.to(device)       # Move any tensor to best device
```

**Pass `device` explicitly through every function** — do not rely on globals:

```python
# In interp.py
def interpolate_refscan_to_target_geometry(refscan_imgs, ..., device=None):
    if device is None:
        device = get_device()
    refscan_data = torch.from_numpy(refscan_imgs).to(device)
    ...

# In get_csm.py
interpolated = interpolate_refscan_to_target_geometry(..., device=device)
kspace = fft3c(interpolated)           # stays on device
csm = espirit(kspace, device=device)   # stays on same device
return csm.cpu().numpy()               # single transfer at the end
```

### Interpolation

**Orders 0 and 1** use `torch.nn.functional.grid_sample` (all devices, no Python loop):

```python
# Input: (1, n_coils, nz_ref, ny_ref, nx_ref)
# Grid:  (1, nz, ny, nx, 3) — last dim = (x_norm, y_norm, z_norm) in [-1, 1]
# align_corners=True: pixel 0 → -1, pixel N-1 → +1
out = F.grid_sample(real_vol, grid, mode='bilinear', align_corners=True, padding_mode='zeros')
```

**Order 3** falls back to `scipy.ndimage.map_coordinates` on CPU with a `UserWarning`.
Result is moved back to the requested device afterwards.

### FFT

Always use `fft3c` / `ifft3c` / `ifft2c` from `utils.py` — these use `torch.fft` and
work transparently on all devices:

```python
from .utils import fft3c
kspace = fft3c(image_tensor)   # Works on CUDA, MPS, CPU
```

### Return Type

`get_csm` always returns a `numpy.ndarray` (shape `(n_coils, nz, ny, nx)`, dtype `complex64`).
Convert to torch only when needed downstream:

```python
csm_np = get_csm('target.sin')          # ndarray
csm_t  = torch.from_numpy(csm_np)      # tensor if needed
```

## Common Mistakes to Avoid

❌ **Importing or using CuPy** — CuPy has been removed; all compute goes through PyTorch.

❌ **Returning numpy from interp.py** — `interpolate_refscan_to_target_geometry` must return a `torch.Tensor` on the requested device so the downstream FFT stays on device.

❌ **Calling `espirit()` without passing `device=`** — espirit auto-detects CUDA if available; if the user passed `force_cpu=True` or `device='cpu'`, you must propagate that:
```python
csm = espirit(kspace, ..., device=device)
```

❌ **Multiple CPU↔device transfers** — the only transfer should be the final `.cpu().numpy()` at the end of `get_csm`.

## File Formats

**Input**: `.cpx` (Philips binary), `.sin` (Philips text metadata)
**Output**: `.npy` (default), `.mat` (MATLAB via CLI `-o file.mat`)
**Shape**: `(n_coils, nz, ny, nx)`, `complex64`

## Version Control

**Ignored files**: `.cpx`, `.sin`, `.npy`, `.mat`, `debug/`, `bart/`
**Key branches**: `main`
