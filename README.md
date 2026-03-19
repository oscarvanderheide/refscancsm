# refscancsm

Compute MRI coil sensitivity maps from Philips SENSE reference scans and interpolate them onto arbitrary target scan geometries using ESPIRiT.

**Required files** (export with Gyrotools' gtPacknGo):
- `refscan.cpx` (SENSE refscan images)
- `refscan.sin` (contains SENSE refscan geometry information)
- `target.sin` (contains target scan geometry information)

Powered by [PyTorch](https://pytorch.org): runs on **CPU**, **CUDA (NVIDIA GPU)**, and **MPS (Apple Silicon)** with a single code path — no backend switching required.  The ESPIRiT calibration step uses the [`espirit`](https://pypi.org/project/espirit/) PyPI package.

## Usage

### Command Line

```bash
# Auto-detect refscan files in same directory as target
get_csm target.sin

# Explicit refscan paths
get_csm target.sin --refscan-cpx ref.cpx --refscan-sin ref.sin

# Choose device explicitly
get_csm target.sin --device cuda
get_csm target.sin --device mps
get_csm target.sin --device cpu


# Verbose timing output
get_csm target.sin -v
```

Run with `uvx` without installing:
```bash
uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm target.sin
```

### Python API

```python
from refscancsm import get_csm
import numpy as np

# Basic usage — device auto-detected (CUDA > MPS > CPU)
csm = get_csm('target.sin')  # shape: (n_coils, nz, ny, nx), dtype: complex64

# Explicit device
csm = get_csm('target.sin', device='cuda')
csm = get_csm('target.sin', device='mps')   # Apple Silicon
csm = get_csm('target.sin', device='cpu')

np.save('csm.npy', csm)
```

## Installation

### CPU / MPS (Apple Silicon)

Standard PyPI install — PyTorch CPU/MPS wheel is pulled automatically:

```bash
uv add git+https://github.com/oscarvanderheide/refscancsm.git
# or
pip install git+https://github.com/oscarvanderheide/refscancsm.git
```

### CUDA (NVIDIA GPU)

Install the package first, then replace the CPU torch wheel with a CUDA-enabled one from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
uv add git+https://github.com/oscarvanderheide/refscancsm.git

# CUDA 12.x (adjust cu128 to match your CUDA version)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Check available CUDA version
nvidia-smi  # look for "CUDA Version" in the top-right corner
```

### Development

```bash
git clone https://github.com/oscarvanderheide/refscancsm.git
cd refscancsm
uv pip install -e .

# With dev dependencies (arrayview, jupyter)
uv pip install -e . --group dev
```

## Options

### Command Line

```bash
get_csm target.sin [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--refscan-cpx PATH` | Path to refscan .cpx file (auto-detected by default) |
| `--refscan-sin PATH` | Path to refscan .sin file (auto-detected by default) |
| `-o, --output PATH` | Output file (.npy or .mat, default: csm.npy) |
| `--calib-size N` | ESPIRiT calibration region size (default: 24) |
| `--kernel-size N` | ESPIRiT kernel size (default: 6) |
| `--threshold F` | Singular value threshold (default: 0.001) |
| `--device STR` | PyTorch device: cuda, mps, cpu (auto-detected by default) |
| `--force-cpu` | Force CPU (equivalent to `--device cpu`) |
| `-v, --verbose` | Show detailed timing information |

### Python API

```python
csm = get_csm(
    sin_path_target='target.sin',
    refscan_cpx_path=None,        # Auto-detected if None
    sin_path_refscan=None,        # Auto-detected if None
    # interpolation_order removed; always uses trilinear (order=1)
    calib_size=24,                # ESPIRiT calibration region
    kernel_size=6,                # ESPIRiT kernel size
    threshold=0.001,              # Singular value threshold
    device=None,                  # torch device: 'cuda', 'mps', 'cpu', or None
    force_cpu=False,              # Shorthand for device='cpu'
    verbose=False,                # Show timing info
)
# Returns: numpy ndarray, shape (n_coils, nz, ny, nx), dtype complex64
```

## Architecture

### Data Pipeline

```
Read .cpx (numpy)
    ↓  to_device()
Interpolate → target geometry   (torch.nn.functional.grid_sample, all devices)
    ↓
3D centered FFT                 (torch.fft, all devices)
    ↓
ESPIRiT calibration             (external espirit package, all devices)
    ↓  .cpu().numpy()
Return ndarray
```

### Interpolation

Only trilinear interpolation (order=1) is supported. This uses `torch.nn.functional.grid_sample` with `mode='bilinear'` on all devices (CUDA, MPS, CPU).

## Troubleshooting

**Device not detected:**
```python
import torch
print(torch.cuda.is_available())          # CUDA
print(torch.backends.mps.is_available())  # MPS (macOS)
```

**CUDA out of memory:** Pass `device='cpu'` or reduce `calib_size`.

**Import errors:** Reinstall with `uv pip install -e .`

**Cubic interpolation warning:** Order 3 always uses scipy on CPU. Use order 1 for GPU-accelerated interpolation.
