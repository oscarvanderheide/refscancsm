# refscancsm

Compute MRI coil sensitivity maps from Philips SENSE reference scans and interpolate them onto arbitrary target scan geometries using ESPIRiT.

**Required files** (export with Gyrotools' gtPacknGo): 
- `refscan.cpx` (SENSE refscan images)
- `refscan.sin` (contains SENSE refscan geometry information)
- `target.sin` (contains target scan geometry information)

The ESPIRiT implementation is a GPU-compatible Python translation from the [BART Toolbox](https://codeberg.org/mrirecon/bart) (Uecker et al., MRM 2014).

**Note**: Not intended for PyPI distribution. Install directly from GitHub.

## Usage

### Command Line

CPU-only (works on all platforms):
```bash
uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm target.sin
```

With GPU acceleration (see [Installation](#installation) to determine your CUDA version):
```bash
# CUDA 11.x
uvx --with cupy-cuda11x --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm target.sin

# CUDA 12.x
uvx --with cupy-cuda12x --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm target.sin
```

Create an alias for convenience:
```bash
# CPU-only
alias get_csm='uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm'

# With GPU (CUDA 11.x)
alias get_csm='uvx --with cupy-cuda11x --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm'

# Then use anywhere
get_csm /path/to/target.sin
```

### Python API

```python
from refscancsm import get_csm

# Auto-detects refscan files in same directory as target
csm = get_csm('target.sin')  # shape: (n_coils, nz, ny, nx), dtype: complex64
```

## Installation

### Determining Your CUDA Version

**If you have a GPU and want acceleration**, first check your CUDA version:

```bash
nvidia-smi
```

Look for "CUDA Version" in the top-right corner (e.g., "CUDA Version: 11.8" or "CUDA Version: 12.1").

- CUDA 11.x → use `--extra cuda11x` (or `--with cupy-cuda11x` for uvx)
- CUDA 12.x → use `--extra cuda12x` (or `--with cupy-cuda12x` for uvx)
- No CUDA or macOS → omit the extra (CPU-only mode)

If `nvidia-smi` is not found, you either don't have a GPU, don't have CUDA installed, or you're on macOS. Use CPU-only mode.

### Python API Installation

For CUDA 11.x:
```bash
uv add git+https://github.com/oscarvanderheide/refscancsm.git --extra cuda11x
```

For CUDA 12.x:
```bash
uv add git+https://github.com/oscarvanderheide/refscancsm.git --extra cuda12x
```

CPU-only (macOS or no GPU):
```bash
uv add git+https://github.com/oscarvanderheide/refscancsm.git
```

The tool automatically detects GPU availability and falls back to CPU if needed.

## Options

### Command Line

```bash
get_csm target.sin [OPTIONS]
```

- `--refscan-cpx PATH` - Path to refscan .cpx file (auto-detected by default)
- `--refscan-sin PATH` - Path to refscan .sin file (auto-detected by default)
- `-o, --output PATH` - Output file (.npy or .mat, default: csm.npy)
- `--interp-order N` - Interpolation: 0=nearest, 1=linear, 3=cubic (default: 1)
- `--calib-size N` - ESPIRiT calibration region size (default: 24)
- `--kernel-size N` - ESPIRiT kernel size (default: 6)
- `--threshold F` - Singular value threshold (default: 0.001)
- `--force-cpu` - Force CPU even when GPU available
- `-v, --verbose` - Show detailed timing information

### Python API

```python
csm = get_csm(
    sin_path_target='target.sin',
    refscan_cpx_path=None,        # Auto-detected if None
    sin_path_refscan=None,        # Auto-detected if None
    interpolation_order=1,        # 0=nearest, 1=linear, 3=cubic
    calib_size=24,                # ESPIRiT calibration region
    kernel_size=6,                # ESPIRiT kernel size
    threshold=0.001,              # Singular value threshold
    force_cpu=False,              # Force CPU
    verbose=False,                # Show timing info
)
```