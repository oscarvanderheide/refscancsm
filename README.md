# refscancsm

Compute MRI coil sensitivity maps from Philips SENSE reference scans and interpolate them onto arbitrary target scan geometries using ESPIRiT.

**Required files**: `.cpx` (SENSE refscan images), `.sin` (SENSE refscan geometry), `.sin` (target scan geometry). Export with Gyrotools' gtPacknGo.

The ESPIRiT implementation is a GPU-compatible Python translation from the [BART Toolbox](https://codeberg.org/mrirecon/bart) (Uecker et al., MRM 2014).

## Usage

### Command Line

```bash
# Auto-detects refscan files (*senserefscan*.{cpx,sin}) in same directory
uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm target.sin
```

Create an alias for convenience:
```bash
alias get_csm='uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm'
get_csm /path/to/target.sin
```

### Python API

```python
from refscancsm import get_csm

# Auto-detects refscan files in same directory as target
csm = get_csm('target.sin')  # shape: (n_coils, nz, ny, nx), dtype: complex64
```

## Installation

### GPU Acceleration (Linux/Windows only)

For CUDA 11.x:
```bash
uv add git+https://github.com/oscarvanderheide/refscancsm.git --extra cuda11x
```

For CUDA 12.x:
```bash
uv add git+https://github.com/oscarvanderheide/refscancsm.git --extra cuda12x
```

### CPU Only (macOS or no GPU)

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