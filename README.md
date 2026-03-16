# RefScanCSM

A Python package for computing MRI coil sensitivity maps from Philips SENSE reference scans and interpolating them onto arbitrary target scan geometries using ESPIRiT.

**Required files**: `.cpx` of SENSE refscan, `.sin` of SENSE refscan and `.sin` of target scan. Can all be exported with Gyrotools' gtPacknGo. 

**Steps performed**: 
1) Read in SENSE refscan images from `.cpx`
2) Read in information about refscan and target scan geometries from `.sin` files
3) Interpolate SENSE refscan images to target geometry
4) FFT interpolated images to k-space data
5) Apply ESPIRiT on k-space data

The ESPIRiT algorithm in this package is a GPU-compatible Python translation of ESPIRiT (Uecker et al., MRM 2014) from the [BART Toolbox](https://codeberg.org/mrirecon/bart). It should become a standalone package.

## Quick Start

### Command Line (no installation needed)

```bash
# Most common usage - auto-detects refscan.{cpx,sin} files in same directory
uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm target.sin
```

**Recommended**: Create an alias for convenience:
```bash
alias get_csm='uvx --from git+https://github.com/oscarvanderheide/refscancsm.git get_csm'
```

Then use it anywhere:
```bash
get_csm /path/to/target.sin
```

### Python API

Add `refscancsm` to your project:
```bash

uv add git+https://github.com/oscarvanderheide/refscancsm.git
```


```python
from refscancsm import get_csm

# Simplest usage - auto-detects refscan files
csm = get_csm('target.sin')

# Result shape: (n_coils, nz, ny, nx), dtype: complex64
print(f"Coil sensitivity maps: {csm.shape}")
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