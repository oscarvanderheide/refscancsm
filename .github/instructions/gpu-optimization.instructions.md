---
description: "Use when modifying interpolation, FFT operations, or data transfer logic in the GPU/CPU critical path. Enforces GPU-first patterns and prevents premature data transfers that cause 10-15x performance degradation."
applyTo: ["**/interp.py", "**/get_csm.py"]
---

# GPU Optimization Guidelines

These files are in the **critical performance path**. Incorrect GPU/CPU data placement causes 10-15x slowdowns for FFT operations.

## Core Principle: Keep Data on GPU as Long as Possible

**Correct data flow**:
```
Interpolation (GPU) → FFT (GPU) → Transfer once → ESPIRiT (CPU)
```

**Incorrect data flow** (causes 10-15x slowdown):
```
Interpolation (GPU) → Transfer → FFT (CPU) ← SLOW!
```

## Rules for `interp.py`

### ✅ DO: Return GPU arrays when available

```python
# In interpolate_refscan_to_target_geometry()
result = xp.zeros((ncoils, nz, ny, nx), dtype=xp.complex64)
# ... perform interpolation ...
return result  # Keep on GPU - let caller decide when to transfer
```

### ❌ DON'T: Convert to NumPy at end of interpolation

```python
# BAD - loses GPU benefit for downstream FFT!
return cp.asnumpy(result) if use_gpu else result
```

**Why**: The result feeds directly into `fft3c()` which is GPU-accelerated. Transferring here forces FFT to run on CPU.

## Rules for `get_csm.py`

### ✅ DO: Transfer after FFT, before ESPIRiT

The **only correct transfer point** is lines 74-82:

```python
with timed("Converting coil images to k-space (3D FFT)"):
    kspace = fft3c(interpolated_coil_imgs)  # May be CuPy array
    # Convert back to CPU if on GPU (espirit expects NumPy)
    if hasattr(kspace, 'get'):  # CuPy array has .get() method
        kspace = kspace.get()

csm = espirit(kspace, ...)  # Requires NumPy array
```

**Why**: ESPIRiT uses NumPy-only operations (scipy.linalg, ThreadPoolExecutor) except for one internal GPU step. This transfer point maximizes GPU usage while maintaining compatibility.

### ❌ DON'T: Transfer before FFT

```python
# BAD - moves expensive FFT to CPU!
interpolated_coil_imgs = interpolate(...)
if hasattr(interpolated_coil_imgs, 'get'):
    interpolated_coil_imgs = interpolated_coil_imgs.get()
kspace = fft3c(interpolated_coil_imgs)  # Now runs on CPU (20+ seconds!)
```

## Checking Array Device

Use these patterns to detect CuPy arrays:

```python
# Method 1: Duck typing (preferred)
if hasattr(arr, 'get'):
    cpu_arr = arr.get()

# Method 2: Module check
if type(arr).__module__ == 'cupy':
    cpu_arr = arr.get()

# Method 3: Get array module
from .utils import cp, _xp
xp = _xp(arr)  # Returns cp or np based on array type
```

## Performance Benchmarks

Typical runtimes for 32-coil, 256³ volume:

| Operation | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| Interpolation | 5-8s | 40-50s | 5-10x |
| FFT (3D) | 1-2s | 15-20s | 10-15x |
| ESPIRiT | Mixed | Mixed | - |

**Impact of premature transfer**: FFT drops from 1.7s → 20s when forced to CPU.

## Common Mistakes

### Mistake 1: Assuming NumPy everywhere
```python
# BAD - fails with CuPy arrays
result = np.zeros_like(input_data)  # TypeError if input_data is CuPy
```

**Fix**: Use array-module-agnostic code:
```python
xp = _xp(input_data)
result = xp.zeros_like(input_data)  # Works with both
```

### Mistake 2: Implicit conversions
```python
# BAD - triggers implicit conversion error
cp_array = cp.zeros(...)
np_result = cp_array + 1  # TypeError in recent CuPy versions
```

**Fix**: Explicit conversion or keep on same device:
```python
np_result = cp_array.get() + 1  # or
cp_result = cp_array + 1
```

### Mistake 3: Multiple unnecessary transfers
```python
# BAD - transfers multiple times
result = interpolate(...).get()  # Transfer 1
fft_result = fft3c(result).get()  # Transfer 2 (fft already on CPU)
```

**Fix**: Single transfer at the right point (after FFT).

## Testing GPU Optimization Changes

After modifying these files:

1. **Verify GPU is used**:
   ```python
   # Should print: GPU detected: <device name>
   from refscancsm.utils import gpu_available
   print(f"GPU available: {gpu_available()}")
   ```

2. **Check FFT timing**: Look for this output:
   ```
   Converting coil images to k-space (3D FFT)...
     (1.7s)  ← Should be ~2s with GPU, 15-20s with CPU
   ```

3. **Monitor memory**: Watch for OOM errors:
   ```bash
   nvidia-smi  # Check GPU memory usage
   ```

## When to Deviate

**Only use CPU** when:
- User explicitly sets `CUDA_VISIBLE_DEVICES=""`
- `gpu_available()` returns `False`
- Data size causes OOM (>8GB for typical GPU)

**Never** force CPU for "simplicity" or "consistency" — the performance cost is too high.

## Related Files

- `utils.py` — Contains `_xp()`, `fft3c()`, `gpu_available()`
- `espirit.py` — Has internal GPU acceleration points (lines 229-232, 550-631)

## Quick Decision Tree

```
Modifying interpolation output?
├─ Data feeds into FFT? → Keep on GPU (return result as-is)
└─ Data for immediate output? → Transfer with .get()

Modifying get_csm workflow?
├─ Before FFT? → Keep on GPU
├─ After FFT, before ESPIRiT? → Transfer once with .get()
└─ After ESPIRiT? → Already NumPy

Adding new operation?
├─ Supported by CuPy? → Use xp = _xp(input) pattern
└─ NumPy-only? → Transfer input with .get() first
```
