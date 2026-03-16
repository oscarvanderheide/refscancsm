"""Shared utilities: GPU/CPU backend selection, thread count, timing, and FFT helpers."""

import os
import time
from contextlib import contextmanager
from functools import lru_cache

import numpy as np


@contextmanager
def timed(label):
    """Context manager that prints a step label, runs the block, then prints elapsed time."""
    print(f"  {label}...")
    t0 = time.perf_counter()
    yield
    print(f"    ({time.perf_counter() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# GPU backend (CuPy)
# ---------------------------------------------------------------------------

try:
    import cupy as cp
except Exception:
    cp = None

_FORCE_CPU = False  # Global flag to force CPU usage


def set_force_cpu(force: bool) -> None:
    """Set whether to force CPU usage even when GPU is available."""
    global _FORCE_CPU
    _FORCE_CPU = force
    if force:
        print("  Forcing CPU usage (GPU disabled)")


def get_force_cpu() -> bool:
    """Return whether CPU usage is forced."""
    return _FORCE_CPU


@lru_cache(maxsize=1)
def _gpu_available_internal() -> bool:
    """Internal check for GPU availability (cached)."""
    if cp is None:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        # Verify that device context creation and a small allocation actually work.
        _ = cp.zeros(1, dtype=cp.float32)
        cp.cuda.runtime.deviceSynchronize()
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"  GPU detected: {device_name}")
        return True
    except Exception:
        return False


def gpu_available() -> bool:
    """Return True when CuPy is installed and can access at least one CUDA device, and CPU is not forced."""
    if _FORCE_CPU:
        return False
    return _gpu_available_internal()


# ---------------------------------------------------------------------------
# Thread count
# ---------------------------------------------------------------------------

_NUM_THREADS = 0  # 0 = auto
_MAX_AUTO_THREADS = 16  # cap to avoid oversubscribing CPU/BLAS threads


def get_num_threads(num_threads: int | None = None) -> int:
    """Return the number of threads to use, applying the auto-policy when 0."""
    resolved = _NUM_THREADS if num_threads is None else num_threads
    if resolved < 0:
        raise ValueError(f"num_threads must be >= 0 (0 = auto), got {resolved}")
    if resolved > 0:
        return resolved
    return min(os.cpu_count() or 1, _MAX_AUTO_THREADS)


def set_num_threads(num_threads: int) -> None:
    """Override the shared thread count (0 = auto-detect, bounded by _MAX_AUTO_THREADS)."""
    if num_threads < 0:
        raise ValueError(f"num_threads must be >= 0, got {num_threads}")
    global _NUM_THREADS
    _NUM_THREADS = num_threads


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------
# All helpers dispatch to CuPy or NumPy based on the input array type, so
# they work transparently for both CPU and GPU arrays.


def _xp(x):
    """Return the array module (numpy or cupy) for the given array."""
    return cp.get_array_module(x) if cp is not None else np


def ifft2c(x):
    """Centered 2D IFFT. Works with NumPy and CuPy arrays."""
    axes = (-2, -1)
    xp = _xp(x)
    return xp.fft.fftshift(
        xp.fft.ifftn(xp.fft.ifftshift(x, axes=axes), axes=axes), axes=axes
    )


def ifft3c(x):
    """Centered 3D IFFT. Works with NumPy and CuPy arrays."""
    axes = (-3, -2, -1)
    xp = _xp(x)
    return xp.fft.fftshift(
        xp.fft.ifftn(xp.fft.ifftshift(x, axes=axes), axes=axes), axes=axes
    )


def fft3c(x):
    """Centered 3D FFT. Works with NumPy and CuPy arrays."""
    axes = (-3, -2, -1)
    xp = _xp(x)
    return xp.fft.fftshift(
        xp.fft.fftn(xp.fft.ifftshift(x, axes=axes), axes=axes), axes=axes
    )
