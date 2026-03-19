"""Shared utilities: device selection, thread count, timing, and FFT helpers."""

import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Verbose output control
# ---------------------------------------------------------------------------

_VERBOSE = False  # Global flag to control verbose output


def set_verbose(verbose: bool) -> None:
    """Set whether to enable verbose output."""
    global _VERBOSE
    _VERBOSE = verbose


def get_verbose() -> bool:
    """Return whether verbose output is enabled."""
    return _VERBOSE


def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if _VERBOSE:
        print(*args, **kwargs)


@contextmanager
def timed(label):
    """Context manager that prints a step label, runs the block, then prints elapsed time."""
    vprint(f"  {label}...")
    t0 = time.perf_counter()
    yield
    vprint(f"    ({time.perf_counter() - t0:.1f}s)")


class Spinner:
    """
    Threaded terminal spinner for non-verbose mode.

    Writes a rotating character to stdout using \\r so it works on any
    terminal emulator, including basic Linux ones (no ANSI codes used).
    Silently disabled when stdout is not a TTY (e.g. piped output).

    Usage::

        with Spinner("Computing CSM"):
            heavy_computation()
        # prints: "Computing CSM... done (12.3s)"
    """

    _FRAMES = r"|/-\\"

    def __init__(self, message: str = "Working"):
        self._message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._active = sys.stdout.isatty()

    def _spin(self):
        prefix = f"  {self._message}... "
        for char in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{prefix}{char}")
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the spinner line so the final message can be printed cleanly
        sys.stdout.write("\r" + " " * (len(prefix) + 1) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self._active:
            self._thread.start()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t0
        self._stop.set()
        if self._active:
            self._thread.join()
            print(f"  {self._message}... done ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Device selection (PyTorch)
# ---------------------------------------------------------------------------

_FORCE_CPU = False  # Global flag to force CPU usage


def set_force_cpu(force: bool) -> None:
    """Force CPU usage even when a GPU or MPS device is available."""
    global _FORCE_CPU
    _FORCE_CPU = force
    if force:
        vprint("  Forcing CPU usage")


def get_force_cpu() -> bool:
    """Return whether CPU usage is forced."""
    return _FORCE_CPU


def get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if _FORCE_CPU:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_available() -> bool:
    """Return True when a non-CPU torch device is available and CPU is not forced."""
    if _FORCE_CPU:
        return False
    return torch.cuda.is_available() or torch.backends.mps.is_available()


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
# FFT helpers (PyTorch — device-agnostic: CUDA, MPS, CPU)
# ---------------------------------------------------------------------------


def fft3c(x: torch.Tensor) -> torch.Tensor:
    """Centered 3D FFT. Works on CUDA, MPS, and CPU tensors."""
    dims = (-3, -2, -1)
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims),
        dim=dims,
    )


def ifft3c(x: torch.Tensor) -> torch.Tensor:
    """Centered 3D IFFT. Works on CUDA, MPS, and CPU tensors."""
    dims = (-3, -2, -1)
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims),
        dim=dims,
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT. Works on CUDA, MPS, and CPU tensors."""
    dims = (-2, -1)
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims),
        dim=dims,
    )
