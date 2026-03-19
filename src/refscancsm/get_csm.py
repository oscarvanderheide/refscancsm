"""Main workflow: load Philips refscan data and compute coil sensitivity maps."""

from pathlib import Path

import numpy as np
import torch

from espirit import espirit
from .interp import interpolate_refscan_to_target_geometry
from .parse_cpx import read_cpx
from .parse_sin import (
    get_idx_to_mps_transform,
    get_matrix_size,
    get_mps_to_xyz_transform,
)
from .utils import fft3c, get_device, set_force_cpu, set_verbose, timed, vprint, Spinner


def get_csm(
    sin_path_target: str,
    refscan_cpx_path: str | None = None,
    sin_path_refscan: str | None = None,
    interpolation_order: int = 1,
    calib_size: int = 24,
    kernel_size: int = 6,
    threshold: float = 0.001,
    device: str | torch.device | None = None,
    force_cpu: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute coil sensitivity maps from a Philips SENSE refscan in target scan geometry.

    This is the top-level function. It orchestrates the complete workflow:
      1. Auto-detect refscan files (if not provided)
      2. Load refscan coil images from the .cpx file
      3. Build the affine transform from target array indices to refscan array indices
      4. Interpolate refscan images onto the target geometry (GPU/MPS/CPU)
      5. Compute 3D FFT (GPU/MPS/CPU)
      6. Run ESPIRiT to extract coil sensitivity maps

    Parameters
    ----------
    sin_path_target : str
        Path to the target scan .sin file.
    refscan_cpx_path : str, optional
        Path to the refscan .cpx file. Auto-detected when None.
    sin_path_refscan : str, optional
        Path to the refscan .sin file. Auto-detected when None.
    interpolation_order : int
        Spline order for spatial interpolation: 0=nearest, 1=linear (default), 3=cubic.
        Order 1 is recommended to avoid overshoot at mask boundaries.  Order 3
        uses a CPU fallback (scipy) with a warning.
    calib_size : int
        Size of the k-space calibration region for ESPIRiT (default: 24).
    kernel_size : int
        ESPIRiT kernel size (default: 6).
    threshold : float
        Relative singular-value threshold for kernel selection (default: 0.001).
    device : str or torch.device, optional
        PyTorch device to run on (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``).
        Auto-detected when None (CUDA > MPS > CPU).
    force_cpu : bool
        Force CPU usage even when a GPU is available.  Equivalent to
        ``device="cpu"``; kept for backward compatibility (default: False).
    verbose : bool
        Enable verbose output with timing information (default: False).

    Returns
    -------
    csm : ndarray, shape (n_coils, nz, ny, nx), dtype complex64
        Coil sensitivity maps in the target scan geometry.
    """
    # Set verbosity and resolve device
    set_verbose(verbose)
    if force_cpu:
        device = torch.device("cpu")
        set_force_cpu(True)
    elif device is not None:
        device = torch.device(device) if isinstance(device, str) else device
    else:
        device = get_device()
    vprint(f"  Device: {device}")

    if refscan_cpx_path is None or sin_path_refscan is None:
        refscan_cpx_path, sin_path_refscan = _find_refscan_files(sin_path_target)

    refscan_coil_imgs = _load_refscan_coil_images(refscan_cpx_path)

    target_idx_to_refscan_idx = _compute_target_to_refscan_idx_transform(
        sin_path_refscan, sin_path_target
    )
    matrix_size_target = get_matrix_size(sin_path_target, "target")

    spinner = Spinner("Computing coil sensitivity maps") if not verbose else None
    if spinner:
        spinner.__enter__()
    try:
        csm = _run_pipeline(
            refscan_coil_imgs,
            target_idx_to_refscan_idx,
            matrix_size_target,
            interpolation_order,
            calib_size,
            kernel_size,
            threshold,
            device,
        )
    except (RuntimeError, Exception) as exc:
        # Catch CUDA / MPS out-of-memory errors and retry on CPU
        _oom_keywords = ("out of memory", "cudaErrorMemoryAllocation", "AcceleratorError")
        if device.type != "cpu" and any(kw.lower() in str(exc).lower() for kw in _oom_keywords):
            import warnings
            warnings.warn(
                f"GPU out of memory ({device}); retrying on CPU. "
                "Pass device='cpu' to avoid this overhead next time.",
                RuntimeWarning,
                stacklevel=2,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            device = torch.device("cpu")
            csm = _run_pipeline(
                refscan_coil_imgs,
                target_idx_to_refscan_idx,
                matrix_size_target,
                interpolation_order,
                calib_size,
                kernel_size,
                threshold,
                device,
            )
        else:
            raise
    finally:
        if spinner:
            spinner.__exit__(None, None, None)

    # Always return a NumPy array
    if isinstance(csm, torch.Tensor):
        return csm.detach().cpu().numpy()
    return np.asarray(csm)


def _run_pipeline(
    refscan_coil_imgs,
    target_idx_to_refscan_idx,
    matrix_size_target,
    interpolation_order,
    calib_size,
    kernel_size,
    threshold,
    device,
):
    """Run the interpolation → FFT → ESPIRiT pipeline on the given device."""
    with timed("Interpolating refscan to target geometry"):
        interpolated_coil_imgs = interpolate_refscan_to_target_geometry(
            refscan_coil_imgs,
            target_idx_to_refscan_idx,
            matrix_size_target,
            interpolation_order,
            device=device,
        )

    with timed("Converting coil images to k-space (3D FFT)"):
        kspace = fft3c(interpolated_coil_imgs)

    return espirit(
        kspace,
        calib_size=calib_size,
        kernel_size=kernel_size,
        threshold=threshold,
        device=device,
    )


# =============================================================================
# HELPERS
# =============================================================================


def _find_refscan_files(target_sin_path: str):
    """
    Auto-detect the senserefscan .cpx and .sin files in the same directory as the target.

    Raises FileNotFoundError if no matching files are found, and ValueError if
    multiple matches exist (the caller must then pass the paths explicitly).
    """
    target_dir = Path(target_sin_path).parent

    sin_candidates = list(target_dir.glob("*senserefscan*.sin"))
    if len(sin_candidates) == 0:
        raise FileNotFoundError(
            f"No senserefscan .sin files found in {target_dir}. "
            "Please provide sin_path_refscan explicitly."
        )
    if len(sin_candidates) > 1:
        raise ValueError(
            f"Multiple senserefscan .sin files found in {target_dir}:\n"
            + "\n".join(f"  {f.name}" for f in sin_candidates)
            + "\nPlease provide sin_path_refscan explicitly."
        )

    cpx_candidates = list(target_dir.glob("*senserefscan*.cpx"))
    if len(cpx_candidates) == 0:
        raise FileNotFoundError(
            f"No senserefscan .cpx files found in {target_dir}. "
            "Please provide refscan_cpx_path explicitly."
        )
    if len(cpx_candidates) > 1:
        raise ValueError(
            f"Multiple senserefscan .cpx files found in {target_dir}:\n"
            + "\n".join(f"  {f.name}" for f in cpx_candidates)
            + "\nPlease provide refscan_cpx_path explicitly."
        )

    vprint(f"  Refscan CPX: {cpx_candidates[0]}")
    vprint(f"  Refscan SIN: {sin_candidates[0]}")
    return str(cpx_candidates[0]), str(sin_candidates[0])


def _load_refscan_coil_images(cpx_path: str):
    """
    Load coil images from a Philips SENSE refscan .cpx file.

    The .cpx file contains both body-coil and receive-coil images; only the
    receive-coil images (index 1 along the second dimension) are used for ESPIRiT.
    The z and xy orientations are also flipped to match the reconframe convention.
    """
    vprint(f"  Reading CPX: {cpx_path}")
    (data, _, _) = read_cpx(cpx_path, squeeze=True)

    # Dimension order after squeeze: (n_coils, 2, nz, ny, nx)
    # Index 0 = body coil, index 1 = receive coils (used for ESPIRiT)
    coil_imgs = data[:, 1, :, :, :]

    # Flip z and rotate 180° in the xy plane to match reconframe convention
    coil_imgs = coil_imgs[:, ::-1, :, :]
    coil_imgs = np.rot90(coil_imgs, k=2, axes=(2, 3))

    vprint(f"  Coil images loaded: {coil_imgs.shape}  [n_coils, nz, ny, nx]")
    return coil_imgs


def _compute_target_to_refscan_idx_transform(
    sin_path_refscan: str, sin_path_target: str
):
    """
    Build a 4x4 affine matrix from target array indices to refscan array indices.

    The chain is:  target_idx -> MPS -> world (xyz) -> MPS -> refscan_idx.
    The intermediate world-coordinate step lets the two scans live in different
    orientations / positions and still be registered correctly.
    """
    refscan_idx_to_xyz = _compute_index_to_world_transform(sin_path_refscan, "refscan")
    target_idx_to_xyz = _compute_index_to_world_transform(sin_path_target, "target")
    return np.linalg.inv(refscan_idx_to_xyz) @ target_idx_to_xyz


def _compute_index_to_world_transform(sin_path: str, scan_type: str):
    """Build a 4x4 affine matrix from array indices to world (xyz) coordinates."""
    idx_to_mps = get_idx_to_mps_transform(sin_path, scan_type)
    mps_to_xyz = get_mps_to_xyz_transform(sin_path, scan_type)
    return mps_to_xyz @ idx_to_mps
