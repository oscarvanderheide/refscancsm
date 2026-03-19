"""Spatial interpolation of refscan coil images onto target scan geometry."""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import get_device, get_verbose


def interpolate_refscan_to_target_geometry(
    refscan_imgs: np.ndarray,
    target_to_refscan_transform: np.ndarray,
    target_shape,
    interpolation_order: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Interpolate refscan coil images to the target scan geometry.

    For each voxel in the target grid the affine transform is applied to find
    the corresponding refscan grid location, and the coil images are sampled
    there.  Orders 0 (nearest) and 1 (trilinear) run on *device* using
    ``torch.nn.functional.grid_sample``.  Order 3 (cubic) falls back to
    ``scipy.ndimage.map_coordinates`` on CPU with a warning, then the result
    is moved to *device*.

    Parameters
    ----------
    refscan_imgs : ndarray, shape (n_coils, nz, ny, nx)
        Refscan coil images in refscan geometry.
    target_to_refscan_transform : ndarray, shape (4, 4)
        Affine matrix mapping target array indices to refscan array indices.
    target_shape : ndarray
        [nx, ny, nz] of the target grid (Philips convention: x first).
    interpolation_order : int
        Spline order: 0 = nearest, 1 = linear / trilinear, 3 = cubic (CPU fallback).
    device : torch.device, optional
        Target device.  Auto-detected when None.

    Returns
    -------
    torch.Tensor, shape (n_coils, nz, ny, nx), dtype complex64
        Coil images resampled onto the target geometry, on *device*.
    """
    if device is None:
        device = get_device()

    ncoils = refscan_imgs.shape[0]
    nz_ref, ny_ref, nx_ref = refscan_imgs.shape[1:4]

    target_shape_int = tuple(target_shape.astype(int))
    nx, ny, nz = target_shape_int  # Philips: x = frequency, y = phase, z = slice

    # Move data to device
    refscan_data = torch.from_numpy(refscan_imgs.astype(np.complex64)).to(device)
    transform = torch.from_numpy(target_to_refscan_transform.astype(np.float32)).to(
        device
    )

    # Build target-grid voxel coordinates and flip axes to match reconframe orientation
    Z, Y, X = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    X = (nx - 1) - X
    Y = (ny - 1) - Y
    Z = (nz - 1) - Z

    # Homogeneous coords (nz*ny*nx, 4) → apply affine → refscan pixel coords
    target_coords = torch.stack(
        [X, Y, Z, torch.ones_like(X)], dim=-1
    )  # (nz, ny, nx, 4)
    refscan_coords = (target_coords.reshape(-1, 4) @ transform.T)[..., :3].reshape(
        nz, ny, nx, 3
    )
    # refscan_coords[..., 0] = X_ref (freq/W), [..., 1] = Y_ref (phase/H), [..., 2] = Z_ref (slice/D)

    if interpolation_order in (0, 1):
        return _grid_sample_interp(
            refscan_data,
            refscan_coords,
            (nz_ref, ny_ref, nx_ref),
            interpolation_order,
        )
    if interpolation_order == 3:
        warnings.warn(
            "Cubic interpolation (order=3) runs on CPU via scipy.ndimage.map_coordinates.",
            UserWarning,
            stacklevel=2,
        )
        refscan_coords_np = refscan_coords.detach().cpu().numpy()
        return _scipy_cubic_interp(
            refscan_imgs,
            refscan_coords_np,
            (ncoils, nz, ny, nx),
            device,
        )

    raise ValueError(
        f"Unsupported interpolation_order {interpolation_order}. "
        "Use 0 (nearest), 1 (linear), or 3 (cubic)."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _grid_sample_interp(
    refscan_data: torch.Tensor,
    refscan_coords: torch.Tensor,
    ref_shape: tuple,
    order: int,
) -> torch.Tensor:
    """
    All-device interpolation using ``torch.nn.functional.grid_sample``.

    Processes all coils at once (no per-coil Python loop) by treating the
    coil axis as the channel dimension.  Real and imaginary parts are
    interpolated separately because ``grid_sample`` does not support complex
    tensors.

    The sampling grid uses ``align_corners=True`` so that pixel coordinate 0
    maps to −1 and pixel coordinate N−1 maps to +1, matching the behaviour of
    ``scipy.ndimage.map_coordinates`` with no ``prefilter``.

    Parameters
    ----------
    refscan_data : Tensor, shape (n_coils, nz_ref, ny_ref, nx_ref), complex64
    refscan_coords : Tensor, shape (nz, ny, nx, 3)
        Pixel coordinates in refscan space [X_ref, Y_ref, Z_ref].
    ref_shape : (nz_ref, ny_ref, nx_ref)
    order : int
        0 = nearest, 1 = bilinear/trilinear.
    """
    nz_ref, ny_ref, nx_ref = ref_shape

    # Normalize pixel coords to [-1, 1] (align_corners=True convention)
    # grid_sample last dim order: (x, y, z) -> maps to (W, H, D) of 5-D input
    norm_x = 2.0 * refscan_coords[..., 0] / (nx_ref - 1) - 1.0
    norm_y = 2.0 * refscan_coords[..., 1] / (ny_ref - 1) - 1.0
    norm_z = 2.0 * refscan_coords[..., 2] / (nz_ref - 1) - 1.0

    # grid: (1, nz, ny, nx, 3)
    grid = torch.stack([norm_x, norm_y, norm_z], dim=-1).unsqueeze(0)

    mode = "nearest" if order == 0 else "bilinear"

    # Input shape for grid_sample: (1, n_coils, nz_ref, ny_ref, nx_ref)
    out_real = F.grid_sample(
        refscan_data.real.unsqueeze(0),
        grid,
        mode=mode,
        align_corners=True,
        padding_mode="zeros",
    )
    out_imag = F.grid_sample(
        refscan_data.imag.unsqueeze(0),
        grid,
        mode=mode,
        align_corners=True,
        padding_mode="zeros",
    )

    # Output: (1, n_coils, nz, ny, nx) -> (n_coils, nz, ny, nx)
    return torch.complex(out_real[0], out_imag[0])


def _scipy_cubic_interp(
    refscan_imgs: np.ndarray,
    refscan_coords_np: np.ndarray,
    out_shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    """CPU cubic interpolation via scipy; result is moved to *device*."""
    from scipy.ndimage import map_coordinates

    ncoils, nz, ny, nx = out_shape
    data = refscan_imgs.astype(np.complex64)

    # map_coordinates expects coordinates in (axis0=Z, axis1=Y, axis2=X) order
    coords = np.stack(
        [
            refscan_coords_np[..., 2],  # Z (slice)
            refscan_coords_np[..., 1],  # Y (phase)
            refscan_coords_np[..., 0],  # X (freq)
        ],
        axis=0,
    )

    result = np.zeros((ncoils, nz, ny, nx), dtype=np.complex64)
    for c in tqdm(
        range(ncoils), desc="  Interpolating coils", disable=not get_verbose()
    ):
        result[c] = map_coordinates(
            data[c].real, coords, order=3, mode="constant", cval=0.0
        ) + 1j * map_coordinates(
            data[c].imag, coords, order=3, mode="constant", cval=0.0
        )

    return torch.from_numpy(result).to(device)
