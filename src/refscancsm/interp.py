"""Spatial interpolation of refscan coil images onto target scan geometry."""

import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from .utils import cp, get_verbose, gpu_available


def interpolate_refscan_to_target_geometry(
    refscan_imgs,
    target_to_refscan_transform,
    target_shape,
    interpolation_order: int,
):
    """
    Interpolate refscan coil images to the target scan geometry.

    For each voxel in the target image grid, the affine transform is applied to
    find the corresponding location in the refscan grid, and the coil images are
    sampled there. GPU acceleration (CuPy) is used automatically when available.

    Parameters
    ----------
    refscan_imgs : ndarray, shape (n_coils, nz, ny, nx)
        Refscan coil images in refscan geometry.
    target_to_refscan_transform : ndarray, shape (4, 4)
        Affine matrix mapping target array indices to refscan array indices.
    target_shape : ndarray
        [nx, ny, nz] of the target grid (Philips convention: x first).
    interpolation_order : int
        Spline order: 0 = nearest, 1 = linear, 3 = cubic.

    Returns
    -------
    ndarray, shape (n_coils, nz, ny, nx)
        Coil images resampled onto the target geometry.
    """
    use_gpu = gpu_available()
    xp = cp if use_gpu else np

    # CuPy's map_coordinates must be imported lazily — importing it at module
    # level would call gpu_available() during import, which may trigger CUDA
    # context initialisation before the user has a chance to configure the device.
    if use_gpu:
        from cupyx.scipy.ndimage import map_coordinates as map_fn
    else:
        map_fn = map_coordinates

    ncoils = refscan_imgs.shape[0]
    target_shape_int = tuple(target_shape.astype(int))
    nx, ny, nz = target_shape_int  # Philips: x = frequency, y = phase, z = slice

    # Move data to the selected device
    refscan_data = xp.asarray(refscan_imgs.astype(np.complex64))
    transform = xp.asarray(target_to_refscan_transform.astype(np.float32))

    # Build target-grid coordinates on device, then flip to match the physical
    # orientation assumed by the transform (see _load_refscan_coil_images).
    Z, Y, X = xp.meshgrid(
        xp.arange(nz, dtype=xp.float32),
        xp.arange(ny, dtype=xp.float32),
        xp.arange(nx, dtype=xp.float32),
        indexing="ij",
    )
    X = (nx - 1) - X
    Y = (ny - 1) - Y
    Z = (nz - 1) - Z

    # Homogeneous coordinates: shape (N, 4) where N = nz * ny * nx
    target_coords_homogeneous = xp.ones((*Z.shape, 4), dtype=xp.float32)
    target_coords_homogeneous[..., 0] = X
    target_coords_homogeneous[..., 1] = Y
    target_coords_homogeneous[..., 2] = Z

    # Apply affine transform: target indices -> refscan indices
    refscan_coords = (
        (transform @ target_coords_homogeneous.reshape(-1, 4).T)
        .T[:, :3]
        .reshape(nz, ny, nx, 3)
    )

    # map_coordinates expects coordinates in axis order [axis0=Z, axis1=Y, axis2=X]
    coords_for_interp = xp.stack(
        [
            refscan_coords[..., 2],  # Z (slice axis)
            refscan_coords[..., 1],  # Y (phase axis)
            refscan_coords[..., 0],  # X (frequency axis)
        ],
        axis=0,
    )

    result = xp.zeros((ncoils, nz, ny, nx), dtype=xp.complex64)

    for coil_idx in tqdm(
        range(ncoils), desc="  Interpolating coils", disable=not get_verbose()
    ):
        result[coil_idx] = map_fn(
            refscan_data[coil_idx].real,
            coords_for_interp,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        ) + 1j * map_fn(
            refscan_data[coil_idx].imag,
            coords_for_interp,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        )

    # Keep data on GPU if available - let downstream operations decide when to transfer
    return result
