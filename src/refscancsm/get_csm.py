"""Main workflow for generating coil sensitivity maps in target geometry."""

from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from .parse_cpx import read_cpx
from .parse_sin import (
    get_mps_to_xyz_transform,
    get_idx_to_mps_transform,
    get_matrix_size,
)
from .walsh import walsh_csm


def get_csm(
    sin_path_target: str,
    refscan_cpx_path: str = None,
    sin_path_refscan: str = None,
    location_idx: int = 1,
    interpolation_order: int = 1,
    squeeze: bool = True,
):
    """
    Get coil sensitivity maps from SENSE refscaninterpolated to target scan geometry.

    This is the main function that orchestrates the complete workflow:
    1. Load reference coil maps from .cpx file
    2. Read geometry information from .sin files
    3. Create transformation matrices
    4. Interpolate coil maps onto target geometry
    5. Return interpolated coil maps and metadata

    Parameters:
    -----------
    sin_path_target : str
        Path to target scan .sin file
    refscan_cpx_path : str, optional
        Path to reference scan .cpx file (with .cpx extension)
        If None, auto-detects file with 'senserefscan' in the same directory
    sin_path_refscan : str, optional
        Path to reference scan .sin file
        If None, auto-detects file with 'senserefscan' in the same directory
    location_idx : int
        Location index to extract from .sin files (default: 1)
    interpolation_order : int
        Interpolation order: 0=nearest, 1=linear, 3=cubic (default: 1)
        Order 1 (linear) is recommended to avoid overshoots at mask boundaries
    squeeze : bool
        Remove singleton dimensions from CPX data (default: True)

    Returns:
    --------
    - csm: numpy array [ncoils, nz, ny, nx] in target geometry
    """

    # Auto-detect refscan files if not provided
    if refscan_cpx_path is None or sin_path_refscan is None:
        print("\n Auto-detecting senserefscan files...")
        refscan_cpx_path, sin_path_refscan = _find_refscan_files(sin_path_target)
        print(f"      ✓ Found CPX: {refscan_cpx_path}")
        print(f"      ✓ Found SIN: {sin_path_refscan}")

    # Load low-resolution coil maps from SENSE refscan (exported as .cpx)
    refscan_coil_imgs = _load_refscan(refscan_cpx_path, squeeze=squeeze)

    # Affine transformation that maps refscan array indices to world coordinates
    refscan_idx_to_xyz = _load_idx_to_xyz_transformation(
        sin_path_refscan, "refscan", location_idx
    )

    # Affine transformation that maps target array indices to world coordinates
    target_idx_to_xyz = _load_idx_to_xyz_transformation(
        sin_path_target, "target", location_idx
    )

    # Affine transformation that maps target array indices to refscan array indices
    target_idx_to_refscan_idx = np.linalg.inv(refscan_idx_to_xyz) @ target_idx_to_xyz

    matrix_size_target = get_matrix_size(sin_path_target, "target")

    # Regrid refscan to target geometry
    interpolated_coil_imgs = _interpolate_to_target_geometry(
        refscan_coil_imgs,
        target_idx_to_refscan_idx,
        matrix_size_target,
        interpolation_order,
    )

    # For fully-sampled data, ESPIRIT can be simplified to the Walsh method for estimating coil sensitivity maps.
    csm = walsh_csm(interpolated_coil_imgs)

    return csm


def _find_refscan_files(target_sin_path: str):
    """
    Auto-detect senserefscan .cpx and .sin files in the same directory as target.

    Parameters:
    -----------
    target_sin_path : str
        Path to the target .sin file

    Returns:
    --------
    tuple: (cpx_path, sin_path) for senserefscan files

    Raises:
    -------
    FileNotFoundError: If no senserefscan files are found
    ValueError: If multiple senserefscan files are found
    """
    target_dir = Path(target_sin_path).parent

    # Find all .sin files with 'senserefscan' in the name
    sin_candidates = list(target_dir.glob("*senserefscan*.sin"))

    if len(sin_candidates) == 0:
        raise FileNotFoundError(
            f"No senserefscan .sin files found in {target_dir}. "
            "Please provide sin_path_refscan explicitly."
        )
    elif len(sin_candidates) > 1:
        raise ValueError(
            f"Multiple senserefscan .sin files found in {target_dir}:\n"
            + "\n".join(f"  - {f.name}" for f in sin_candidates)
            + "\nPlease provide sin_path_refscan explicitly."
        )

    sin_refscan = sin_candidates[0]

    # Find all .cpx files with 'senserefscan' in the name
    cpx_candidates = list(target_dir.glob("*senserefscan*.cpx"))

    if len(cpx_candidates) == 0:
        raise FileNotFoundError(
            f"No senserefscan .cpx files found in {target_dir}. "
            "Please provide refscan_cpx_path explicitly."
        )
    elif len(cpx_candidates) > 1:
        raise ValueError(
            f"Multiple senserefscan .cpx files found in {target_dir}:\n"
            + "\n".join(f"  - {f.name}" for f in cpx_candidates)
            + "\nPlease provide refscan_cpx_path explicitly."
        )

    cpx_refscan = cpx_candidates[0]

    # Return with .cpx extension
    return str(cpx_refscan), str(sin_refscan)


def _load_refscan(cpx_path: str, squeeze: bool = True):
    """Load and prepare coil sensitivity maps from SENSE refscan exported to .cpx file."""
    print(f"\n Loading refscan from {cpx_path}...")
    (csm, _, _) = read_cpx(cpx_path, squeeze=squeeze)
    print(f"      ✓ Loaded shape: {csm.shape}")

    # The coil maps have shape (ncoils, 2, nz, ny, nx)
    # where index 1 of the second dimension to corresponds to receive coils
    # and index 0 of the second dimension to corresponds to body coil
    # Because we're going to be using ESPIRiT, we don't use the body coil information
    csm = csm[:, 1, :, :, :]

    print(f"      ✓ Coil maps shape: {csm.shape} [ncoils, nz, ny, nx]")
    return csm


def _load_idx_to_xyz_transformation(sin_path: str, scan_type: str, location_idx: int):
    """Load transformation from array indices to world coordinates for a given scan."""
    idx_to_mps = get_idx_to_mps_transform(sin_path, scan_type)

    print(idx_to_mps)

    mps_to_xyz = get_mps_to_xyz_transform(sin_path, scan_type, location_idx)

    print(mps_to_xyz)
    idx_to_xyz = mps_to_xyz @ idx_to_mps

    return idx_to_xyz


def _interpolate_to_target_geometry(
    refscan_imgs, target_to_refscan_transform, target_shape, interpolation_order: int
):
    """Interpolate refscan images to target geometry."""
    print("\n Interpolating ...")

    ncoils = refscan_imgs.shape[0]
    target_shape_tuple = tuple(target_shape.astype(int))
    nx, ny, nz = target_shape_tuple

    print(f"      Target shape: {target_shape_tuple}")
    print(
        f"      Interpolation: {['nearest', 'linear', '', 'cubic'][interpolation_order]}"
    )

    target_coords = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1
    )
    target_coords_homogeneous = np.ones((*target_coords.shape[:-1], 4))
    target_coords_homogeneous[..., :3] = target_coords

    print("      Transforming coordinates...")
    refscan_coords_flat = (
        target_to_refscan_transform @ target_coords_homogeneous.reshape(-1, 4).T
    ).T
    refscan_coords = refscan_coords_flat[:, :3].reshape(*target_shape_tuple, 3)

    interpolated_imgs = np.zeros((ncoils, nz, ny, nx), dtype=np.complex64)

    for coil_idx in tqdm(range(ncoils)):
        coords_for_interp = np.array(
            [
                refscan_coords[..., 2].ravel(),  # z
                refscan_coords[..., 1].ravel(),  # y
                refscan_coords[..., 0].ravel(),  # x
            ]
        )

        real_part = map_coordinates(
            refscan_imgs[coil_idx, ...].real,
            coords_for_interp,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        ).reshape(nx, ny, nz)

        imag_part = map_coordinates(
            refscan_imgs[coil_idx, ...].imag,
            coords_for_interp,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        ).reshape(nx, ny, nz)

        interpolated_imgs[coil_idx, ...] = (real_part + 1j * imag_part).transpose(
            2, 1, 0
        )

    print("      ✓ Interpolation complete!")
    print(f"      Output shape: {interpolated_imgs.shape} [ncoils, nz, ny, nx]")

    return interpolated_imgs
