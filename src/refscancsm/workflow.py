"""Main workflow for generating coil sensitivity maps in target geometry."""

import numpy as np
from scipy.ndimage import map_coordinates

from .parse_cpx import read_cpx
from .parse_sin import (
    get_mps_to_xyz_transform,
    get_idx_to_mps_transform,
    get_matrix_size
)


def get_csm(
    refscan_cpx_path: str,
    sin_path_refscan: str,
    sin_path_target: str,
    location_idx: int = 1,
    interpolation_order: int = 1,
    verbose: bool = True,
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
    refscan_cpx_path : str
        Path to reference scan .cpx file (without .cpx extension)
    sin_path_refscan : str
        Path to reference scan .sin file
    sin_path_target : str
        Path to target scan .sin file
    location_idx : int
        Location index to extract from .sin files (default: 1)
    interpolation_order : int
        Interpolation order: 0=nearest, 1=linear, 3=cubic (default: 1)
        Order 1 (linear) is recommended to avoid overshoots at mask boundaries

    Returns:
    --------
    tuple
        (interpolated_coil_maps, metadata)
        - interpolated_coil_maps: numpy array [ncoils, nz, ny, nx] in target geometry
        - metadata: dict with geometry information
    """
    # Special print function that only prints when verbose=True
    global vprint
    vprint = _create_printer(verbose)
    
    # Load low-resolution coil maps from SENSE refscan (exported as .cpx)
    source_csm = _load_source_csm_from_cpx(refscan_cpx_path)
    
    # Affine transformation that maps source array indices to world coordinates
    source_idx_to_xyz = _load_idx_to_xyz_transformation(
        sin_path_refscan, "source", location_idx
    )
    
    # Affine transformation that maps target array indices to world coordinates
    target_idx_to_xyz = _load_idx_to_xyz_transformation(
        sin_path_target, "target", location_idx
    )
    
    # Affine transformation that maps target array indices to source array indices
    target_idx_to_source_idx = np.linalg.inv(source_idx_to_xyz) @ target_idx_to_xyz

    matrix_size_target = get_matrix_size(sin_path_target)
    
    # Regrid coil maps from source to target geometry
    target_csm = _interpolate_csm_to_target_geometry(
        source_csm,
        target_idx_to_source_idx,
        matrix_size_target,
        interpolation_order
    )
    
    return target_csm


def _create_printer(verbose: bool):
    """Create a print function that only prints when verbose is True."""
    if verbose:
        return print
    else:
        return lambda *args, **kwargs: None


def _load_source_csm_from_cpx(cpx_path: str):
    """Load and prepare coil sensitivity maps from SENSE refscan exported to .cpx file."""
    vprint(f"\n[1/5] Loading coil maps from {cpx_path}.cpx...")
    (csm, _, _) = read_cpx(cpx_path)
    vprint(f"      ✓ Loaded shape: {csm.shape}")

    # The coil maps have shape (ncoils, 1, 1, 1, 1, 1, 2, nz, ny, nx)
    # Squeeze, and select first volume that corresponds to receiver coil maps
    csm = csm.squeeze()
    csm = csm[:,0,:,:,:]

    vprint(f"      ✓ Coil maps shape: {csm.shape} [ncoils, nz, ny, nx]")
    return csm


def _load_idx_to_xyz_transformation(sin_path: str, scan_type: str, location_idx: int):
    """Load transformation from array indices to world coordinates for a given scan."""
    idx_to_mps = get_idx_to_mps_transform(sin_path)
    mps_to_xyz = get_mps_to_xyz_transform(sin_path, scan_type, location_idx)
    idx_to_xyz = mps_to_xyz @ idx_to_mps
    
    return idx_to_xyz

def _interpolate_csm_to_target_geometry(
    source_csm, 
    target_to_source_transform,
    target_shape,
    interpolation_order: int
):
    """Interpolate coil maps from source to target geometry."""
    vprint("\n[5/5] Interpolating coil maps onto target geometry...")
    
    ncoils = source_csm.shape[0]
    target_shape_tuple = tuple(target_shape.astype(int))
    nx_t, ny_t, nz_t = target_shape_tuple
    
    vprint(f"      Target shape: {target_shape_tuple}")
    vprint(f"      Interpolation: {['nearest', 'linear', '', 'cubic'][interpolation_order]}")
    
    target_coords = np.stack(
        np.meshgrid(np.arange(nx_t), np.arange(ny_t), np.arange(nz_t), indexing="ij"), 
        axis=-1
    )
    target_coords_homogeneous = np.ones((*target_coords.shape[:-1], 4))
    target_coords_homogeneous[..., :3] = target_coords
    
    vprint("      Transforming coordinates...")
    source_coords_flat = (target_to_source_transform @ target_coords_homogeneous.reshape(-1, 4).T).T
    source_coords = source_coords_flat[:, :3].reshape(*target_shape_tuple, 3)
    
    target_csm = np.zeros((ncoils, nz_t, ny_t, nx_t), dtype=np.complex64)
    
    for coil_idx in range(ncoils):
        if (coil_idx + 1) % 10 == 0 or coil_idx == 0 or coil_idx == ncoils - 1:
            vprint(f"      Processing coil {coil_idx + 1}/{ncoils}...")
        
        coords_for_interp = np.array([
            source_coords[..., 2].ravel(),  # z
            source_coords[..., 1].ravel(),  # y
            source_coords[..., 0].ravel(),  # x
        ])
        
        real_part = map_coordinates(
            source_csm[coil_idx, ...].real,
            coords_for_interp,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        ).reshape(nx_t, ny_t, nz_t)
        
        imag_part = map_coordinates(
            source_csm[coil_idx, ...].imag,
            coords_for_interp,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        ).reshape(nx_t, ny_t, nz_t)
        
        target_csm[coil_idx, ...] = (real_part + 1j * imag_part).transpose(2, 1, 0)
    
    vprint("      ✓ Interpolation complete!")
    vprint(f"      Output shape: {target_csm.shape} [ncoils, nz, ny, nx]")
    
    return target_csm
