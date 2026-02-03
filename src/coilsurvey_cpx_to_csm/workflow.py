"""Main workflow for generating coil sensitivity maps in target geometry."""

import numpy as np
from scipy.ndimage import map_coordinates

from .read_cpx import read_cpx
from .read_sin import (
    read_location_matrix,
    read_voxel_sizes,
    read_matrix_size,
)
from .transforms import (
    transform_to_MPS_refscan,
    transform_to_MPS_target,
    create_mps_matrix,
)


def get_csm(
    refscan_cpx_path,
    refscan_sin_path,
    target_sin_path,
    location_idx=1,
    interpolation_order=1,
):
    """
    Get coil sensitivity maps interpolated to target scan geometry.
    
    This is the main function that orchestrates the complete workflow:
    1. Load reference coil maps from CPX file
    2. Read geometry information from SIN files
    3. Create transformation matrices
    4. Interpolate coil maps onto target geometry
    4. Interpolate coil maps onto target geometry
    
    Parameters:
    -----------
    refscan_cpx_path : str
        Path to reference scan CPX file (without .cpx extension)
    refscan_sin_path : str
        Path to reference scan SIN file
    target_sin_path : str
        Path to target scan SIN file
    location_idx : int
        Location index to extract from SIN files (default: 1)
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
    print("\n" + "=" * 70)
    print("COIL SENSITIVITY MAP INTERPOLATION")
    print("=" * 70)
    
    # Step 1: Load reference coil maps from CPX file
    print("\n[1/5] Loading reference coil maps from CPX file...")
    print(f"      File: {refscan_cpx_path}.cpx")
    ref_data, ref_hdr, ref_labels = read_cpx(refscan_cpx_path)
    print(f"      ✓ Loaded shape: {ref_data.shape}")
    print(f"      ✓ Data labels: {ref_labels}")
    
    # Get number of coils (first dimension in Python order)
    ncoils = ref_data.shape[0]
    
    # Reshape to [ncoils, nz, ny, nx] if needed
    # Assuming the last 3 dimensions are spatial
    ref_coil_maps = ref_data.reshape(ncoils, -1, ref_data.shape[-2], ref_data.shape[-1])
    # Take first volume if there are multiple dynamics/phases
    if ref_coil_maps.shape[1] > 1:
        print(f"      Note: Multiple volumes detected, using first volume only")
        ref_coil_maps = ref_coil_maps[:, 0, :, :]
    else:
        ref_coil_maps = ref_coil_maps[:, 0, :, :]
    
    print(f"      ✓ Coil maps shape: {ref_coil_maps.shape} [ncoils, nz, ny, nx]")
    
    # Step 2: Read reference scan geometry
    print(f"\n[2/5] Reading reference scan geometry...")
    print(f"      File: {refscan_sin_path}")
    ref_loc_matrix = read_location_matrix(refscan_sin_path, location_idx)
    ref_voxel_sizes = read_voxel_sizes(refscan_sin_path)
    ref_matrix_size = read_matrix_size(refscan_sin_path)
    
    print(f"      ✓ Matrix size: {ref_matrix_size}")
    print(f"      ✓ Voxel sizes: {ref_voxel_sizes} mm")
    
    # Create reference MPS transformation matrix
    T_MPS_refscan = transform_to_MPS_refscan(ref_loc_matrix)
    ref_mps_matrix = create_mps_matrix(ref_matrix_size, ref_voxel_sizes)
    
    # Step 3: Read target scan geometry
    print(f"\n[3/5] Reading target scan geometry...")
    print(f"      File: {target_sin_path}")
    target_loc_matrix = read_location_matrix(target_sin_path, location_idx)
    target_voxel_sizes = read_voxel_sizes(target_sin_path)
    target_matrix_size = read_matrix_size(target_sin_path)
    
    print(f"      ✓ Matrix size: {target_matrix_size}")
    print(f"      ✓ Voxel sizes: {target_voxel_sizes} mm")
    
    # Create target MPS transformation matrix
    T_MPS_target = transform_to_MPS_target(target_loc_matrix)
    target_mps_matrix = create_mps_matrix(target_matrix_size, target_voxel_sizes)
    
    # Step 4: Compute transformation from target to reference
    print(f"\n[4/5] Computing coordinate transformations...")
    
    # Combined transformation: Target indices -> mm -> reference mm -> reference indices
    # T_TargetToRef = inv(ref_mps_matrix) @ inv(T_MPS_refscan) @ T_MPS_target @ target_mps_matrix
    T_TargetToRef = (
        np.linalg.inv(ref_mps_matrix)
        @ np.linalg.inv(T_MPS_refscan)
        @ T_MPS_target
        @ target_mps_matrix
    )
    
    print(f"      ✓ Transformation matrix computed")
    
    # Step 5: Create target coordinate grid and interpolate
    print(f"\n[5/5] Interpolating coil maps onto target geometry...")
    target_shape = tuple(target_matrix_size.astype(int))
    print(f"      Target shape: {target_shape}")
    print(f"      Interpolation order: {interpolation_order} " 
          f"({'nearest' if interpolation_order == 0 else 'linear' if interpolation_order == 1 else 'cubic'})")
    
    # Create meshgrid for target indices
    # Note: meshgrid order matches [x, y, z] but we need to be careful with dimensions
    nx_t, ny_t, nz_t = target_shape
    x_coords = np.arange(nx_t)
    y_coords = np.arange(ny_t)
    z_coords = np.arange(nz_t)
    
    # Create coordinate arrays for all target voxels
    target_coords = np.stack(np.meshgrid(x_coords, y_coords, z_coords, indexing='ij'), axis=-1)
    target_coords_homogeneous = np.ones((*target_coords.shape[:-1], 4))
    target_coords_homogeneous[..., :3] = target_coords
    
    # Transform to reference coordinates
    print(f"      Transforming {nx_t}×{ny_t}×{nz_t} = {nx_t*ny_t*nz_t:,} voxels...")
    ref_coords_flat = (T_TargetToRef @ target_coords_homogeneous.reshape(-1, 4).T).T
    ref_coords = ref_coords_flat[:, :3].reshape(*target_shape, 3)
    
    # Interpolate each coil
    interpolated_maps = np.zeros((ncoils, nz_t, ny_t, nx_t), dtype=np.complex64)
    
    # Get reference coordinates for interpolation
    # ref_coil_maps is [ncoils, nz, ny, nx]
    nz_r, ny_r, nx_r = ref_coil_maps.shape[1:]
    
    for coil_idx in range(ncoils):
        if (coil_idx + 1) % 10 == 0 or coil_idx == 0 or coil_idx == ncoils - 1:
            print(f"      Processing coil {coil_idx + 1}/{ncoils}...")
        
        # Interpolate real and imaginary parts separately
        # map_coordinates expects coordinates in array order [z, y, x]
        # We need to reorder ref_coords from [x, y, z] to [z, y, x]
        coords_for_interp = np.array([
            ref_coords[..., 2].ravel(),  # z
            ref_coords[..., 1].ravel(),  # y  
            ref_coords[..., 0].ravel(),  # x
        ])
        
        real_part = map_coordinates(
            ref_coil_maps[coil_idx, ...].real,
            coords_for_interp,
            order=interpolation_order,
            mode='constant',
            cval=0.0,
        ).reshape(nx_t, ny_t, nz_t)
        
        imag_part = map_coordinates(
            ref_coil_maps[coil_idx, ...].imag,
            coords_for_interp,
            order=interpolation_order,
            mode='constant',
            cval=0.0,
        ).reshape(nx_t, ny_t, nz_t)
        
        # Store in [ncoils, nz, ny, nx] format
        interpolated_maps[coil_idx, ...] = (real_part + 1j * imag_part).transpose(2, 1, 0)
    
    print(f"      ✓ Interpolation complete!")
    print(f"      Output shape: {interpolated_maps.shape} [ncoils, nz, ny, nx]")
    
    # Prepare metadata
    metadata = {
        'ncoils': ncoils,
        'reference_shape': ref_coil_maps.shape,
        'reference_voxel_sizes': ref_voxel_sizes,
        'reference_matrix_size': ref_matrix_size,
        'target_shape': interpolated_maps.shape,
        'target_voxel_sizes': target_voxel_sizes,
        'target_matrix_size': target_matrix_size,
        'interpolation_order': interpolation_order,
        'T_MPS_refscan': T_MPS_refscan,
        'T_MPS_target': T_MPS_target,
    }
    
    print("\n" + "=" * 70)
    print("✓ INTERPOLATION SUCCESSFUL")
    print("=" * 70 + "\n")
    
    return interpolated_maps, metadata
