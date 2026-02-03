"""
Example usage of the coilsurvey_cpx_to_csm package.

This script demonstrates how to:
1. Interpolate coil sensitivity maps from reference scan to target scan
2. Use the lower-level API to read individual files
3. Save results in different formats
"""

import numpy as np
from coilsurvey_cpx_to_csm import (
    interpolate_coil_maps,
    readCpx,
    read_location_matrix,
    read_voxel_sizes,
    read_matrix_size,
    transform_to_MPS_refscan,
    transform_to_MPS_target,
)


def example_full_workflow():
    """Example: Complete workflow to interpolate coil maps."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Full workflow - Interpolate coil maps")
    print("="*70 + "\n")
    
    # Interpolate coil sensitivity maps from reference to target geometry
    coil_maps, metadata = interpolate_coil_maps(
        refscan_cpx_path="path/to/refscan",  # without .cpx extension
        refscan_sin_path="path/to/refscan.sin",
        target_sin_path="path/to/target.sin",
        location_idx=1,
        interpolation_order=1,  # 0=nearest, 1=linear (default), 3=cubic
    )
    
    print(f"\nResults:")
    print(f"  Interpolated shape: {coil_maps.shape}")  # [ncoils, nz, ny, nx]
    print(f"  Number of coils: {metadata['ncoils']}")
    print(f"  Target voxel sizes: {metadata['target_voxel_sizes']} mm")
    print(f"  Target matrix size: {metadata['target_matrix_size']}")
    
    # Save the result
    np.save("coil_maps_interpolated.npy", coil_maps)
    print(f"\n✓ Saved to: coil_maps_interpolated.npy")
    
    # Optionally save as MATLAB file
    try:
        from scipy.io import savemat
        savemat("coil_maps_interpolated.mat", {
            'coil_maps': coil_maps,
            'metadata': {
                'ncoils': metadata['ncoils'],
                'target_shape': metadata['target_shape'],
                'target_voxel_sizes': metadata['target_voxel_sizes'],
            }
        })
        print(f"✓ Saved to: coil_maps_interpolated.mat")
    except ImportError:
        print("  (scipy.io not available for MATLAB output)")


def example_read_individual_files():
    """Example: Read individual files using lower-level API."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Read individual files")
    print("="*70 + "\n")
    
    # Read CPX file
    print("1. Reading CPX file...")
    data, hdr, labels = readCpx("path/to/refscan")
    print(f"   CPX data shape: {data.shape}")
    print(f"   Data labels: {labels}")
    
    # Read SIN file geometry
    print("\n2. Reading SIN file geometry...")
    loc_matrix = read_location_matrix("path/to/refscan.sin", location_idx=1)
    voxel_sizes = read_voxel_sizes("path/to/refscan.sin")
    matrix_size = read_matrix_size("path/to/refscan.sin")
    
    print(f"   Location matrix shape: {loc_matrix.shape}")
    print(f"   Voxel sizes: {voxel_sizes} mm")
    print(f"   Matrix size: {matrix_size}")
    
    # Transform to MPS coordinate system
    print("\n3. Creating transformation matrices...")
    T_MPS_ref = transform_to_MPS_refscan(loc_matrix)
    print(f"   MPS transformation matrix shape: {T_MPS_ref.shape}")
    print(f"   Transformation matrix:\n{T_MPS_ref}")


def example_different_interpolation_orders():
    """Example: Compare different interpolation methods."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Different interpolation orders")
    print("="*70 + "\n")
    
    for order, name in [(0, "nearest"), (1, "linear"), (3, "cubic")]:
        print(f"\nInterpolation order {order} ({name}):")
        
        coil_maps, metadata = interpolate_coil_maps(
            refscan_cpx_path="path/to/refscan",
            refscan_sin_path="path/to/refscan.sin",
            target_sin_path="path/to/target.sin",
            interpolation_order=order,
        )
        
        print(f"  ✓ Shape: {coil_maps.shape}")
        print(f"  ✓ Data range: [{coil_maps.real.min():.3f}, {coil_maps.real.max():.3f}]")
        
        # Save with descriptive name
        np.save(f"coil_maps_order{order}_{name}.npy", coil_maps)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  Coilsurvey CPX to CSM - Example Usage                              ║
╚══════════════════════════════════════════════════════════════════════╝

Note: These examples use placeholder paths. Replace with your actual file paths.
""")
    
    # Uncomment to run examples:
    # example_full_workflow()
    # example_read_individual_files()
    # example_different_interpolation_orders()
    
    print("\nTo run these examples:")
    print("  1. Edit the file paths in the examples above")
    print("  2. Uncomment the example you want to run")
    print("  3. Run: uv run python examples.py")
    print()
