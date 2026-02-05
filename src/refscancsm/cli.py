"""Command-line interface for refscancsm."""

import argparse
import sys
import numpy as np
from pathlib import Path
from .workflow import get_target_csm


def main():
    """Main CLI entry point for refscancsm."""
    parser = argparse.ArgumentParser(
        prog="refscancsm",
        description="Interpolate coil sensitivity maps from reference scan to target scan geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with .npy output
  refscancsm refscan.cpx refscan.sin target.sin -o coil_maps.npy
  
  # With detailed output information
  refscancsm refscan.cpx refscan.sin target.sin -o coil_maps.npy -v
  
  # Use cubic interpolation
  refscancsm refscan.cpx refscan.sin target.sin -o coil_maps.npy --interp-order 3
""",
    )

    parser.add_argument(
        "refscan_cpx",
        type=str,
        help="Path to reference scan CPX file (with or without .cpx extension)",
    )

    parser.add_argument(
        "refscan_sin",
        type=str,
        help="Path to reference scan SIN file",
    )

    parser.add_argument(
        "target_sin",
        type=str,
        help="Path to target scan SIN file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: coil_maps_interpolated.npy)",
    )

    parser.add_argument(
        "--location-idx",
        type=int,
        default=1,
        help="Location index to extract from SIN files (default: 1)",
    )

    parser.add_argument(
        "--interp-order",
        type=int,
        default=1,
        choices=[0, 1, 3],
        help="Interpolation order: 0=nearest, 1=linear (default), 3=cubic",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information about the data",
    )

    args = parser.parse_args()

    # Process file paths
    refscan_cpx = args.refscan_cpx

    # Remove .cpx extension if provided (readCpx adds it automatically)
    if refscan_cpx.endswith(".cpx") or refscan_cpx.endswith(".CPX"):
        refscan_cpx = refscan_cpx[:-4]

    try:
        coil_maps, metadata = get_target_csm(
            refscan_cpx,
            args.refscan_sin,
            args.target_sin,
            location_idx=args.location_idx,
            interpolation_order=args.interp_order,
        )
    except Exception as e:
        print(f"\n✗ Error during interpolation: {e}", file=sys.stderr)
        import traceback

        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    # Display summary
    if args.verbose:
        print("Detailed Metadata:")
        print(f"  Number of coils: {metadata['ncoils']}")
        print(f"  Reference shape: {metadata['reference_shape']}")
        print(f"  Reference voxel sizes: {metadata['reference_voxel_sizes']} mm")
        print(f"  Target shape: {metadata['target_shape']}")
        print(f"  Target voxel sizes: {metadata['target_voxel_sizes']} mm")
        print(f"  Interpolation order: {metadata['interpolation_order']}")

    # Save data if requested
    if not args.no_save:
        if args.output:
            output_file = args.output
        else:
            # Default output name
            if args.matlab:
                output_file = "coil_maps_interpolated.mat"
            else:
                output_file = "coil_maps_interpolated.npy"

        try:
            if args.matlab or output_file.endswith(".mat"):
                # Save as MATLAB file
                try:
                    from scipy.io import savemat
                except ImportError:
                    print(
                        "\n✗ Error: scipy.io not available for MATLAB output",
                        file=sys.stderr,
                    )
                    print("   Install with: uv pip install scipy", file=sys.stderr)
                    sys.exit(1)

                # Prepare data for MATLAB (transpose to match MATLAB convention if needed)
                mat_data = {
                    "coil_maps": coil_maps,
                    "metadata": {
                        "ncoils": metadata["ncoils"],
                        "target_shape": metadata["target_shape"],
                        "target_voxel_sizes": metadata["target_voxel_sizes"],
                        "reference_shape": metadata["reference_shape"],
                        "reference_voxel_sizes": metadata["reference_voxel_sizes"],
                    },
                }
                savemat(output_file, mat_data)
                print(f"\n✓ Coil maps saved to MATLAB file: {output_file}")
            else:
                # Save as numpy file
                np.save(output_file, coil_maps)
                print(f"\n✓ Coil maps saved to: {output_file}")

                # Also save metadata
                metadata_file = Path(output_file).stem + "_metadata.npy"
                np.save(metadata_file, metadata, allow_pickle=True)
                print(f"✓ Metadata saved to: {metadata_file}")

        except Exception as e:
            print(f"\n✗ Error saving data: {e}", file=sys.stderr)
            sys.exit(1)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
