"""Command-line interface for refscancsm."""

import argparse
import sys
import numpy as np
from pathlib import Path
from .get_csm import get_csm


def main():
    """Main CLI entry point for get_csm."""
    parser = argparse.ArgumentParser(
        prog="get_csm",
        description="Interpolate coil sensitivity maps from reference scan to target scan geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Basic usage with .npy output
  get_csm refscan.cpx refscan.sin target.sin -o csm.npy
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
        help="Output file path (default: csm.npy)",
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

    parser.add_argument(
        "--squeeze",
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=True,
        metavar="BOOL",
        help="Squeeze singleton dimensions from CPX data (default: true)",
    )

    args = parser.parse_args()

    # Process file paths
    refscan_cpx = args.refscan_cpx

    # Remove .cpx extension if provided (readCpx adds it automatically)
    if refscan_cpx.endswith(".cpx") or refscan_cpx.endswith(".CPX"):
        refscan_cpx = refscan_cpx[:-4]

    try:
        coil_maps = get_csm(
            refscan_cpx,
            args.refscan_sin,
            args.target_sin,
            location_idx=args.location_idx,
            interpolation_order=args.interp_order,
            verbose=args.verbose,
            squeeze=args.squeeze,
        )
    except Exception as e:
        print(f"\n✗ Error during interpolation: {e}", file=sys.stderr)
        import traceback

        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = "csm.npy"

    # Save data
    try:
        if output_file.endswith(".mat"):
            # Save as MATLAB file
            try:
                from scipy.io import savemat
            except ImportError:
                print(
                    "\nError: scipy.io not available for MATLAB output",
                    file=sys.stderr,
                )
                print("   Install with: uv pip install scipy", file=sys.stderr)
                sys.exit(1)

            mat_data = {"csm": coil_maps}
            savemat(output_file, mat_data)
            print(f"\nCoil maps saved to MATLAB file: {output_file}")
        elif output_file.endswith(".npy"):
            # Save as numpy file
            np.save(output_file, coil_maps)
            print(f"\nCoil maps saved to: {output_file}")
        else:
            print(
                f"\nError: Unsupported output format. Use .npy or .mat extension.",
                file=sys.stderr,
            )
            sys.exit(1)

    except Exception as e:
        print(f"\nError saving data: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
