"""Command-line interface for refscan2csm."""

import argparse
import sys
import numpy as np
from pathlib import Path
from .reader import readCpx


def main():
    """Main CLI entry point for refscan2csm."""
    parser = argparse.ArgumentParser(
        prog="refscan2csm",
        description="Read and convert CPX files to CSM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "cpx_file",
        type=str,
        help="Path to the .cpx file to read (with or without .cpx extension)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path for saving data (default: cpx_output.npy)",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output file, just display information",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information about the data",
    )

    args = parser.parse_args()

    # Process the CPX file
    cpx_file = args.cpx_file

    # Remove .cpx extension if provided (readCpx adds it automatically)
    if cpx_file.endswith(".cpx") or cpx_file.endswith(".CPX"):
        cpx_file = cpx_file[:-4]

    print(f"\n{'=' * 70}")
    print(f"Reading: {cpx_file}.cpx")
    print("=" * 70)

    try:
        data, hdr, labels = readCpx(cpx_file)
    except Exception as e:
        print(f"Error reading CPX file: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n✓ Successfully read CPX file")
    print(f"  Shape: {data.shape}")
    print(f"  Labels: {labels}")
    print(f"  Data type: {data.dtype}")

    if args.verbose:
        print("\nDetailed information:")
        print(f"  Number of channels: {data.shape[0]}")
        if len(data.shape) > 1:
            print(f"  Dimensions: {' × '.join(map(str, data.shape))}")
        print(
            f"  Number of headers: {len([k for k in hdr.keys() if k.startswith('hdr_')])}"
        )
        print(f"  Header type: {hdr.get('headerType', 'unknown')}")

    # Save data if requested
    if not args.no_save:
        if args.output:
            output_file = args.output
        else:
            # Default output name
            input_path = Path(cpx_file)
            output_file = input_path.stem + "_output.npy"

        try:
            np.save(output_file, data)
            print(f"\n✓ Data saved to: {output_file}")
        except Exception as e:
            print(f"\n✗ Error saving data: {e}", file=sys.stderr)
            sys.exit(1)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
