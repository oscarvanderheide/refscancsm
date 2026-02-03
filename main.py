import numpy as np
import os
import struct
import sys


def oset(seq):
    """Order-preserving unique set."""
    seen = {}
    result = []
    for item in seq:
        if isinstance(item, (tuple, list)):
            marker = tuple(item)
        else:
            marker = item
        if marker not in seen:
            seen[marker] = 1
            result.append(item)
    return result


def filename_extcase(fn):
    """Find correct case-sensitive filename."""
    if os.path.exists(fn):
        return fn
    pn, ext = os.path.splitext(fn)
    bn = os.path.basename(pn)
    dir_path = os.path.dirname(fn) or "."
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            if f.lower() == (bn + ext).lower():
                return os.path.join(dir_path, f)
    return ""


def readCpx(filename):
    """
    Parse and read a .cpx file.
    """

    filename = filename_extcase(filename + ".cpx")

    # open the cpx file
    if type(filename) is not str:
        print("Input filename is not a string.")
        sys.exit(1)

    if os.path.splitext(filename)[1] not in [".cpx", ".CPX"]:
        print("input filename is not a .cpx file")
        sys.exit(1)

    # opens the file
    try:
        fil = open(filename, "rb")
    except IOError:
        print("cannot open .cpx file ", filename)
        sys.exit(1)

    hdr = dict()

    index = 0
    offset = 0
    # pre-allocate info array
    info = np.zeros(
        [10000, 128]
    )  # increased to store matrix_data_blocks and data_offset
    fmt = "<" + "15i2f111i"  # MATLAB format: 15 long + 2 float + 111 long = 512 bytes

    # Read first header to get matrix_data_blocks and data_offset
    fil.seek(0)
    line = fil.read(512)
    header_vals = struct.unpack("<15i2f111i", line)
    h1 = header_vals[0:15]
    h2 = header_vals[17:128]
    matrix_data_blocks = h1[12]  # h1(13) in MATLAB = index 12 in Python
    data_offset = (
        h2[25] if h2[25] != 0 else h1[9]
    )  # h2(26) in MATLAB = index 25 in Python, fallback to h1(10)

    # generate hdr table
    while True:
        fil.seek(offset)
        line = fil.read(512)

        # make sure line exists
        if len(line) == 512:
            h1_vals = struct.unpack("<15i", line[0:60])
            factor_vals = struct.unpack("<2f", line[60:68])
            h2_vals = struct.unpack("<111i", line[68:512])

            # Store in info array: h1 (15) + factors (2) + h2 (111) = 128 values
            info[index, 0:15] = h1_vals
            info[index, 15:17] = factor_vals
            info[index, 17:128] = h2_vals

            # Check if image exists (h1[8] = Complex Matrix Existence)
            if h1_vals[8] == 0:
                break
        else:
            break

        index += 1
        # Use MATLAB's offset calculation with matrix_data_blocks
        offset = (matrix_data_blocks * 512 + data_offset) * index

        # put info into dictionary
        key = "hdr_" + str(index)
        hdr[key] = info[index, :]
    hdr["headerType"] = "cpx"

    # truncate info array
    info = info[:index, :]

    num_images = index

    # pre-allocate data array
    mixes = oset(info[:, 0])  # h1[0] = mix
    locs = oset(info[:, 1])  # h1[1] = stack
    slices = oset(info[:, 2])  # h1[2] = slice
    echoes = oset(info[:, 4])  # h1[4] = echo
    phases = oset(info[:, 5])  # h1[5] = heart phase
    dynamics = oset(info[:, 6])  # h1[6] = dynamics
    rows = oset(info[:, 7])  # h1[7] = segments
    x_size = oset(info[:, 10])  # h1[10] = resolution x
    y_size = oset(info[:, 11])  # h1[11] = resolution y
    coils = oset(info[:, 18])  # h2[1] = coil (h2 starts at index 17)

    nmix = len(mixes)
    nloc = len(locs)
    nslice = len(slices)
    necho = len(echoes)
    ncard = len(phases)
    ndyn = len(dynamics)
    nrow = len(rows)
    nx = int(np.max(x_size))
    ny = int(np.max(y_size))
    nchan = len(coils)

    data_string = np.array(
        ["chan", "mix", "dyn", "card", "echo", "row", "loc", "slice", "y", "x"]
    )
    data = np.zeros(
        [nchan, nmix, ndyn, ncard, necho, nrow, nloc, nslice, ny, nx],
        dtype=np.complex64,
    )

    # read in the cpx file
    for index in range(num_images):
        # Calculate offset using MATLAB's method with matrix_data_blocks
        offset = (matrix_data_blocks * 512 + data_offset) * index + data_offset
        fil.seek(offset)

        mix = mixes.index(info[index, 0])
        loc = locs.index(info[index, 1])
        slice_idx = slices.index(info[index, 2])
        echo = echoes.index(info[index, 4])
        card = phases.index(info[index, 5])
        dyn = dynamics.index(info[index, 6])
        row = rows.index(info[index, 7])
        coil = coils.index(info[index, 18])  # h2[1] at index 18
        nx = int(info[index, 10])
        ny = int(info[index, 11])
        compression_factor = info[index, 13]
        size_bytes = int(8 * nx * ny // compression_factor)

        unparsed_data = fil.read(size_bytes)
        if compression_factor == 1:
            temp_data = np.frombuffer(unparsed_data, dtype=np.float32)
        elif compression_factor == 2:
            temp_data = np.frombuffer(unparsed_data, dtype=np.int16)
        elif compression_factor == 4:
            temp_data = np.frombuffer(unparsed_data, dtype=np.int8)
        temp_data.shape = [ny, nx, 2]
        complex_data = temp_data[:, :, 0] + 1j * temp_data[:, :, 1]
        data[coil, mix, dyn, card, echo, row, loc, slice_idx, :, :] = complex_data

    # setup the data labels
    data_labels = data_string[
        (np.array(data.shape[0 : len(data_string)]) > 1).nonzero()[0]
    ]

    return (data, hdr, data_labels)


if __name__ == "__main__":
    """
    Test the CPX reader on the files from load_test_cpx_files.m
    """
    import sys

    if len(sys.argv) > 1:
        # Read single file from command line
        cpx_file = sys.argv[1]
        print(f"\n{'=' * 70}")
        print(f"Reading: {cpx_file}")
        print("=" * 70)
        data, hdr, labels = readCpx(cpx_file)
        # data, hdr, labels = readCpxOriginal(cpx_file, 0, 0)
        print(f"\nFinal shape: {data.shape}")
        print(f"Labels: {labels}")
        print(f"Data shape: {data.shape}")
        print(f"Receive coil data shape: {data[..., 0].shape}")
        print(f"Body coil data shape: {data[:, :, :, 0, ..., 1].shape}")

        # save data to npy
        npy_file = "cpx_test.npy"
        np.save(npy_file, data)
        print(f"Data saved to: {npy_file}")
    else:
        # Test on the files listed in load_test_cpx_files.m
        test_files = [
            "/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/studies/parkinson/data/patients/MRSTAT_MRPARK_110187/GTpackNgo/mr_25072023_2133020_1000_9_wip_coilsurveyscanV4.cpx",
            "/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/studies/tempo/prototypes/data/24_067_UMCU/GTpackGO/Clinic/2024_11_28/HI_240721/hi_28112024_1433175_1000_1_coilsurveyscanV4.cpx",
            "/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/experiments/fei/20220330_MR21_Tube11Tube8/ov_30032022_1659243_1000_3_wip_coilsurveyscanV4.cpx",
            "/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/experiments/fei/20241205_phantomtest_saverawdata/ph_05122024_1652180_1000_2_coilsurveyscanV4.cpx",
            "/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/experiments/fei/20230713_MR22_tubesSeqTEST/20_13072023_1640257_1000_3_wip_coilsurveyscanV4.cpx",
        ]

        expected_sizes = [
            [48, 32, 47, 44, 1, 1, 1, 2],
            [48, 32, 47, 44, 1, 1, 1, 2],
            [48, 32, 47, 27, 1, 1, 1, 2],
            [48, 32, 47, 27, 1, 1, 1, 2],
            [48, 32, 47, 27, 1, 1, 1, 2],
        ]

        print("\nTesting CPX reader on all test files...")
        print("=" * 70)

        all_match = True
        for i, cpx_file in enumerate(test_files):
            if not os.path.exists(cpx_file):
                print(f"\nFile {i + 1}: SKIPPED (not found)")
                print(f"  {cpx_file}")
                continue

            print(f"\n{'=' * 70}")
            print(f"File {i + 1}: {os.path.basename(cpx_file)}")
            print("=" * 70)

            try:
                data, hdr, labels = readCpx(cpx_file)

                expected = expected_sizes[i]
                matlab_shape = list(data.shape)

                print(f"\nExpected (MATLAB): {expected}")
                print(f"Got (Python):      {matlab_shape}")

                # Check if shapes match
                if matlab_shape == expected:
                    print("✓ PERFECT MATCH!")
                else:
                    print("✗ MISMATCH")
                    all_match = False

            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback

                traceback.print_exc()
                all_match = False

        print("\n" + "=" * 70)
        if all_match:
            print("SUCCESS: All files match MATLAB dimensions!")
        else:
            print("PARTIAL: Some files don't match")
        print("=" * 70)
