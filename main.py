import os
import numpy as np
import struct
import sys


def filename_extcase(fn):
    pn, ext = os.path.splitext(fn)
    bn = os.path.basename(pn)
    dirname = os.path.dirname(fn) or '.'
    if bn+ext.lower() in os.listdir(dirname):
        return pn+ext.lower()
    elif bn+ext.upper() in os.listdir(dirname):
        return pn+ext.upper()
    return ''


# order preserving set generation
# pulled off the web (http://www.peterbe.com/plog/uniqifiers-benchmark)
def oset(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x):
            if isinstance(x, (tuple, list)):
                return tuple(x)
            else:
                return(x)
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen.keys():
            continue
        seen[marker] = 1
        result.append(item)
    return result


# sortd version of above function
def oset_sorted(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x):
            if isinstance(x, (tuple, list)):
                return tuple(x)
            else:
                return(x)
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen.keys():
            continue
        seen[marker] = 1
        result.append(item)
    result = sorted(result, key=lambda x: float(x))
    return result

def readCpx(filename, cur_coil, cur_loc):
    '''
        Parse and read a .cpx file.
    '''

    # Only add .cpx extension if the filename doesn't already have an extension
    if not os.path.splitext(filename)[1]:
        filename = filename_extcase(filename+'.cpx')
    else:
        filename = filename_extcase(filename)

    # open the cpx file
    if not type(filename) is str or not filename:
        print(f"Input filename is not valid: {filename}")
        sys.exit(1)

    if os.path.splitext(filename)[1] not in ['.cpx', '.CPX']:
        print("input filename is not a .cpx file")
        sys.exit(1)

    # opens the file
    try:
        fil = open(filename, 'rb')
    except IOError:
        print('cannot open .cpx file ', filename)
        sys.exit(1)

    hdr = dict()

    index = 0
    offset = 0
    # pre-allocate info array
    info = np.zeros([10000, 127])
    fmt = '<'+'15i2f15i10f1q84i'
    # generate hdr table
    while True:
        fil.seek(offset)
        line = fil.read(512)

        # make sure line exists
        if len(line) == 512:
            info[index, :] = struct.unpack(fmt, line)
        else:
            break

        # calculate position of next header
        nx = info[index, 10]
        ny = info[index, 11]
        compression_factor = info[index, 13]

        if compression_factor == 0:
            break
        else:
            index += 1
            offset += 512 + int(nx*ny*8 // compression_factor)

        # put info into dictionary
        key = 'hdr_'+str(index)
        hdr[key] = info[index, :]
    hdr['headerType'] = 'cpx'

    # truncate info array
    info = info[:index, :]

    num_images = index

    # pre-allocate data array
    mixes = oset(info[:, 0])
    locs = oset(info[:, 1])
    echoes = oset(info[:, 4])
    phases = oset(info[:, 5])
    dynamics = oset(info[:, 6])
    rows = oset(info[:, 7])
    x_size = oset(info[:, 10])
    y_size = oset(info[:, 11])
    coils = oset(info[:, 21])

    nmix = len(mixes)
    nloc = len(locs)
    necho = len(echoes)
    ncard = len(phases)
    ndyn = len(dynamics)
    nrow = len(rows)
    nx = int(np.max(x_size))
    ny = int(np.max(y_size))
    nchan = len(coils)

    data_string = np.array(['chan', 'mix', 'dyn', 'card', 'echo', 'row',
                            'loc', 'y', 'x'])
    data = np.zeros([nchan, nmix, ndyn, ncard, necho, nrow, nloc, ny, nx],
                    dtype=np.complex64)

    # read in the cpx file
    offset = 512
    for index in range(num_images):

        fil.seek(offset)

        mix = mixes.index(info[index, 0])
        loc = locs.index(info[index, 1])
        echo = echoes.index(info[index, 4])
        card = phases.index(info[index, 5])
        dyn = dynamics.index(info[index, 6])
        row = rows.index(info[index, 7])
        coil = coils.index(info[index, 21])
        nx = int(info[index, 10])
        ny = int(info[index, 11])
        compression_factor = info[index, 13]
        size_bytes = int(8 * nx * ny // compression_factor)
        offset += 512 + size_bytes

        unparsed_data = fil.read(size_bytes)
        if compression_factor == 1:
            temp_data = np.frombuffer(unparsed_data, dtype=np.float32)
        elif compression_factor == 2:
            temp_data = np.frombuffer(unparsed_data, dtype=np.int16)
        elif compression_factor == 4:
            temp_data = np.frombuffer(unparsed_data, dtype=np.int8)
        temp_data.shape = [ny, nx, 2]
        complex_data = temp_data[:, :, 0] + 1j*temp_data[:, :, 1]
        data[coil, mix, dyn, card, echo, row, loc, :, :] = complex_data

    # setup the data labels
    data_labels = data_string[(np.array(
        data.shape[0:len(data_string)]) > 1).nonzero()[0]]

    return (data, hdr, data_labels)

if __name__ == '__main__':
    """
    Test the CPX reader on the files from load_test_cpx_files.m
    """
    import sys
    
    if len(sys.argv) > 1:
        # Read single file from command line
        cpx_file = sys.argv[1]
        print(f"\n{'='*70}")
        print(f"Reading: {cpx_file}")
        print('='*70)
        data, hdr, labels = readCpx(cpx_file, 0, 0)
        print(f"\nFinal shape: {data.shape}")
        print(f"Labels: {labels}")
    else:
        # Test on the files listed in load_test_cpx_files.m
        test_files = [
            '/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/studies/parkinson/data/patients/MRSTAT_MRPARK_110187/GTpackNgo/mr_25072023_2133020_1000_9_wip_coilsurveyscanV4.cpx',
            '/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/studies/tempo/prototypes/data/24_067_UMCU/GTpackGO/Clinic/2024_11_28/HI_240721/hi_28112024_1433175_1000_1_coilsurveyscanV4.cpx',
            '/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/experiments/fei/20220330_MR21_Tube11Tube8/ov_30032022_1659243_1000_3_wip_coilsurveyscanV4.cpx',
            '/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/experiments/fei/20241205_phantomtest_saverawdata/ph_05122024_1652180_1000_2_coilsurveyscanV4.cpx',
            '/smb/user/oheide/DS-Data/Radiotherapie/Research/Project/MRSTAT/experiments/fei/20230713_MR22_tubesSeqTEST/20_13072023_1640257_1000_3_wip_coilsurveyscanV4.cpx'
        ]
        
        expected_sizes = [
            [48, 32, 47, 44, 1, 1, 1, 2],
            [48, 32, 47, 44, 1, 1, 1, 2],
            [48, 32, 47, 27, 1, 1, 1, 2],
            [48, 32, 47, 27, 1, 1, 1, 2],
            [48, 32, 47, 27, 1, 1, 1, 2]
        ]
        
        print("\nTesting CPX reader on all test files...")
        print("="*70)
        
        for i, cpx_file in enumerate(test_files):
            if not os.path.exists(cpx_file):
                print(f"\nFile {i+1}: SKIPPED (not found)")
                print(f"  {cpx_file}")
                continue
                
            print(f"\n{'='*70}")
            print(f"File {i+1}: {os.path.basename(cpx_file)}")
            print('='*70)
            
            try:
                data, hdr, labels = readCpx(cpx_file, 0, 0)
                
                # Get only non-singleton dimensions for comparison
                shape_nonsingleton = [s for s in data.shape if s > 1]
                expected = expected_sizes[i]
                
                # MATLAB includes trailing singleton dimensions, so compare carefully
                matlab_shape = list(data.shape)
                
                print(f"\nExpected (MATLAB): {expected}")
                print(f"Got (Python):      {matlab_shape}")
                
                # Check if shapes match
                if matlab_shape == expected:
                    print("✓ MATCH!")
                else:
                    print("✗ MISMATCH")
                    print(f"  Non-singleton dims: {shape_nonsingleton}")
                    
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*70)
        print("Testing complete")
        print("="*70)
