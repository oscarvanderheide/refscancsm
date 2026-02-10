import numpy as np
from scipy.ndimage import filters

def walsh_csm(img, smoothing=5, niter=10):
    '''Calculates the coil sensitivities for 2D or 3D data using an iterative version of the Walsh method

    :param img: Input images, ``[coil, y, x]`` or ``[coil, z, y, x]``
    :param smoothing: Smoothing block size (default ``5``)
    :parma niter: Number of iterations for the eigenvector power method (default ``10``)

    :returns csm: Relative coil sensitivity maps, ``[coil, y, x]`` or ``[coil, z, y, x]``
    :returns rho: Total power in the estimated coils maps, ``[y, x]`` or ``[z, y, x]``

    Taken from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/coils.py
    '''

    assert img.ndim in [3, 4], "Coil sensitivity map must have 3 (2D) or 4 (3D) dimensions"

    ncoils = img.shape[0]
    
    if img.ndim == 3:
        # 2D case: [coil, y, x]
        ny = img.shape[1]
        nx = img.shape[2]
        nz = 1
        img_reshape = img[:, np.newaxis, :, :]  # [coil, 1, y, x]
    else:
        # 3D case: [coil, z, y, x]
        nz = img.shape[1]
        ny = img.shape[2]
        nx = img.shape[3]
        img_reshape = img

    # Smooth each coil image first (more memory efficient than smoothing covariance)
    img_smooth = np.zeros_like(img_reshape)
    for p in range(ncoils):
        img_smooth[p] = smooth(img_reshape[p], smoothing)

    # At each point in the image, compute covariance matrix on-the-fly,
    # then find the dominant eigenvector and corresponding eigenvalue
    # using the power method
    rho = np.zeros((nz, ny, nx))
    csm = np.zeros((ncoils, nz, ny, nx), dtype=img.dtype)
    
    for z in range(nz):
        print(f"Processing slice {z + 1}/{nz}...")
        for y in range(ny):
            for x in range(nx):
                # Compute covariance matrix for this voxel
                img_voxel = img_smooth[:, z, y, x]  # [ncoils]
                R = np.outer(img_voxel, np.conj(img_voxel))  # [ncoils, ncoils]
                
                # Power method to find dominant eigenvector
                v = np.sum(R, axis=0)
                lam = np.linalg.norm(v)
                if lam > 0:
                    v = v / lam
                else:
                    v = np.zeros(ncoils, dtype=img.dtype)
                    lam = 0

                for iter in range(niter):
                    v = np.dot(R, v)
                    lam = np.linalg.norm(v)
                    if lam > 0:
                        v = v / lam

                rho[z, y, x] = lam
                csm[:, z, y, x] = v

    # Remove singleton dimension for 2D case
    if img.ndim == 3:
        csm = csm[:, 0, :, :]
        rho = rho[0, :, :]

    return csm

def smooth(img, box=5):
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``

    Taken from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/coils.py
    '''

    t_real = np.zeros(img.shape)
    t_imag = np.zeros(img.shape)

    filters.uniform_filter(img.real,size=box,output=t_real)
    filters.uniform_filter(img.imag,size=box,output=t_imag)

    simg = t_real + 1j*t_imag

    return simg