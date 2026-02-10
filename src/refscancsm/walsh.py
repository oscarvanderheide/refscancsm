import numpy as np
from tqdm import tqdm

from scipy.ndimage import filters

def walsh_csm(img, smoothing=5, niter=10, use_mask=True):
    '''Calculates the coil sensitivities for 2D or 3D data using an iterative version of the Walsh method

    :param img: Input images, ``[coil, y, x]`` or ``[coil, z, y, x]``
    :param smoothing: Smoothing block size (default ``5``)
    :parma niter: Number of iterations for the eigenvector power method (default ``10``)
    :param use_mask: Skip power iterations for empty voxels (default ``True``)

    :returns csm: Relative coil sensitivity maps, ``[coil, y, x]`` or ``[coil, z, y, x]``
    :returns rho: Total power in the estimated coils maps, ``[y, x]`` or ``[z, y, x]``

    Taken from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/coils.py
    '''

    print("Calculating coil sensitivity maps using Walsh method...")
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
    print("Smoothing coil images...")
    img_smooth = np.zeros_like(img_reshape)
    for p in range(ncoils):
        img_smooth[p] = smooth(img_reshape[p], smoothing)

    # Vectorized power method without storing covariance matrices
    # Reshape to [ncoils, nvoxels] for vectorized operations
    img_flat = img_smooth.reshape(ncoils, -1)  # [ncoils, nz*ny*nx]
    
    # Create mask to skip empty voxels (e.g., air in MRI)
    if use_mask:
        print("Creating mask to skip empty voxels...")
        signal_strength = np.sum(np.abs(img_flat)**2, axis=0)  # [nvoxels]
        threshold = 0.005 * np.max(signal_strength)  # 1% of max signal
        mask = signal_strength > threshold
        print(f"Processing {np.sum(mask)} / {len(mask)} voxels ({100*np.sum(mask)/len(mask):.1f}%)")
        img_masked = img_flat[:, mask]  # [ncoils, n_masked_voxels]
    else:
        mask = None
        img_masked = img_flat
    
    # Initialize eigenvectors: v = conj(img) * sum(img)
    # This is equivalent to sum(R, axis=0) where R[i,j] = img[i] * conj(img[j])
    v = np.conj(img_masked) * np.sum(img_masked, axis=0, keepdims=True)  # [ncoils, n_masked_voxels]
    
    # Normalize
    v_norms = np.linalg.norm(v, axis=0, keepdims=True)  # [1, n_masked_voxels]
    v = np.where(v_norms > 0, v / v_norms, 0)  # [ncoils, n_masked_voxels]
    
    # Power method iterations (vectorized, memory efficient)
    # R @ v = (img ⊗ conj(img)) @ v = img * (conj(img).H @ v)
    for iter in tqdm(range(niter)):
        # Compute inner product: conj(img).H @ v = sum over coils
        inner = np.sum(np.conj(img_masked) * v, axis=0, keepdims=True)  # [1, n_masked_voxels]
        # Compute R @ v = img * inner
        v = img_masked * inner  # [ncoils, n_masked_voxels]
        # Normalize
        v_norms = np.linalg.norm(v, axis=0, keepdims=True)  # [1, n_masked_voxels]
        v = np.where(v_norms > 0, v / v_norms, 0)  # [ncoils, n_masked_voxels]
    
    # Reconstruct full result, with zeros for masked-out voxels
    if use_mask:
        v_full = np.zeros((ncoils, img_flat.shape[1]), dtype=img.dtype)
        v_full[:, mask] = v
        v = v_full
    
    # Reshape back to original dimensions
    csm = v.reshape(ncoils, nz, ny, nx)  # [ncoils, nz, ny, nx]

    # Remove singleton dimension for 2D case
    if img.ndim == 3:
        csm = csm[:, 0, :, :]

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