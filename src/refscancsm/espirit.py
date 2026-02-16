#!/usr/bin/env python3
"""
Python implementation of ESPIRiT (Eigenvalue-based Self-consistent Parallel Imaging Reconstruction)

This is a pure Python translation of the BART ecalib.c implementation.
Based on: Uecker et al., "ESPIRiT - An Eigenvalue Approach to Autocalibrating
Parallel MRI: Where SENSE meets GRAPPA", MRM 2014.

Main steps:
1. Extract calibration region from k-space
2. Build calibration matrix and compute null-space kernels via SVD
3. Transform kernels to image domain
4. Compute image-space covariance matrices
5. Point-wise eigen-decomposition for sensitivity maps
6. Post-processing (crop, phase rotation, normalization)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import linalg

# Match BART's single precision (complex float). Set to np.complex128 for double.
DEFAULT_DTYPE = np.complex64

# Number of threads for parallelizing per-voxel eigendecomposition.
# Set to 0 to auto-detect (uses cpu_count).
NUM_THREADS = 0


def _get_num_threads():
    """Get the number of threads to use for parallel eigenmaps."""
    if NUM_THREADS > 0:
        return NUM_THREADS
    return min(os.cpu_count() or 1, 16)


def fftNc(x, axes=None):
    """Centered N-dimensional FFT"""
    if axes is None:
        axes = tuple(range(-len(x.shape), 0))  # All axes
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes
    )


def ifftNc(x, axes=None):
    """Centered N-dimensional IFFT"""
    if axes is None:
        axes = tuple(range(-len(x.shape), 0))  # All axes
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes
    )


# Convenience aliases
def fft2c(x):
    """Centered 2D FFT"""
    return fftNc(x, axes=(-2, -1))


def ifft2c(x):
    """Centered 2D IFFT"""
    return ifftNc(x, axes=(-2, -1))


def fft3c(x):
    """Centered 3D FFT"""
    return fftNc(x, axes=(-3, -2, -1))


def ifft3c(x):
    """Centered 3D IFFT"""
    return ifftNc(x, axes=(-3, -2, -1))


def sinc_interp_1d(data, target_size, axis):
    """
    Sinc interpolation along one axis via FFT zero-padding.

    Matches BART's sinc_resize: centered FFT → zero-pad → centered IFFT.
    Processes one axis at a time (BART's sinc_zeropad calls sinc_resize
    recursively, one dimension per call).

    All non-interpolated axes are treated as batch dimensions.

    Parameters:
    -----------
    data : ndarray
        Input data (arbitrary shape)
    target_size : int
        Target size along the specified axis
    axis : int
        Axis to interpolate along

    Returns:
    --------
    result : ndarray
        Interpolated data with data.shape[axis] replaced by target_size
    """
    source_size = data.shape[axis]
    if source_size == target_size:
        return data.copy()

    # Centered FFT along axis
    tmp = np.fft.ifftshift(data, axes=axis)
    tmp = np.fft.fft(tmp, axis=axis)
    tmp = np.fft.fftshift(tmp, axes=axis)

    # Zero-pad or crop (centered) along axis
    new_shape = list(tmp.shape)
    new_shape[axis] = target_size

    if target_size > source_size:
        # Upsampling: embed source spectrum in center of target
        padded = np.zeros(new_shape, dtype=tmp.dtype)
        start = (target_size - source_size) // 2
        dst_sl = [slice(None)] * tmp.ndim
        dst_sl[axis] = slice(start, start + source_size)
        padded[tuple(dst_sl)] = tmp
    else:
        # Downsampling: extract center of source spectrum
        start = (source_size - target_size) // 2
        src_sl = [slice(None)] * tmp.ndim
        src_sl[axis] = slice(start, start + target_size)
        padded = tmp[tuple(src_sl)].copy()

    # Centered IFFT back
    result = np.fft.ifftshift(padded, axes=axis)
    result = np.fft.ifft(result, axis=axis)
    result = np.fft.fftshift(result, axes=axis)

    # Scale to preserve signal level
    # numpy: fft has no normalization, ifft divides by N
    # After fft(N_in) → pad → ifft(N_out): values scaled by N_in/N_out
    # Multiply by N_out/N_in to restore original signal level
    result *= target_size / source_size

    return result


def compute_sinc_matrix(n_in, n_out):
    """
    Precompute sinc interpolation matrix of shape (n_out, n_in).

    M[i, j] is the weight of input sample j contributing to output sample i.
    Equivalent to sinc_interp_1d but as a precomputed matrix for efficient
    repeated application via matrix multiply (avoids FFT overhead).
    """
    if n_in == n_out:
        return np.eye(n_in, dtype=complex)

    M = np.zeros((n_out, n_in), dtype=complex)
    for j in range(n_in):
        e_j = np.zeros(n_in, dtype=complex)
        e_j[j] = 1.0
        M[:, j] = sinc_interp_1d(e_j, n_out, axis=0)
    return M


def extract_calib(kspace, calib_size):
    """
    Extract calibration region from center of k-space.

    Parameters:
    -----------
    kspace : ndarray
        K-space data [n_coils, ...spatial dims...]
        Can be 2D [n_coils, ny, nx] or 3D [n_coils, nz, ny, nx]
    calib_size : tuple
        Size of calibration region (cal_z, cal_y, cal_x) for 3D or (cal_y, cal_x) for 2D

    Returns:
    --------
    calib : ndarray
        Calibration region from center
    """
    n_coils = kspace.shape[0]
    spatial_dims = kspace.shape[1:]
    n_dims = len(spatial_dims)

    # Ensure calib_size matches number of spatial dimensions
    if len(calib_size) != n_dims:
        raise ValueError(f"calib_size {calib_size} doesn't match {n_dims}D data")

    # Build slicing for center extraction
    slices = [slice(None)]  # Keep all coils
    for i, (full_size, cal_size_i) in enumerate(zip(spatial_dims, calib_size)):
        start = (full_size - cal_size_i) // 2
        slices.append(slice(start, start + cal_size_i))

    calib = kspace[tuple(slices)].copy()

    return calib


def build_calibration_matrix(calib_data, kernel_size):
    """
    Build calibration matrix using sliding window approach.
    This creates overlapping patches that will be used for covariance calculation.

    Parameters:
    -----------
    calib_data : ndarray
        Calibration region [n_coils, ...spatial...]
        2D: [n_coils, cal_y, cal_x]
        3D: [n_coils, cal_z, cal_y, cal_x]
    kernel_size : tuple
        Size of kernel - must match spatial dimensions
        2D: (ky, kx)
        3D: (kz, ky, kx)

    Returns:
    --------
    cal_matrix : ndarray
        Calibration matrix [n_patches, n_coils * kernel_elements]
    """
    n_coils = calib_data.shape[0]
    spatial_shape = calib_data.shape[1:]
    n_dims = len(spatial_shape)

    if len(kernel_size) != n_dims:
        raise ValueError(f"kernel_size {kernel_size} doesn't match {n_dims}D data")

    # Number of patches in each dimension
    n_patches_per_dim = [spatial_shape[i] - kernel_size[i] + 1 for i in range(n_dims)]
    n_patches = np.prod(n_patches_per_dim)

    # Kernel dimensions in vectorized form
    kernel_elements = n_coils * np.prod(kernel_size)

    # Build calibration matrix by extracting all patches
    cal_matrix = np.zeros((n_patches, kernel_elements), dtype=complex)

    # Generate all patch starting positions
    if n_dims == 2:
        patch_idx = 0
        for py in range(n_patches_per_dim[0]):
            for px in range(n_patches_per_dim[1]):
                patch = calib_data[
                    :, py : py + kernel_size[0], px : px + kernel_size[1]
                ]
                cal_matrix[patch_idx] = patch.ravel()
                patch_idx += 1
    elif n_dims == 3:
        patch_idx = 0
        for pz in range(n_patches_per_dim[0]):
            for py in range(n_patches_per_dim[1]):
                for px in range(n_patches_per_dim[2]):
                    patch = calib_data[
                        :,
                        pz : pz + kernel_size[0],
                        py : py + kernel_size[1],
                        px : px + kernel_size[2],
                    ]
                    cal_matrix[patch_idx] = patch.ravel()
                    patch_idx += 1
    else:
        raise ValueError(f"Only 2D and 3D data supported, got {n_dims}D")

    return cal_matrix


def compute_kernels_svd(cal_matrix, threshold=0.01, num_kernels=None):
    """
    Compute ESPIRiT kernels via eigendecomposition of the calibration Gram matrix.

    Instead of computing the full SVD of the (tall) calibration matrix A, we compute
    the eigendecomposition of the Gram matrix A^H @ A, which is mathematically equivalent
    but computationally more efficient:

    If SVD: A = U Σ V^H, then A^H @ A = V Σ^2 V^H
    So: eigenvectors of A^H @ A = right singular vectors of A
        eigenvalues of A^H @ A = squared singular values of A

    This extracts the SIGNAL SUBSPACE (large singular values) from k-space patches.
    These kernels represent the coil sensitivity patterns present in the data.

    Note: Despite "null-space" terminology in ESPIRiT literature, we extract the
    signal space here. The "null-space" refers to the image reconstruction problem,
    not the calibration SVD.

    Parameters:
    -----------
    cal_matrix : ndarray
        Calibration matrix [n_patches, kernel_elements]
    threshold : float
        Threshold for selecting kernels (relative to max singular value)
        Keeps singular values > threshold * max_sval
    num_kernels : int or None
        If specified, keep this many kernels regardless of threshold

    Returns:
    --------
    kernels : ndarray
        ESPIRiT kernels from SIGNAL SPACE [n_kernels, kernel_elements]
        These are eigenvectors with the LARGEST eigenvalues (= right singular vectors)
    svals : ndarray
        Singular values (sorted descending)
    """
    # Compute Gram matrix: A^H @ A (unnormalized, matching BART's casorati_gram)
    # BART computes raw A^H @ A with no normalization.
    # Eigenvectors are identical regardless of scaling, and kernel selection
    # uses ratio val[i]/val[0] which is also scale-invariant.
    gram = cal_matrix.conj().T @ cal_matrix

    # Eigendecomposition of Hermitian Gram matrix
    # Eigenvectors = right singular vectors of cal_matrix
    # Eigenvalues = squared singular values of cal_matrix (normalized)
    eigenvalues, eigenvectors = linalg.eigh(gram)

    # Sort by descending eigenvalue (largest first = signal space)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Convert eigenvalues to singular values
    svals = np.sqrt(np.abs(eigenvalues))

    # Determine number of kernels to keep
    if num_kernels is not None:
        n_keep = num_kernels
    else:
        # BART thresholds as: val[i]/val[0] > sqrt(threshold)
        # See src/calib/calib.c:857 - number_of_kernels()
        max_sval = svals[0]
        n_keep = np.sum(svals / max_sval > np.sqrt(threshold))

    # Ensure at least 1 kernel
    n_keep = max(1, n_keep)

    print(f"  Keeping {n_keep} kernels (threshold: {threshold})")
    print(f"  Singular value range: [{svals[n_keep - 1]:.6f}, {svals[0]:.6f}]")

    # Extract signal space kernels (eigenvectors with LARGEST eigenvalues)
    # These are unit eigenvectors - no additional weighting
    kernels = eigenvectors[:, :n_keep].T.conj()

    return kernels, svals


def kernels_to_image_domain(kernels, kernel_size, n_coils, img_size):
    """
    Transform kernels from k-space to image domain.

    This zero-pads the kernels and applies inverse FFT.

    Parameters:
    -----------
    kernels : ndarray
        Kernels [n_kernels, n_coils * kernel_elements]
    kernel_size : tuple
        Kernel dimensions - 2D: (ky, kx), 3D: (kz, ky, kx)
    n_coils : int
        Number of coils
    img_size : tuple
        Target image size - typically 2x kernel size per dimension
        2D: (ny, nx), 3D: (nz, ny, nx)

    Returns:
    --------
    img_kernels : ndarray
        Image-domain kernels [n_kernels, n_coils, ...img_size...]
    """
    n_kernels = kernels.shape[0]
    n_dims = len(kernel_size)

    if len(img_size) != n_dims:
        raise ValueError(f"img_size {img_size} doesn't match kernel_size {kernel_size}")

    # Reshape kernels to spatial format
    spatial_shape = (n_kernels, n_coils) + kernel_size
    kernels_spatial = kernels.reshape(spatial_shape)

    # Zero-pad to image size
    full_shape = (n_kernels, n_coils) + img_size
    img_kernels = np.zeros(full_shape, dtype=complex)

    # Calculate padding positions
    starts = tuple((img_size[i] - kernel_size[i]) // 2 for i in range(n_dims))

    # Build slicing
    if n_dims == 2:
        img_kernels[
            :,
            :,
            starts[0] : starts[0] + kernel_size[0],
            starts[1] : starts[1] + kernel_size[1],
        ] = kernels_spatial
        # IFFT to image domain
        for k in range(n_kernels):
            for c in range(n_coils):
                img_kernels[k, c] = ifft2c(img_kernels[k, c])
    elif n_dims == 3:
        img_kernels[
            :,
            :,
            starts[0] : starts[0] + kernel_size[0],
            starts[1] : starts[1] + kernel_size[1],
            starts[2] : starts[2] + kernel_size[2],
        ] = kernels_spatial
        # IFFT to image domain (3D)
        for k in range(n_kernels):
            for c in range(n_coils):
                img_kernels[k, c] = ifft3c(img_kernels[k, c])
    else:
        raise ValueError(f"Only 2D and 3D supported, got {n_dims}D")

    return img_kernels


def compute_image_covariance(img_kernels, kernel_size):
    """
    Compute point-wise covariance matrices in image domain.

    For each spatial location, compute G = H^H @ H where H are the kernels.
    Normalized to ensure eigenvalues are in reasonable range [0,1].

    Parameters:
    -----------
    img_kernels : ndarray
        Image kernels [n_kernels, n_coils, ...spatial...]
        2D: [n_kernels, n_coils, ny, nx]
        3D: [n_kernels, n_coils, nz, ny, nx]
    kernel_size : tuple
        Original kernel size in k-space (before zero-padding)
        2D: (ky, kx), 3D: (kz, ky, kx)

    Returns:
    --------
    img_cov : ndarray
        Image covariance [...spatial..., n_coils, n_coils]
        2D: [ny, nx, n_coils, n_coils]
        3D: [nz, ny, nx, n_coils, n_coils]
    """
    n_kernels = img_kernels.shape[0]
    n_coils = img_kernels.shape[1]
    spatial_shape = img_kernels.shape[2:]
    n_dims = len(spatial_shape)

    # Compute image volume (size of FFT'd space) and kernel volume
    image_volume = np.prod(spatial_shape)
    kernel_volume = np.prod(kernel_size)

    # BART normalization accounting:
    # BART uses unnormalized IFFT: h_bart = image_vol * h_numpy
    # BART's raw Gram: G_bart = image_vol^2 * G_numpy
    # BART divides by scalesq = kernel_vol * image_vol → G_bart/scalesq
    # BART then sinc_zeropad resizes covariance, which scales by image_vol
    # Net BART result: G_bart * image_vol / scalesq = image_vol^2 * G_numpy / kernel_vol
    # To match: G_numpy / normalization = image_vol^2 * G_numpy / kernel_vol
    # → normalization = kernel_vol / image_vol^2
    normalization = kernel_volume / (image_volume**2)

    print("  Covariance normalization:")
    print(f"    kernel_vol={kernel_volume}, image_vol={image_volume}")
    print(f"    Dividing by kernel_vol/image_vol^2 = {normalization:.6e}")
    print("    (Accounts for BART's unnormalized IFFT + sinc_zeropad scaling)")

    # Compute H^H @ H for all spatial locations at once using einsum.
    # img_kernels: [n_kernels, n_coils, ...spatial...]
    # We need: cov[..., i, j] = sum_k conj(H[k, i, ...]) * H[k, j, ...]
    # Move spatial dims to front, coils to back:
    #   H_t: [...spatial..., n_kernels, n_coils]
    axes_order = list(range(2, 2 + n_dims)) + [0, 1]
    H_t = np.transpose(img_kernels, axes_order)
    # Batched outer product: H^H @ H at each spatial location
    img_cov = np.einsum("...ki,...kj->...ij", H_t.conj(), H_t) / normalization

    return img_cov


def resize_sensitivity_maps(sens_maps, target_shape):
    """
    Resize sensitivity maps to full image size using sinc interpolation.

    This is much more memory efficient than resizing the covariance matrix,
    since we only interpolate n_coils values per pixel instead of n_coils^2.

    Parameters:
    -----------
    sens_maps : ndarray
        Sensitivity maps [num_maps, n_coils, ...spatial_small...]
        2D: [num_maps, n_coils, ny_small, nx_small]
        3D: [num_maps, n_coils, nz_small, ny_small, nx_small]
    target_shape : tuple
        Target size - must match spatial dimensions
        2D: (ny_full, nx_full)
        3D: (nz_full, ny_full, nx_full)

    Returns:
    --------
    sens_maps_full : ndarray
        Resized sensitivity maps [num_maps, n_coils, ...spatial_full...]
    """
    num_maps = sens_maps.shape[0]
    n_coils = sens_maps.shape[1]
    spatial_small = sens_maps.shape[2:]
    n_dims = len(spatial_small)

    if len(target_shape) != n_dims:
        raise ValueError(f"target_shape {target_shape} doesn't match {n_dims}D data")

    sens_full_shape = (num_maps, n_coils) + target_shape
    sens_maps_full = np.zeros(sens_full_shape, dtype=complex)

    shape_str = "x".join(map(str, spatial_small))
    target_str = "x".join(map(str, target_shape))
    print(f"  Resizing sensitivity maps from {shape_str} to {target_str}...")

    # Interpolate each map and coil
    for m in range(num_maps):
        for c in range(n_coils):
            sens_map = sens_maps[m, c]

            if n_dims == 2:
                # FFT to k-space
                kspace = fft2c(sens_map)
                # Zero-pad
                kspace_padded = np.zeros(target_shape, dtype=complex)
                y_start = (target_shape[0] - spatial_small[0]) // 2
                x_start = (target_shape[1] - spatial_small[1]) // 2
                kspace_padded[
                    y_start : y_start + spatial_small[0],
                    x_start : x_start + spatial_small[1],
                ] = kspace
                # IFFT back
                scale = np.prod(target_shape) / np.prod(spatial_small)
                sens_maps_full[m, c] = ifft2c(kspace_padded) * scale

            elif n_dims == 3:
                # FFT to k-space (3D)
                kspace = fft3c(sens_map)
                # Zero-pad
                kspace_padded = np.zeros(target_shape, dtype=complex)
                z_start = (target_shape[0] - spatial_small[0]) // 2
                y_start = (target_shape[1] - spatial_small[1]) // 2
                x_start = (target_shape[2] - spatial_small[2]) // 2
                kspace_padded[
                    z_start : z_start + spatial_small[0],
                    y_start : y_start + spatial_small[1],
                    x_start : x_start + spatial_small[2],
                ] = kspace
                # IFFT back (3D)
                scale = np.prod(target_shape) / np.prod(spatial_small)
                sens_maps_full[m, c] = ifft3c(kspace_padded) * scale
            else:
                raise ValueError(f"Only 2D and 3D supported, got {n_dims}D")

    return sens_maps_full


def compute_phase_rotation_matrix(calib_data):
    """
    Compute phase rotation matrix for coil phase alignment.

    This implements BART's 'scc' (self-consistency coil) function which computes
    the coil covariance Gram matrix from calibration data and extracts eigenvectors.
    These eigenvectors represent the principal coil patterns and are used to
    align all coil phases consistently.

    Based on: src/calib/cc.c:scc()

    Parameters:
    -----------
    calib_data : ndarray
        Calibration region [n_coils, ...spatial...]

    Returns:
    --------
    rotation_matrix : ndarray
        Coil rotation matrix [n_coils, n_coils]
        Eigenvectors of coil covariance (flipped to match BART ordering)
    """
    n_coils = calib_data.shape[0]

    # Flatten spatial dimensions
    calib_flat = calib_data.reshape(n_coils, -1)  # [n_coils, n_pixels]

    # Compute Gram matrix: coil-to-coil covariance
    # gram[i,j] = <coil_i, coil_j> across all calibration pixels
    gram = calib_flat @ calib_flat.conj().T / calib_flat.shape[1]

    # Eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(gram)

    # Sort by descending eigenvalue (most to least significant)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Print energy distribution (for debugging, matches BART)
    energy_sum = eigenvalues.sum()
    print("  Phase rotation coil energy distribution:")
    print(f"  {' '.join([f'{e / energy_sum:.3f}' for e in eigenvalues])}")

    # Return eigenvectors as rotation matrix
    # Note: BART flips the maps dimension, but for rotation matrix this is already
    # in the correct order (columns are eigenvectors)
    return eigenvectors


def fix_phase(sens_maps, rotation_matrix):
    """
    Apply phase rotation to align coil phases with first principal component.

    This implements BART's 'fixphase2' function which:
    1. Projects each voxel's coil values onto the rotation vectors
    2. Extracts the phase of this projection
    3. Multiplies by conjugate of phase to remove it

    Based on: src/misc/utils.c:fixphase2()

    Parameters:
    -----------
    sens_maps : ndarray
        Sensitivity maps [num_maps, n_coils, ...spatial...]
    rotation_matrix : ndarray
        Rotation matrix [n_coils, n_coils] (eigenvectors)

    Returns:
    --------
    sens_maps_rotated : ndarray
        Phase-rotated sensitivity maps (same shape as input)
    """
    num_maps, n_coils = sens_maps.shape[0], sens_maps.shape[1]
    spatial_shape = sens_maps.shape[2:]

    # Flatten spatial dimensions for easier computation
    sens_flat = sens_maps.reshape(
        num_maps, n_coils, -1
    )  # [num_maps, n_coils, n_pixels]

    # For each map
    sens_rotated = np.zeros_like(sens_flat)
    for m in range(num_maps):
        # Project onto first rotation vector (first principal component)
        # phase_ref[pixel] = sum_c (rotation[c, 0]^* * sens[m, c, pixel])
        phase_ref = np.einsum("c,cp->p", rotation_matrix[:, 0].conj(), sens_flat[m])

        # Extract phase (normalize complex numbers to unit magnitude)
        phase_ref = phase_ref / (np.abs(phase_ref) + 1e-10)

        # Remove this phase from all coils at each pixel
        # sens_rotated[m, c, pixel] = sens[m, c, pixel] * phase_ref[pixel]^*
        sens_rotated[m] = sens_flat[m] * phase_ref.conj()[None, :]

    # Reshape back to original shape
    sens_rotated = sens_rotated.reshape(sens_maps.shape)

    return sens_rotated


def gram_schmidt(vectors):
    """
    Gram-Schmidt orthonormalization.

    Orthonormalizes column vectors in-place.

    Parameters:
    -----------
    vectors : ndarray
        Matrix [n, m] where columns are vectors to orthonormalize

    Returns:
    --------
    ortho : ndarray
        Orthonormalized vectors [n, m]
    norms : ndarray
        Norms of each vector [m]
    """
    n, m = vectors.shape
    ortho = vectors.copy()
    norms = np.zeros(m)

    for j in range(m):
        # Subtract projections onto previous vectors
        for i in range(j):
            proj = np.vdot(ortho[:, i], ortho[:, j])
            ortho[:, j] -= proj * ortho[:, i]

        # Normalize
        norm = np.linalg.norm(ortho[:, j])
        if norm > 1e-10:
            ortho[:, j] /= norm
        norms[j] = norm

    return ortho, norms


def orthogonal_iteration(matrix, num_vecs, num_iter=30):
    """
    Compute eigenvectors via orthogonal iteration (power iteration + Gram-Schmidt).

    This is more numerically stable than direct eigendecomposition for
    computing the dominant eigenvectors of a Hermitian matrix.

    Based on: src/num/linalg.c:orthiter()

    Parameters:
    -----------
    matrix : ndarray
        Hermitian matrix [n, n]
    num_vecs : int
        Number of eigenvectors to compute
    num_iter : int
        Number of iterations (default: 30, BART default)

    Returns:
    --------
    eigvals : ndarray
        Eigenvalues [num_vecs] (sorted descending)
    eigvecs : ndarray
        Eigenvectors [n, num_vecs] (columns are eigenvectors)
    """
    n = matrix.shape[0]
    num_vecs = min(num_vecs, n)

    # Initialize with identity
    vecs = np.eye(n, num_vecs, dtype=matrix.dtype)

    # Iterate: multiply by matrix and orthonormalize
    for _ in range(num_iter):
        # Multiply by matrix
        vecs = matrix @ vecs

        # Gram-Schmidt orthonormalization
        vecs, norms = gram_schmidt(vecs)

    # Final multiplication to get eigenvalues
    vecs_proj = matrix @ vecs
    eigvals = np.zeros(num_vecs)
    for j in range(num_vecs):
        eigvals[j] = np.real(np.vdot(vecs[:, j], vecs_proj[:, j]))

    return eigvals, vecs


def _orthiter_chunk(args):
    """
    Worker function for threaded orthogonal iteration.
    Processes a chunk of flattened covariance matrices.

    Parameters: (cov_chunk, num_maps, num_orthiter)
        cov_chunk: [B, nc, nc] batch of Hermitian matrices
        num_maps: number of eigenvectors to compute
        num_orthiter: number of power iterations

    Returns: (eigvecs, eigvals) where
        eigvecs: [B, nc, num_maps]
        eigvals: [B, num_maps]
    """
    cov_chunk, num_maps, num_orthiter = args
    B, nc = cov_chunk.shape[0], cov_chunk.shape[1]

    if num_maps == 1:
        # Optimized path for single map: pure power iteration, no Gram-Schmidt
        v = np.zeros((B, nc), dtype=cov_chunk.dtype)
        v[:, 0] = 1.0
        for _ in range(num_orthiter):
            v = np.einsum("...ij,...j->...i", cov_chunk, v)
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            norm = np.where(norm > 1e-30, norm, 1.0)
            v = v / norm
        # Eigenvalue: v^H A v
        Av = np.einsum("...ij,...j->...i", cov_chunk, v)
        eigvals = np.real(np.sum(v.conj() * Av, axis=-1, keepdims=True))
        return v[..., np.newaxis], eigvals  # [B, nc, 1], [B, 1]
    else:
        # General case: orthogonal iteration with Gram-Schmidt
        vecs = np.zeros((B, nc, num_maps), dtype=cov_chunk.dtype)
        for m in range(num_maps):
            vecs[:, m % nc, m] = 1.0
        for _ in range(num_orthiter):
            # Batched matmul: [B, nc, nc] @ [B, nc, M] -> [B, nc, M]
            vecs = np.einsum("...ij,...jk->...ik", cov_chunk, vecs)
            # Gram-Schmidt
            for m in range(num_maps):
                v = vecs[..., m]  # [B, nc]
                for j in range(m):
                    u = vecs[..., j]  # [B, nc]
                    proj = np.sum(u.conj() * v, axis=-1, keepdims=True)
                    v = v - proj * u
                norm = np.linalg.norm(v, axis=-1, keepdims=True)
                norm = np.where(norm > 1e-30, norm, 1.0)
                vecs[..., m] = v / norm
        # Eigenvalues
        Av = np.einsum("...ij,...jk->...ik", cov_chunk, vecs)
        eigvals = np.real(np.sum(vecs.conj() * Av, axis=-2))
        return vecs, eigvals  # [B, nc, M], [B, M]


def eigenmaps_batched(img_cov, num_maps=1, orthiter=True, num_orthiter=30):
    """
    Batched eigendecomposition of covariance matrices.

    Matches BART's eigenmaps() function which uses OpenMP to parallelize
    across voxels. This Python version uses ThreadPoolExecutor + einsum
    to achieve similar parallelism.

    Performance optimizations vs naive implementation:
    - Uses einsum for batched matrix-vector product (avoids extra dimension overhead)
    - Splits work across threads (numpy releases GIL during C-level ops)
    - For num_maps=1: simplified power iteration (no Gram-Schmidt needed)

    Parameters:
    -----------
    img_cov : ndarray
        Covariance matrices [...spatial..., n_coils, n_coils]
    num_maps : int
        Number of sensitivity maps to extract (default: 1)
    orthiter : bool
        Use orthogonal iteration (default: True, BART default)
    num_orthiter : int
        Number of iterations for orthiter (default: 30, BART default)

    Returns:
    --------
    sens_maps : ndarray
        Sensitivity maps [num_maps, n_coils, ...spatial...]
    eigenvalues : ndarray
        Eigenvalue maps [num_maps, ...spatial...]
    """
    n_coils = img_cov.shape[-1]
    spatial_shape = img_cov.shape[:-2]
    n_voxels = int(np.prod(spatial_shape))
    dtype = img_cov.dtype

    if orthiter:
        # Flatten spatial dims for chunked processing: [B, nc, nc]
        cov_flat = img_cov.reshape(n_voxels, n_coils, n_coils)

        nthreads = _get_num_threads()
        if nthreads > 1 and n_voxels > 256:
            # Split into chunks and process in parallel
            # (mirrors BART's #pragma omp parallel for collapse(3) in eigenmaps)
            chunks = np.array_split(cov_flat, nthreads)
            with ThreadPoolExecutor(max_workers=nthreads) as executor:
                results = list(
                    executor.map(
                        _orthiter_chunk,
                        [(c, num_maps, num_orthiter) for c in chunks],
                    )
                )
            all_vecs = np.concatenate([r[0] for r in results], axis=0)
            all_vals = np.concatenate([r[1] for r in results], axis=0)
        else:
            # Single-threaded
            all_vecs, all_vals = _orthiter_chunk((cov_flat, num_maps, num_orthiter))

        # Reshape back: [B, nc, M] -> [M, nc, ...spatial...]
        all_vecs = all_vecs.reshape(spatial_shape + (n_coils, num_maps))
        all_vals = all_vals.reshape(spatial_shape + (num_maps,))

        sens_maps = np.zeros((num_maps, n_coils) + spatial_shape, dtype=dtype)
        eigenvalues = np.zeros((num_maps,) + spatial_shape)
        for m in range(num_maps):
            sens_maps[m] = np.moveaxis(all_vecs[..., m], -1, 0)
            eigenvalues[m] = all_vals[..., m]

    else:
        # Full eigendecomposition via LAPACK
        eigvals_all, eigvecs_all = np.linalg.eigh(img_cov)

        sens_maps = np.zeros((num_maps, n_coils) + spatial_shape, dtype=dtype)
        eigenvalues = np.zeros((num_maps,) + spatial_shape)

        for m in range(num_maps):
            idx = -(m + 1)
            sens_maps[m] = np.moveaxis(eigvecs_all[..., idx], -1, 0)
            eigenvalues[m] = eigvals_all[..., idx]

    return sens_maps, eigenvalues


def _process_zslice(args):
    """
    Process a single z-slice: sinc-resize y,x + eigenmaps.
    Designed to be called from a process pool or directly.

    Parameters: tuple (z, slc_z, M_y, M_x, ny, nx, ny_s, nx_s, nc,
                        cosize, tri_i, tri_j, num_maps, orthiter, num_orthiter, dtype)
    Returns: (z, sens_slice, eig_slice)
    """
    (
        z,
        slc_z,
        M_y,
        M_x,
        ny,
        nx,
        ny_s,
        nx_s,
        nc,
        cosize,
        tri_i,
        tri_j,
        num_maps,
        orthiter,
        num_orthiter,
        dtype,
    ) = args

    # Resize packed slice along y via GEMM
    slc = (M_y @ slc_z.reshape(ny_s, -1)).reshape(ny, nx_s, cosize)

    # Resize packed slice along x via GEMM
    slc = (
        (M_x @ slc.transpose(1, 0, 2).reshape(nx_s, -1))
        .reshape(nx, ny, cosize)
        .transpose(1, 0, 2)
    )  # (ny, nx, cosize)

    # Unpack triangle to full Hermitian matrix
    cov_full = np.zeros((ny, nx, nc, nc), dtype=dtype)
    cov_full[..., tri_i, tri_j] = slc
    cov_full[..., tri_j, tri_i] = slc.conj()

    # Eigenmaps for this slice (full ny×nx at once)
    s, e = eigenmaps_batched(cov_full, num_maps, orthiter, num_orthiter)

    return z, s, e


def caltwo(img_cov, target_shape, num_maps=1, orthiter=True, num_orthiter=30):
    """
    Resize covariance to full resolution and compute eigenmaps.

    This matches BART's caltwo() function:
    - 2D: sinc_zeropad covariance to full size, then eigenmaps
    - 3D: uses econdim approach (process one z-slice at a time for memory)

    BART's econdim trick for 3D:
      1. sinc_zeropad covariance along z to full nz
      2. For each z-slice: sinc_zeropad along y,x → eigenmaps
    This keeps memory at O(ny * nx * n_coils^2) per slice instead of
    O(nz * ny * nx * n_coils^2) for the full volume.

    Parameters:
    -----------
    img_cov : ndarray
        Low-resolution covariance [...spatial_small..., n_coils, n_coils]
    target_shape : tuple
        Full-resolution spatial shape
    num_maps : int
        Number of sensitivity maps
    orthiter : bool
        Use orthogonal iteration (default: True, BART default)
    num_orthiter : int
        Number of orthiter iterations (default: 30)

    Returns:
    --------
    sens_maps : ndarray [num_maps, n_coils, ...target_shape...]
    eigenvalues : ndarray [num_maps, ...target_shape...]
    """
    n_coils = img_cov.shape[-1]
    n_dims = img_cov.ndim - 2
    dtype = img_cov.dtype

    if n_dims == 2:
        ny, nx = target_shape
        # Resize covariance to full resolution (all at once for 2D)
        cov_full = sinc_interp_1d(img_cov, ny, axis=0)
        cov_full = sinc_interp_1d(cov_full, nx, axis=1)
        print(f"  Full-res covariance shape: {cov_full.shape}")

        # Batched eigenmaps at full resolution
        sens_maps, eigenvalues = eigenmaps_batched(
            cov_full, num_maps, orthiter, num_orthiter
        )
        return sens_maps, eigenvalues

    elif n_dims == 3:
        nz, ny, nx = target_shape
        nz_s, ny_s, nx_s = img_cov.shape[:3]
        nc = n_coils

        # Pack covariance to upper triangle (like BART) — ~2× less data to interpolate
        cosize = nc * (nc + 1) // 2
        cov_packed = np.zeros((nz_s, ny_s, nx_s, cosize), dtype=dtype)
        idx = 0
        tri_i = np.zeros(cosize, dtype=int)
        tri_j = np.zeros(cosize, dtype=int)
        for i in range(nc):
            for j in range(i + 1):
                cov_packed[..., idx] = img_cov[..., i, j]
                tri_i[idx] = i
                tri_j[idx] = j
                idx += 1
        print(
            f"  Packed covariance: {img_cov.shape} -> {cov_packed.shape} ({cosize} vs {nc * nc} elements/voxel)"
        )

        # Precompute sinc interpolation matrices
        M_z = compute_sinc_matrix(nz_s, nz)
        M_y = compute_sinc_matrix(ny_s, ny)
        M_x = compute_sinc_matrix(nx_s, nx)
        print(
            f"  Precomputed interpolation matrices: z({nz},{nz_s}) y({ny},{ny_s}) x({nx},{nx_s})"
        )

        # Cast interpolation matrices to working dtype
        M_z = M_z.astype(dtype)
        M_y = M_y.astype(dtype)
        M_x = M_x.astype(dtype)

        # Step A: sinc-resize packed covariance along z
        print(f"  Step A: Resizing packed covariance along z ({nz_s} -> {nz})...")
        cov_z = (M_z @ cov_packed.reshape(nz_s, -1)).reshape(nz, ny_s, nx_s, cosize)
        mem_mb = cov_z.nbytes / 1e6
        print(f"    z-resized packed covariance: {cov_z.shape} ({mem_mb:.0f} MB)")

        # Step B: process each z-slice (full slice at once)
        nthreads = _get_num_threads()
        print(
            f"  Step B: Processing {nz} z-slices (resize y,x + eigenmaps, "
            f"{nthreads} threads for eigenmaps)..."
        )
        sens_maps = np.zeros((num_maps, nc, nz, ny, nx), dtype=dtype)
        eigenvalues = np.zeros((num_maps, nz, ny, nx))

        t0 = time.perf_counter()
        for z in range(nz):
            t_slice = time.perf_counter()

            # Resize packed slice along y via GEMM
            slc = cov_z[z]  # (ny_s, nx_s, cosize)
            slc = (M_y @ slc.reshape(ny_s, -1)).reshape(ny, nx_s, cosize)

            # Resize packed slice along x via GEMM
            slc = (
                (M_x @ slc.transpose(1, 0, 2).reshape(nx_s, -1))
                .reshape(nx, ny, cosize)
                .transpose(1, 0, 2)
            )  # (ny, nx, cosize)

            # Unpack triangle to full Hermitian matrix
            cov_full = np.zeros((ny, nx, nc, nc), dtype=dtype)
            cov_full[..., tri_i, tri_j] = slc
            cov_full[..., tri_j, tri_i] = slc.conj()

            # Eigenmaps for this slice (threaded internally)
            s, e = eigenmaps_batched(cov_full, num_maps, orthiter, num_orthiter)

            sens_maps[:, :, z] = s
            eigenvalues[:, z] = e

            t_total_slice = time.perf_counter() - t_slice
            elapsed = time.perf_counter() - t0
            eta = elapsed / (z + 1) * (nz - z - 1) if z < nz - 1 else 0
            if (z + 1) % 10 == 0 or z == 0 or z == nz - 1:
                print(
                    f"    Slice {z + 1}/{nz}: {t_total_slice:.2f}s "
                    f"[{elapsed:.1f}s elapsed, ~{eta:.0f}s left]"
                )

        elapsed = time.perf_counter() - t0
        print(f"    Completed in {elapsed:.1f}s")
        return sens_maps, eigenvalues

    else:
        raise ValueError(f"Only 2D and 3D supported, got {n_dims}D")


def pointwise_eigenmaps(
    img_cov, num_maps=1, phase_smooth=True, orthiter=True, num_orthiter=30
):
    """
    Compute sensitivity maps via point-wise eigendecomposition.

    At each pixel, we compute eigenvectors of the covariance G = H^H @ H,
    where H are the image-domain kernels. The eigenvector with the LARGEST
    eigenvalue (closest to 1) represents the dominant sensitivity pattern.

    This is where ESPIRiT extracts sensitivity maps - eigenvectors with
    eigenvalues near 1 indicate regions with consistent coil patterns.

    Parameters:
    -----------
    img_cov : ndarray
        Image covariance [...spatial..., n_coils, n_coils]
        2D: [ny, nx, n_coils, n_coils]
        3D: [nz, ny, nx, n_coils, n_coils]
    num_maps : int
        Number of sensitivity maps to extract (usually 1)
    phase_smooth : bool
        Apply phase smoothing to enforce spatial continuity (default: True)
    orthiter : bool
        Use orthogonal iteration for eigendecomposition (default: True, BART default)
    num_orthiter : int
        Number of orthogonal iterations (default: 30, BART default)

    Returns:
    --------
    sens_maps : ndarray
        Sensitivity maps [num_maps, n_coils, ...spatial...]
        2D: [num_maps, n_coils, ny, nx]
        3D: [num_maps, n_coils, nz, ny, nx]
    eigenvalues : ndarray
        Eigenvalue maps [num_maps, ...spatial...]
        Eigenvalues near 1.0 indicate well-determined sensitivities
        2D: [num_maps, ny, nx]
        3D: [num_maps, nz, ny, nx]
    """
    n_coils = img_cov.shape[-1]
    spatial_shape = img_cov.shape[:-2]
    n_dims = len(spatial_shape)

    sens_shape = (num_maps, n_coils) + spatial_shape
    eigval_shape = (num_maps,) + spatial_shape

    sens_maps = np.zeros(sens_shape, dtype=complex)
    eigenvalues = np.zeros(eigval_shape)

    method = "orthogonal iteration" if orthiter else "direct eigendecomposition"
    print(f"  Computing {num_maps} sensitivity map(s) via point-wise {method}...")

    # Eigendecomposition at each voxel with optional phase smoothing
    if n_dims == 2:
        ny, nx = spatial_shape
        for y in range(ny):
            for x in range(nx):
                cov = img_cov[y, x]

                # Choose eigendecomposition method
                if orthiter:
                    eigvals, eigvecs = orthogonal_iteration(cov, num_maps, num_orthiter)
                else:
                    eigvals, eigvecs = linalg.eigh(cov)
                    idx = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]

                for m in range(num_maps):
                    eigvec = eigvecs[:, m]

                    # Phase smoothing: align with previous pixel
                    if phase_smooth and (x > 0 or y > 0):
                        # Use left neighbor, or top if at left edge
                        if x > 0:
                            ref = sens_maps[m, :, y, x - 1]
                        else:
                            ref = sens_maps[m, :, y - 1, x]

                        # Align phase by maximizing correlation
                        if np.abs(ref).sum() > 1e-10:
                            phase_factor = np.sum(ref.conj() * eigvec)
                            if np.abs(phase_factor) > 1e-10:
                                eigvec *= phase_factor / np.abs(phase_factor)

                    sens_maps[m, :, y, x] = eigvec
                    eigenvalues[m, y, x] = eigvals[m]

    elif n_dims == 3:
        nz, ny, nx = spatial_shape
        for z in range(nz):
            if (z + 1) % 10 == 0 or z == nz - 1:
                print(f"    Processing slice {z + 1}/{nz}...")
            for y in range(ny):
                for x in range(nx):
                    cov = img_cov[z, y, x]

                    # Choose eigendecomposition method
                    if orthiter:
                        eigvals, eigvecs = orthogonal_iteration(
                            cov, num_maps, num_orthiter
                        )
                    else:
                        eigvals, eigvecs = linalg.eigh(cov)
                        idx = np.argsort(eigvals)[::-1]
                        eigvals = eigvals[idx]
                        eigvecs = eigvecs[:, idx]

                    for m in range(num_maps):
                        eigvec = eigvecs[:, m]

                        # Phase smoothing: align with previous pixel
                        if phase_smooth and (x > 0 or y > 0 or z > 0):
                            # Use left neighbor, or top if at left edge, or previous slice if at corner
                            if x > 0:
                                ref = sens_maps[m, :, z, y, x - 1]
                            elif y > 0:
                                ref = sens_maps[m, :, z, y - 1, x]
                            else:
                                ref = sens_maps[m, :, z - 1, y, x]

                            # Align phase by maximizing correlation
                            if np.abs(ref).sum() > 1e-10:
                                phase_factor = np.sum(ref.conj() * eigvec)
                                if np.abs(phase_factor) > 1e-10:
                                    eigvec *= phase_factor / np.abs(phase_factor)

                        sens_maps[m, :, z, y, x] = eigvec
                        eigenvalues[m, z, y, x] = eigvals[m]
    else:
        raise ValueError(f"Only 2D and 3D supported, got {n_dims}D")

    return sens_maps, eigenvalues


def crop_sensitivity_maps(sens_maps, eigenvalues, crop_threshold=0.8, softcrop=False):
    """
    Crop/weight sensitivity maps based on eigenvalue threshold.

    Implements BART's crop_sens function with support for both hard threshold
    and smooth S-curve weighting.

    Based on: src/calib/calib.c:crop_sens() and helper functions

    Parameters:
    -----------
    sens_maps : ndarray
        Sensitivity maps [num_maps, n_coils, ...spatial...]
    eigenvalues : ndarray
        Eigenvalue maps [num_maps, ...spatial...]
    crop_threshold : float
        Threshold (default: 0.8, BART default)
        - Hard threshold: regions with sqrt(eigenvalue) < threshold are zeroed
        - Soft threshold: smooth weighting centered at threshold
    softcrop : bool
        If True, use smooth S-curve weighting (BART softcrop)
        If False, use hard binary threshold (BART default)

    Returns:
    --------
    sens_cropped : ndarray
        Cropped/weighted sensitivity maps
    mask : ndarray
        Weight/mask [...spatial...]
    """
    # Use first map eigenvalues for cropping
    eigmap = eigenvalues[0]

    # Debug: print eigenvalue statistics
    print(f"    Eigenvalue range: [{np.min(eigmap):.6f}, {np.max(eigmap):.6f}]")
    print(
        f"    Eigenvalue mean: {np.mean(eigmap):.6f}, median: {np.median(eigmap):.6f}"
    )
    sqrt_min = np.sqrt(np.maximum(np.min(eigmap), 0))
    sqrt_max = np.sqrt(np.max(eigmap))
    print(f"    sqrt(eigenvalue) range: [{sqrt_min:.6f}, {sqrt_max:.6f}]")
    print(f"    Crop threshold: {crop_threshold}")

    # Compute weights based on cropping mode
    if softcrop:
        # Soft weighting with S-curve (BART md_crop_weight_fun)
        # BART: sqrt(|val|) → (sqrt(|val|) - crth) / (1 - crth) → scurve
        weight = np.sqrt(np.abs(eigmap))
        weight = (weight - crop_threshold) / (1.0 - crop_threshold)

        # BART's md_scurve: x/(x^2+1) for |x|<=1, 0 for x<-1, 1+(x/(x^2+1)) for x>=1
        def bart_scurve(x):
            result = np.zeros_like(x)
            mask_mid = x >= -1  # x >= -1
            mask_high = x >= 1  # x >= 1
            s = x / (x**2 + 1)  # core s-curve
            result[mask_mid] = s[mask_mid]
            result[mask_high] += 1.0  # add 1 for x >= 1
            return result

        weight = bart_scurve(weight)

    else:
        # Hard threshold (BART md_crop_thresh_fun)
        # BART: md_zabs → md_zsgreatequal, i.e., |eigenvalue| >= threshold
        # NO sqrt for hard threshold (sqrt is only used in soft/scurve mode)
        weight = (np.abs(eigmap) >= crop_threshold).astype(float)

    # Debug: print masking statistics
    n_total = weight.size
    n_kept = np.sum(weight > 0)
    pct_kept = 100.0 * n_kept / n_total
    print(f"    Voxels kept: {n_kept}/{n_total} ({pct_kept:.1f}%)")
    print(f"    Weight range: [{np.min(weight):.3f}, {np.max(weight):.3f}]")

    # Apply weights - broadcast weight over coil dimension
    sens_cropped = sens_maps.copy()
    for m in range(sens_maps.shape[0]):
        sens_cropped[m] = sens_maps[m] * weight[None, ...]

    return sens_cropped, weight


def normalize_sensitivity_maps(sens_maps, mode="rss"):
    """
    Normalize sensitivity maps.

    Parameters:
    -----------
    sens_maps : ndarray
        Sensitivity maps [num_maps, n_coils, ...spatial...]
    mode : str
        'rss' - root sum of squares
        'l1' - L1 (Walsh) normalization

    Returns:
    --------
    sens_normalized : ndarray
        Normalized sensitivity maps
    """
    sens_normalized = sens_maps.copy()

    for m in range(sens_maps.shape[0]):
        if mode == "rss":
            # Root sum of squares - sum over coil dimension (axis=1)
            norm = np.sqrt(np.sum(np.abs(sens_maps[m]) ** 2, axis=0, keepdims=True))
        elif mode == "l1":
            # L1 Walsh normalization
            norm = np.sum(np.abs(sens_maps[m]), axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        # Avoid division by zero
        norm = np.where(norm > 1e-10, norm, 1.0)
        sens_normalized[m] = sens_maps[m] / norm

    return sens_normalized


def espirit(
    kspace,
    calib_size=None,
    kernel_size=None,
    num_maps=1,
    threshold=0.001,
    crop_threshold=0.8,
    normalize=True,
    phase_smooth=True,
    rotphase=True,
    orthiter=True,
    num_orthiter=30,
    softcrop=False,
):
    """
    Main ESPIRiT function - complete pipeline for 2D or 3D data.

    Defaults match BART's ecalib tool (src/calib/calib.c:686-710).

    Parameters:
    -----------
    kspace : ndarray
        K-space data
        2D: [n_coils, ny, nx]
        3D: [n_coils, nz, ny, nx]
    calib_size : tuple or int or None
        Size of calibration region. Can be:
        - tuple: explicit size per dimension, e.g., (24, 24) for 2D or (24, 24, 24) for 3D
        - int: same size for all dimensions, e.g., 24 → (24, 24) for 2D or (24, 24, 24) for 3D
        - None: auto-select based on data size (default: 24 for all dimensions)
    kernel_size : tuple or int or None
        Size of ESPIRiT kernel. Can be:
        - tuple: explicit size per dimension, e.g., (6, 6) for 2D or (6, 6, 6) for 3D
        - int: same size for all dimensions, e.g., 6 → (6, 6) for 2D or (6, 6, 6) for 3D
        - None: auto-select (default: 6 for all dimensions)
    num_maps : int
        Number of sensitivity maps to compute (default: 1)
    threshold : float
        Threshold for kernel selection (default: 0.001, BART default)
    crop_threshold : float
        Eigenvalue threshold for cropping/weighting (default: 0.8, BART default)
    normalize : bool
        Whether to normalize maps (default: True)
    phase_smooth : bool
        Apply phase smoothing to enforce spatial continuity (default: True)
        Aligns eigenvector phase with neighboring voxels to prevent discontinuities
    rotphase : bool
        Rotate phase with respect to first principal component (default: True, BART default)
        Aligns coil phases by computing coil covariance eigenvectors from calibration data
    orthiter : bool
        Use orthogonal iteration to refine eigenvectors (default: True, BART default)
        Improves numerical accuracy via power iteration with Gram-Schmidt orthonormalization
    num_orthiter : int
        Number of orthogonal iterations (default: 30, BART default)
    softcrop : bool
        Use smooth S-curve weighting instead of hard threshold (default: False, BART default)
        When True, applies smooth transitions; when False, uses hard threshold

    Returns:
    --------
    sens_maps : ndarray
        Sensitivity maps [num_maps, n_coils, ...spatial...]
        2D: [num_maps, n_coils, ny, nx]
        3D: [num_maps, n_coils, nz, ny, nx]
    info : dict
        Additional information:
        - 'eigenvalues': eigenvalue maps
        - 'mask': crop mask
        - 'svals': singular values from calibration
    """
    n_coils = kspace.shape[0]
    spatial_shape = kspace.shape[1:]
    n_dims = len(spatial_shape)

    # Cast to working dtype (BART uses complex float = complex64)
    kspace = kspace.astype(DEFAULT_DTYPE)

    # Auto-detect or expand calib_size
    if calib_size is None:
        calib_size = tuple([24] * n_dims)
    elif isinstance(calib_size, int):
        calib_size = tuple([calib_size] * n_dims)
    elif len(calib_size) != n_dims:
        raise ValueError(f"calib_size {calib_size} doesn't match {n_dims}D data")

    # Auto-detect or expand kernel_size
    if kernel_size is None:
        kernel_size = tuple([6] * n_dims)
    elif isinstance(kernel_size, int):
        kernel_size = tuple([kernel_size] * n_dims)
    elif len(kernel_size) != n_dims:
        raise ValueError(f"kernel_size {kernel_size} doesn't match {n_dims}D data")

    print("=" * 70)
    print(f"ESPIRiT Calibration ({n_dims}D)")
    print("=" * 70)
    print(f"Input k-space shape: {kspace.shape}, dtype: {kspace.dtype}")
    print(f"Calibration size: {calib_size}")
    print(f"Kernel size: {kernel_size}")
    print(f"Number of maps: {num_maps}")
    print(f"Threads: {_get_num_threads()}")
    print()

    t_total = time.perf_counter()

    # Step 1: Extract calibration region
    t0 = time.perf_counter()
    print("Step 1: Extracting calibration region...")
    calib_data = extract_calib(kspace, calib_size)
    print(f"  Calibration region shape: {calib_data.shape}")
    print(
        f"  K-space magnitude - mean: {np.mean(np.abs(kspace)):.2e}, max: {np.max(np.abs(kspace)):.2e}"
    )
    print(
        f"  Calib magnitude - mean: {np.mean(np.abs(calib_data)):.2e}, max: {np.max(np.abs(calib_data)):.2e}"
    )
    print(f"  [Step 1: {time.perf_counter() - t0:.2f}s]")

    # Step 2: Build calibration matrix
    t0 = time.perf_counter()
    print("\nStep 2: Building calibration matrix...")
    cal_matrix = build_calibration_matrix(calib_data, kernel_size)
    print(f"  Calibration matrix shape: {cal_matrix.shape}")
    print(
        f"  Calmat magnitude - mean: {np.mean(np.abs(cal_matrix)):.2e}, max: {np.max(np.abs(cal_matrix)):.2e}"
    )
    print(f"  [Step 2: {time.perf_counter() - t0:.2f}s]")

    # Step 3: Compute ESPIRiT kernels (signal space) via eigendecomposition
    t0 = time.perf_counter()
    print(
        "\nStep 3: Computing ESPIRiT kernels via eigendecomposition of Gram matrix..."
    )
    print(
        "  (Computing A^H @ A and its eigenvectors - equivalent to SVD but more efficient)"
    )
    kernels, svals = compute_kernels_svd(cal_matrix, threshold=threshold)
    print(f"  Kernel shape: {kernels.shape}")
    print(f"  [Step 3: {time.perf_counter() - t0:.2f}s]")

    # Step 4: Transform kernels to image domain
    t0 = time.perf_counter()
    print("\nStep 4: Transforming kernels to image domain...")
    # Standard: 2x kernel size in each dimension
    img_size = tuple(2 * k for k in kernel_size)
    img_kernels = kernels_to_image_domain(kernels, kernel_size, n_coils, img_size)
    print(f"  Image kernel shape: {img_kernels.shape}")
    print(
        f"  Img kernel magnitude - mean: {np.mean(np.abs(img_kernels)):.2e}, max: {np.max(np.abs(img_kernels)):.2e}"
    )
    print(f"  [Step 4: {time.perf_counter() - t0:.2f}s]")

    # Step 5: Compute image-space covariance (at low resolution)
    t0 = time.perf_counter()
    print("\nStep 5: Computing image-space covariance matrices...")
    img_cov = compute_image_covariance(img_kernels, kernel_size)
    print(f"  Covariance shape: {img_cov.shape}")
    print(
        f"  Covariance magnitude - mean: {np.mean(np.abs(img_cov)):.2e}, max: {np.max(np.abs(img_cov)):.2e}"
    )
    print(f"  [Step 5: {time.perf_counter() - t0:.2f}s]")

    # Steps 6-7: Resize covariance to full resolution, then eigenmaps
    # This matches BART's caltwo() approach:
    #   - sinc_zeropad covariance to full image size
    #   - point-wise eigendecomposition at full resolution
    # For 3D, BART uses econdim to process one z-slice at a time (memory efficient).
    # This avoids Gibbs ringing artifacts that would occur if we resized sensitivity
    # maps instead of the covariance.
    t0 = time.perf_counter()
    print("\nSteps 6-7: Resize covariance to full resolution + eigenmaps...")
    print(f"  Target shape: {spatial_shape}")
    sens_maps, eigenvalues = caltwo(
        img_cov,
        spatial_shape,
        num_maps=num_maps,
        orthiter=orthiter,
        num_orthiter=num_orthiter,
    )
    print(f"  Sensitivity maps shape: {sens_maps.shape}")
    print(f"  Eigenvalue range: [{np.min(eigenvalues):.6f}, {np.max(eigenvalues):.6f}]")
    print(f"  [Steps 6-7: {time.perf_counter() - t0:.2f}s]")

    # Step 8: Post-processing
    t0 = time.perf_counter()
    print("\nStep 8: Post-processing...")

    # Debug: check sensitivity maps before cropping
    sens_mag = np.abs(sens_maps)
    print("  Sensitivity map magnitude before cropping:")
    print(f"    Range: [{np.min(sens_mag):.6e}, {np.max(sens_mag):.6e}]")
    print(
        f"    Mean: {np.mean(sens_mag):.6e}, Non-zero fraction: {np.mean(sens_mag > 1e-10):.3f}"
    )

    # Crop based on eigenvalues
    print(f"  Cropping (threshold={crop_threshold})...")
    sens_maps, mask = crop_sensitivity_maps(
        sens_maps, eigenvalues, crop_threshold, softcrop=softcrop
    )

    # Phase rotation (align coil phases)
    if rotphase:
        print("  Applying phase rotation (rotphase)...")
        rotation_matrix = compute_phase_rotation_matrix(calib_data)
        sens_maps = fix_phase(sens_maps, rotation_matrix)

        # Debug: check after phase rotation
        sens_mag = np.abs(sens_maps)
        print(
            f"    After phase rotation - Range: [{np.min(sens_mag):.6e}, {np.max(sens_mag):.6e}]"
        )
        print(f"    Non-zero fraction: {np.mean(sens_mag > 1e-10):.3f}")

    # Normalize
    if normalize:
        print("  Normalizing sensitivity maps...")
        sens_maps = normalize_sensitivity_maps(sens_maps, mode="rss")

        # Debug: check after normalization
        sens_mag = np.abs(sens_maps)
        print(
            f"    After normalization - Range: [{np.min(sens_mag):.6e}, {np.max(sens_mag):.6e}]"
        )
        print(f"    Non-zero fraction: {np.mean(sens_mag > 1e-10):.3f}")

    print(f"  [Step 8: {time.perf_counter() - t0:.2f}s]")

    print("\n" + "=" * 70)
    print(f"ESPIRiT calibration complete! Total: {time.perf_counter() - t_total:.2f}s")
    print("=" * 70)

    info = {
        "eigenvalues": eigenvalues,
        "mask": mask,
        "svals": svals,
        "img_cov": img_cov,
        "kernels": kernels,
    }

    return sens_maps, info


# Example usage
if __name__ == "__main__":
    print("ESPIRiT Python Implementation")
    print("Based on BART ecalib.c")
    print("Supports 2D and 3D data")
    print()

    # Test case selection
    import sys

    test_3d = "--3d" in sys.argv

    if test_3d:
        print("\n" + "=" * 70)
        print("TESTING 3D ESPIRiT")
        print("=" * 70)

        # Create synthetic 3D test data
        n_coils = 8
        nz, ny, nx = 32, 64, 64

        print("\nCreating synthetic 3D multi-coil k-space data...")

        # Create coordinate grids
        z_coords = np.linspace(-1, 1, nz)
        y_coords = np.linspace(-1, 1, ny)
        x_coords = np.linspace(-1, 1, nx)
        z, y, x = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        r = np.sqrt(x**2 + y**2 + z**2)

        # 3D sphere phantom
        phantom = (r < 0.6).astype(complex)

        # Create smooth 3D coil sensitivities
        kspace = np.zeros((n_coils, nz, ny, nx), dtype=complex)

        for i in range(n_coils):
            angle_xy = 2 * np.pi * i / n_coils
            angle_z = np.pi * (i % 3) / 3  # Vary in z as well

            coil_x = 0.8 * np.cos(angle_xy)
            coil_y = 0.8 * np.sin(angle_xy)
            coil_z = 0.5 * np.sin(angle_z)

            r_coil = np.sqrt((x - coil_x) ** 2 + (y - coil_y) ** 2 + (z - coil_z) ** 2)
            sens_mag = np.exp(-(r_coil**2) / 2.0)
            sens_phase = 0.2 * (
                x * np.cos(angle_xy) + y * np.sin(angle_xy) + z * np.sin(angle_z)
            )

            sensitivity = sens_mag * np.exp(1j * sens_phase)
            image = sensitivity * phantom

            # Transform to k-space
            kspace[i] = fft3c(image)

        # Add noise
        kspace += 0.01 * (
            np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape)
        )

        print(f"K-space shape: {kspace.shape}")
        print()

        # Run ESPIRiT 3D - using integer shorthand for sizes
        sens_maps, info = espirit(
            kspace,
            calib_size=16,  # Will expand to (16, 16, 16)
            kernel_size=4,  # Will expand to (4, 4, 4)
            num_maps=1,
            threshold=0.02,
            crop_threshold=0.85,
            normalize=True,
        )

        print("\nResults (3D):")
        print(f"  Sensitivity maps shape: {sens_maps.shape}")
        print(
            f"  Eigenvalue range: [{info['eigenvalues'].min():.3f}, {info['eigenvalues'].max():.3f}]"
        )
        print(f"  Mask coverage: {np.sum(info['mask']) / info['mask'].size * 100:.1f}%")

    else:
        print("\n" + "=" * 70)
        print("TESTING 2D ESPIRiT (use --3d for 3D test)")
        print("=" * 70)

        # Create synthetic 2D test data
        n_coils = 8
        ny, nx = 128, 128

        print("\nCreating synthetic multi-coil k-space data...")

        # Create coordinate grids
        y, x = np.meshgrid(
            np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij"
        )
        r = np.sqrt(x**2 + y**2)

        # Simple disk phantom in image space
        phantom = (r < 0.6).astype(complex)

        # Create smooth coil sensitivities
        kspace = np.zeros((n_coils, ny, nx), dtype=complex)

        for i in range(n_coils):
            angle = 2 * np.pi * i / n_coils
            coil_x = 0.8 * np.cos(angle)
            coil_y = 0.8 * np.sin(angle)

            r_coil = np.sqrt((x - coil_x) ** 2 + (y - coil_y) ** 2)
            sens_mag = np.exp(-(r_coil**2) / 2.0)
            sens_phase = 0.2 * (x * np.cos(angle) + y * np.sin(angle))

            sensitivity = sens_mag * np.exp(1j * sens_phase)
            image = sensitivity * phantom

            # Transform to k-space
            kspace[i] = fft2c(image)

        # Add noise
        kspace += 0.01 * (
            np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape)
        )

        print(f"K-space shape: {kspace.shape}")
        print()

        # Run ESPIRiT - can also use None to use defaults (24, 24) and (6, 6)
        # Or use integers: calib_size=32 expands to (32, 32)
        sens_maps, info = espirit(
            kspace,
            calib_size=32,
            kernel_size=6,
            num_maps=1,
            threshold=0.02,
            crop_threshold=0.9,
            normalize=True,
        )

        print("\nResults (2D):")
        print(f"  Sensitivity maps shape: {sens_maps.shape}")
        print(
            f"  Eigenvalue range: [{info['eigenvalues'].min():.3f}, {info['eigenvalues'].max():.3f}]"
        )
        print(f"  Mask coverage: {np.sum(info['mask']) / info['mask'].size * 100:.1f}%")
