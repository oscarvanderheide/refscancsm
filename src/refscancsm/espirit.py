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

# --- GPU SUPPORT CHECK ---
try:
    import cupy as cp

    HAVE_CUPY = True
except ImportError:
    cp = None
    HAVE_CUPY = False

# Match BART's single precision (complex float). Set to np.complex128 for double.
DEFAULT_DTYPE = np.complex64

# Number of threads for parallelizing per-voxel eigendecomposition (CPU mode).
# Set to 0 to auto-detect (uses cpu_count).
NUM_THREADS = 0


def _get_num_threads():
    """Get the number of threads to use for parallel eigenmaps."""
    if NUM_THREADS > 0:
        return NUM_THREADS
    return min(os.cpu_count() or 1, 16)


# =============================================================================
# FFT AND UTILS
# =============================================================================


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
    """Sinc interpolation along one axis via FFT zero-padding."""
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
        # Upsampling
        padded = np.zeros(new_shape, dtype=tmp.dtype)
        start = (target_size - source_size) // 2
        dst_sl = [slice(None)] * tmp.ndim
        dst_sl[axis] = slice(start, start + source_size)
        padded[tuple(dst_sl)] = tmp
    else:
        # Downsampling
        start = (source_size - target_size) // 2
        src_sl = [slice(None)] * tmp.ndim
        src_sl[axis] = slice(start, start + target_size)
        padded = tmp[tuple(src_sl)].copy()

    # Centered IFFT back
    result = np.fft.ifftshift(padded, axes=axis)
    result = np.fft.ifft(result, axis=axis)
    result = np.fft.fftshift(result, axes=axis)

    # Scale to preserve signal level
    result *= target_size / source_size

    return result


def compute_sinc_matrix(n_in, n_out):
    """Precompute sinc interpolation matrix of shape (n_out, n_in)."""
    if n_in == n_out:
        return np.eye(n_in, dtype=complex)

    M = np.zeros((n_out, n_in), dtype=complex)
    for j in range(n_in):
        e_j = np.zeros(n_in, dtype=complex)
        e_j[j] = 1.0
        M[:, j] = sinc_interp_1d(e_j, n_out, axis=0)
    return M


def extract_calib(kspace, calib_size):
    """Extract calibration region from center of k-space."""
    n_coils = kspace.shape[0]
    spatial_dims = kspace.shape[1:]
    n_dims = len(spatial_dims)

    if len(calib_size) != n_dims:
        raise ValueError(f"calib_size {calib_size} doesn't match {n_dims}D data")

    slices = [slice(None)]
    for i, (full_size, cal_size_i) in enumerate(zip(spatial_dims, calib_size)):
        start = (full_size - cal_size_i) // 2
        slices.append(slice(start, start + cal_size_i))

    return kspace[tuple(slices)].copy()


def build_calibration_matrix(calib_data, kernel_size):
    """Build calibration matrix using sliding window approach."""
    n_coils = calib_data.shape[0]
    spatial_shape = calib_data.shape[1:]
    n_dims = len(spatial_shape)

    if len(kernel_size) != n_dims:
        raise ValueError(f"kernel_size {kernel_size} doesn't match {n_dims}D data")

    n_patches_per_dim = [spatial_shape[i] - kernel_size[i] + 1 for i in range(n_dims)]
    n_patches = np.prod(n_patches_per_dim)
    kernel_elements = n_coils * np.prod(kernel_size)

    cal_matrix = np.zeros((n_patches, kernel_elements), dtype=complex)

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


# =============================================================================
# CPU IMPLEMENTATIONS (Fallback)
# =============================================================================


def compute_kernels_svd(cal_matrix, threshold=0.01, num_kernels=None):
    """CPU: Compute ESPIRiT kernels via eigendecomposition of Gram matrix."""
    gram = cal_matrix.conj().T @ cal_matrix
    eigenvalues, eigenvectors = linalg.eigh(gram)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    svals = np.sqrt(np.abs(eigenvalues))

    if num_kernels is not None:
        n_keep = num_kernels
    else:
        max_sval = svals[0]
        n_keep = np.sum(svals / max_sval > np.sqrt(threshold))

    n_keep = max(1, n_keep)
    print(f"  [CPU] Keeping {n_keep} kernels (threshold: {threshold})")
    print(f"  [CPU] Singular value range: [{svals[n_keep - 1]:.6f}, {svals[0]:.6f}]")

    kernels = eigenvectors[:, :n_keep].T.conj()
    return kernels, svals


def gram_schmidt(vectors):
    """CPU: Gram-Schmidt orthonormalization."""
    n, m = vectors.shape
    ortho = vectors.copy()
    norms = np.zeros(m)
    for j in range(m):
        for i in range(j):
            proj = np.vdot(ortho[:, i], ortho[:, j])
            ortho[:, j] -= proj * ortho[:, i]
        norm = np.linalg.norm(ortho[:, j])
        if norm > 1e-10:
            ortho[:, j] /= norm
        norms[j] = norm
    return ortho, norms


def _orthiter_chunk(args):
    """CPU: Worker function for threaded orthogonal iteration."""
    cov_chunk, num_maps, num_orthiter = args
    B, nc = cov_chunk.shape[0], cov_chunk.shape[1]

    if num_maps == 1:
        v = np.zeros((B, nc), dtype=cov_chunk.dtype)
        v[:, 0] = 1.0
        for _ in range(num_orthiter):
            v = np.einsum("...ij,...j->...i", cov_chunk, v)
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            norm = np.where(norm > 1e-30, norm, 1.0)
            v = v / norm
        Av = np.einsum("...ij,...j->...i", cov_chunk, v)
        eigvals = np.real(np.sum(v.conj() * Av, axis=-1, keepdims=True))
        return v[..., np.newaxis], eigvals
    else:
        vecs = np.zeros((B, nc, num_maps), dtype=cov_chunk.dtype)
        for m in range(num_maps):
            vecs[:, m % nc, m] = 1.0
        for _ in range(num_orthiter):
            vecs = np.einsum("...ij,...jk->...ik", cov_chunk, vecs)
            for m in range(num_maps):
                v = vecs[..., m]
                for j in range(m):
                    u = vecs[..., j]
                    proj = np.sum(u.conj() * v, axis=-1, keepdims=True)
                    v = v - proj * u
                norm = np.linalg.norm(v, axis=-1, keepdims=True)
                norm = np.where(norm > 1e-30, norm, 1.0)
                vecs[..., m] = v / norm
        Av = np.einsum("...ij,...jk->...ik", cov_chunk, vecs)
        eigvals = np.real(np.sum(vecs.conj() * Av, axis=-2))
        return vecs, eigvals


def eigenmaps_batched(img_cov, num_maps=1, orthiter=True, num_orthiter=30):
    """CPU: Batched eigendecomposition of covariance matrices."""
    n_coils = img_cov.shape[-1]
    spatial_shape = img_cov.shape[:-2]
    n_voxels = int(np.prod(spatial_shape))
    dtype = img_cov.dtype

    if orthiter:
        cov_flat = img_cov.reshape(n_voxels, n_coils, n_coils)
        nthreads = _get_num_threads()
        if nthreads > 1 and n_voxels > 256:
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
            all_vecs, all_vals = _orthiter_chunk((cov_flat, num_maps, num_orthiter))

        all_vecs = all_vecs.reshape(spatial_shape + (n_coils, num_maps))
        all_vals = all_vals.reshape(spatial_shape + (num_maps,))

        sens_maps = np.zeros((num_maps, n_coils) + spatial_shape, dtype=dtype)
        eigenvalues = np.zeros((num_maps,) + spatial_shape)
        for m in range(num_maps):
            sens_maps[m] = np.moveaxis(all_vecs[..., m], -1, 0)
            eigenvalues[m] = all_vals[..., m]
    else:
        eigvals_all, eigvecs_all = np.linalg.eigh(img_cov)
        sens_maps = np.zeros((num_maps, n_coils) + spatial_shape, dtype=dtype)
        eigenvalues = np.zeros((num_maps,) + spatial_shape)
        for m in range(num_maps):
            idx = -(m + 1)
            sens_maps[m] = np.moveaxis(eigvecs_all[..., idx], -1, 0)
            eigenvalues[m] = eigvals_all[..., idx]

    return sens_maps, eigenvalues


def caltwo(img_cov, target_shape, num_maps=1, orthiter=True, num_orthiter=30):
    """CPU: Resize covariance to full resolution and compute eigenmaps."""
    n_coils = img_cov.shape[-1]
    n_dims = img_cov.ndim - 2
    dtype = img_cov.dtype

    if n_dims == 2:
        ny, nx = target_shape
        cov_full = sinc_interp_1d(img_cov, ny, axis=0)
        cov_full = sinc_interp_1d(cov_full, nx, axis=1)
        return eigenmaps_batched(cov_full, num_maps, orthiter, num_orthiter)

    elif n_dims == 3:
        nz, ny, nx = target_shape
        nz_s, ny_s, nx_s = img_cov.shape[:3]
        nc = n_coils

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

        M_z = compute_sinc_matrix(nz_s, nz).astype(dtype)
        M_y = compute_sinc_matrix(ny_s, ny).astype(dtype)
        M_x = compute_sinc_matrix(nx_s, nx).astype(dtype)

        print(f"  [CPU] Resizing packed covariance along z ({nz_s} -> {nz})...")
        cov_z = (M_z @ cov_packed.reshape(nz_s, -1)).reshape(nz, ny_s, nx_s, cosize)

        sens_maps = np.zeros((num_maps, nc, nz, ny, nx), dtype=dtype)
        eigenvalues = np.zeros((num_maps, nz, ny, nx))

        t0 = time.perf_counter()
        for z in range(nz):
            slc = cov_z[z]
            slc = (M_y @ slc.reshape(ny_s, -1)).reshape(ny, nx_s, cosize)
            slc = (
                (M_x @ slc.transpose(1, 0, 2).reshape(nx_s, -1))
                .reshape(nx, ny, cosize)
                .transpose(1, 0, 2)
            )

            cov_full = np.zeros((ny, nx, nc, nc), dtype=dtype)
            cov_full[..., tri_i, tri_j] = slc
            cov_full[..., tri_j, tri_i] = slc.conj()

            s, e = eigenmaps_batched(cov_full, num_maps, orthiter, num_orthiter)
            sens_maps[:, :, z] = s
            eigenvalues[:, z] = e

            if (z + 1) % 10 == 0 or z == nz - 1:
                elapsed = time.perf_counter() - t0
                print(f"    [CPU] Slice {z + 1}/{nz}: {elapsed:.1f}s elapsed")

        return sens_maps, eigenvalues
    else:
        raise ValueError(f"Only 2D and 3D supported, got {n_dims}D")


# =============================================================================
# GPU IMPLEMENTATIONS (Optimized)
# =============================================================================


def compute_kernels_svd_gpu(cal_matrix, threshold=0.01, num_kernels=None):
    """GPU: Accelerated computation of ESPIRiT kernels."""
    if not HAVE_CUPY:
        raise RuntimeError("CuPy not available")

    # Prefer complex64 for speed
    if cal_matrix.dtype == np.complex128 or cal_matrix.dtype == np.dtype("complex128"):
        print("  Warning: Converting cal_matrix to complex64 for GPU speed")
        cal_matrix_gpu = cp.asarray(cal_matrix, dtype=cp.complex64)
    else:
        cal_matrix_gpu = cp.asarray(cal_matrix)

    print("  [GPU] Computing Gram matrix...")
    gram_gpu = cal_matrix_gpu.conj().T @ cal_matrix_gpu

    print("  [GPU] Computing Eigendecomposition...")
    w_gpu, v_gpu = cp.linalg.eigh(gram_gpu)

    eigenvalues = cp.asnumpy(w_gpu)
    eigenvectors = cp.asnumpy(v_gpu)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    svals = np.sqrt(np.abs(eigenvalues))

    if num_kernels is not None:
        n_keep = num_kernels
    else:
        max_sval = svals[0]
        if max_sval > 1e-12:
            n_keep = np.sum(svals / max_sval > np.sqrt(threshold))
        else:
            n_keep = 1

    n_keep = max(1, n_keep)
    print(f"  [GPU] Keeping {n_keep} kernels (threshold: {threshold})")
    print(f"  [GPU] Singular value range: [{svals[n_keep - 1]:.6f}, {svals[0]:.6f}]")

    kernels = eigenvectors[:, :n_keep].T.conj()
    return kernels, svals


def power_iteration_gpu(cov, num_iter=30):
    """GPU: Batched Power Iteration."""
    n_pixels, n_coils, _ = cov.shape

    if cov.dtype == np.complex64 or cov.dtype == np.dtype("complex64"):
        real_dtype = np.float32
    else:
        real_dtype = np.float64

    # Generate real and imaginary parts separately (fix for CuPy random)
    real_part = cp.random.randn(n_pixels, n_coils, 1, dtype=real_dtype)
    imag_part = cp.random.randn(n_pixels, n_coils, 1, dtype=real_dtype)
    v = real_part + 1j * imag_part

    norm = cp.linalg.norm(v, axis=1, keepdims=True)
    v = v / (norm + 1e-12)

    for _ in range(num_iter):
        v = cp.matmul(cov, v)
        norm = cp.linalg.norm(v, axis=1, keepdims=True)
        v = v / (norm + 1e-12)

    Av = cp.matmul(cov, v)
    eig_val = cp.sum(cp.conj(v) * Av, axis=1).real.reshape(n_pixels)
    eig_vec = v.reshape(n_pixels, n_coils)

    return eig_vec, eig_val


def caltwo_gpu(img_cov, target_shape, num_maps=1, orthiter=True, num_orthiter=30):
    """GPU: Highly Optimized version of caltwo."""
    if not HAVE_CUPY:
        raise RuntimeError("CuPy not available")

    n_coils = img_cov.shape[-1]
    n_dims = img_cov.ndim - 2
    dtype = img_cov.dtype

    if dtype == np.complex128 or dtype == np.dtype("complex128"):
        print("  Warning: Converting to complex64 for GPU speed")
        dtype = np.complex64
        real_dtype = np.float32
    elif dtype == np.complex64 or dtype == np.dtype("complex64"):
        real_dtype = np.float32
    else:
        real_dtype = np.float64

    if n_dims == 3:
        nz, ny, nx = target_shape
        nz_s, ny_s, nx_s = img_cov.shape[:3]
        nc = n_coils

        # Pack covariance to upper triangle (CPU)
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

        M_z = compute_sinc_matrix(nz_s, nz).astype(dtype)
        M_y_cpu = compute_sinc_matrix(ny_s, ny).astype(dtype)
        M_x_cpu = compute_sinc_matrix(nx_s, nx).astype(dtype)

        print(f"  [GPU] Resizing Z on CPU ({nz_s} -> {nz})...")
        cov_z = (M_z @ cov_packed.reshape(nz_s, -1)).reshape(nz, ny_s, nx_s, cosize)

        sens_maps = np.zeros((num_maps, nc, nz, ny, nx), dtype=dtype)
        eigenvalues = np.zeros((num_maps, nz, ny, nx), dtype=real_dtype)

        M_y_gpu = cp.asarray(M_y_cpu)
        M_x_gpu = cp.asarray(M_x_cpu)
        tri_i_gpu = cp.asarray(tri_i)
        tri_j_gpu = cp.asarray(tri_j)

        print(f"  [GPU] Processing {nz} slices using Power Iteration...")
        t0 = time.perf_counter()

        for z in range(nz):
            slc_gpu = cp.asarray(cov_z[z])
            slc_gpu = (M_y_gpu @ slc_gpu.reshape(ny_s, -1)).reshape(ny, nx_s, cosize)
            slc_gpu = (
                (M_x_gpu @ slc_gpu.transpose(1, 0, 2).reshape(nx_s, -1))
                .reshape(nx, ny, cosize)
                .transpose(1, 0, 2)
            )

            cov_full_gpu = cp.zeros((ny, nx, nc, nc), dtype=dtype)
            cov_full_gpu[..., tri_i_gpu, tri_j_gpu] = slc_gpu
            cov_full_gpu[..., tri_j_gpu, tri_i_gpu] = cp.conj(slc_gpu)

            cov_flat = cov_full_gpu.reshape(-1, nc, nc)

            if num_maps == 1 and orthiter:
                vecs, vals = power_iteration_gpu(cov_flat, num_iter=num_orthiter)
                vecs = vecs.reshape(ny, nx, nc)
                vals = vals.reshape(ny, nx)
                sens_maps[0, :, z, :, :] = cp.asnumpy(vecs.transpose(2, 0, 1))
                eigenvalues[0, z, :, :] = cp.asnumpy(vals)
            else:
                w, v = cp.linalg.eigh(cov_full_gpu)
                maps_slice = v[..., -num_maps:].transpose(3, 2, 0, 1)
                vals_slice = w[..., -num_maps:].transpose(2, 0, 1)
                sens_maps[:, :, z, :, :] = cp.asnumpy(maps_slice)
                eigenvalues[:, z, :, :] = cp.asnumpy(vals_slice)

            if (z + 1) % 5 == 0 or z == nz - 1:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (z + 1) * (nz - z - 1)
                print(
                    f"    Slice {z + 1}/{nz}: {elapsed:.1f}s elapsed, ~{eta:.0f}s left"
                )

        return sens_maps, eigenvalues
    else:
        # Fallback for 2D to CPU or implement 2D GPU logic if needed
        # For now, we raise error to ensure we don't silently fail,
        # but in espirit() we handle the dispatch logic.
        raise ValueError("GPU optimization currently focuses on 3D data.")


# =============================================================================
# COMMON POST-PROCESSING
# =============================================================================


def kernels_to_image_domain(kernels, kernel_size, n_coils, img_size):
    """Transform kernels from k-space to image domain."""
    n_kernels = kernels.shape[0]
    n_dims = len(kernel_size)
    spatial_shape = (n_kernels, n_coils) + kernel_size
    kernels_spatial = kernels.reshape(spatial_shape)
    full_shape = (n_kernels, n_coils) + img_size
    img_kernels = np.zeros(full_shape, dtype=complex)
    starts = tuple((img_size[i] - kernel_size[i]) // 2 for i in range(n_dims))

    if n_dims == 2:
        img_kernels[
            :,
            :,
            starts[0] : starts[0] + kernel_size[0],
            starts[1] : starts[1] + kernel_size[1],
        ] = kernels_spatial
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
        for k in range(n_kernels):
            for c in range(n_coils):
                img_kernels[k, c] = ifft3c(img_kernels[k, c])
    return img_kernels


def compute_image_covariance(img_kernels, kernel_size):
    """Compute point-wise covariance matrices in image domain."""
    n_dims = len(img_kernels.shape) - 2
    image_volume = np.prod(img_kernels.shape[2:])
    kernel_volume = np.prod(kernel_size)
    normalization = kernel_volume / (image_volume**2)

    axes_order = list(range(2, 2 + n_dims)) + [0, 1]
    H_t = np.transpose(img_kernels, axes_order)
    img_cov = np.einsum("...ki,...kj->...ij", H_t.conj(), H_t) / normalization
    return img_cov


def crop_sensitivity_maps(sens_maps, eigenvalues, crop_threshold=0.8, softcrop=False):
    """Crop/weight sensitivity maps based on eigenvalue threshold."""
    eigmap = eigenvalues[0]
    if softcrop:
        weight = np.sqrt(np.abs(eigmap))
        weight = (weight - crop_threshold) / (1.0 - crop_threshold)

        def bart_scurve(x):
            result = np.zeros_like(x)
            mask_mid = x >= -1
            mask_high = x >= 1
            s = x / (x**2 + 1)
            result[mask_mid] = s[mask_mid]
            result[mask_high] += 1.0
            return result

        weight = bart_scurve(weight)
    else:
        weight = (np.abs(eigmap) >= crop_threshold).astype(float)

    sens_cropped = sens_maps.copy()
    for m in range(sens_maps.shape[0]):
        sens_cropped[m] = sens_maps[m] * weight[None, ...]
    return sens_cropped, weight


def compute_phase_rotation_matrix(calib_data):
    """Compute phase rotation matrix for coil phase alignment."""
    n_coils = calib_data.shape[0]
    calib_flat = calib_data.reshape(n_coils, -1)
    gram = calib_flat @ calib_flat.conj().T / calib_flat.shape[1]
    eigenvalues, eigenvectors = linalg.eigh(gram)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors


def fix_phase(sens_maps, rotation_matrix):
    """Apply phase rotation to align coil phases."""
    num_maps, n_coils = sens_maps.shape[0], sens_maps.shape[1]
    sens_flat = sens_maps.reshape(num_maps, n_coils, -1)
    sens_rotated = np.zeros_like(sens_flat)
    for m in range(num_maps):
        phase_ref = np.einsum("c,cp->p", rotation_matrix[:, 0].conj(), sens_flat[m])
        phase_ref = phase_ref / (np.abs(phase_ref) + 1e-10)
        sens_rotated[m] = sens_flat[m] * phase_ref.conj()[None, :]
    return sens_rotated.reshape(sens_maps.shape)


def normalize_sensitivity_maps(sens_maps, mode="rss"):
    """Normalize sensitivity maps."""
    sens_normalized = sens_maps.copy()
    for m in range(sens_maps.shape[0]):
        if mode == "rss":
            norm = np.sqrt(np.sum(np.abs(sens_maps[m]) ** 2, axis=0, keepdims=True))
        elif mode == "l1":
            norm = np.sum(np.abs(sens_maps[m]), axis=0, keepdims=True)
        norm = np.where(norm > 1e-10, norm, 1.0)
        sens_normalized[m] = sens_maps[m] / norm
    return sens_normalized


# =============================================================================
# MAIN ESPIRIT FUNCTION
# =============================================================================


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
    use_gpu=True,  # New parameter
):
    """
    Main ESPIRiT function.

    Parameters:
    -----------
    ... (standard params) ...
    use_gpu : bool
        If True, attempts to use CuPy for acceleration. Falls back to CPU if CuPy is missing.
    """
    n_coils = kspace.shape[0]
    spatial_shape = kspace.shape[1:]
    n_dims = len(spatial_shape)

    # GPU Availability Check
    if use_gpu and not HAVE_CUPY:
        print("Warning: GPU requested but CuPy not found. Falling back to CPU.")
        use_gpu = False

    # Cast to working dtype
    kspace = kspace.astype(DEFAULT_DTYPE)

    # Auto-detect sizes
    if calib_size is None:
        calib_size = tuple([24] * n_dims)
    elif isinstance(calib_size, int):
        calib_size = tuple([calib_size] * n_dims)

    if kernel_size is None:
        kernel_size = tuple([6] * n_dims)
    elif isinstance(kernel_size, int):
        kernel_size = tuple([kernel_size] * n_dims)

    print("=" * 70)
    print(f"ESPIRiT Calibration ({n_dims}D)")
    print(f"Mode: {'GPU (CuPy)' if use_gpu else 'CPU (NumPy/SciPy)'}")
    print("=" * 70)

    t_total = time.perf_counter()

    # Step 1: Extract calibration region
    t0 = time.perf_counter()
    print("Step 1: Extracting calibration region...")
    calib_data = extract_calib(kspace, calib_size)
    print(f"  [Step 1: {time.perf_counter() - t0:.2f}s]")

    # Step 2: Build calibration matrix
    t0 = time.perf_counter()
    print("\nStep 2: Building calibration matrix...")
    cal_matrix = build_calibration_matrix(calib_data, kernel_size)
    print(f"  [Step 2: {time.perf_counter() - t0:.2f}s]")

    # Step 3: Compute ESPIRiT kernels
    t0 = time.perf_counter()
    print("\nStep 3: Computing ESPIRiT kernels...")
    if use_gpu:
        kernels, svals = compute_kernels_svd_gpu(cal_matrix, threshold=threshold)
    else:
        kernels, svals = compute_kernels_svd(cal_matrix, threshold=threshold)
    print(f"  [Step 3: {time.perf_counter() - t0:.2f}s]")

    # Step 4: Transform kernels to image domain
    t0 = time.perf_counter()
    print("\nStep 4: Transforming kernels to image domain...")
    img_size = tuple(2 * k for k in kernel_size)
    img_kernels = kernels_to_image_domain(kernels, kernel_size, n_coils, img_size)
    print(f"  [Step 4: {time.perf_counter() - t0:.2f}s]")

    # Step 5: Compute image-space covariance
    t0 = time.perf_counter()
    print("\nStep 5: Computing image-space covariance matrices...")
    img_cov = compute_image_covariance(img_kernels, kernel_size)
    print(f"  [Step 5: {time.perf_counter() - t0:.2f}s]")

    # Steps 6-7: Resize covariance + eigenmaps
    t0 = time.perf_counter()
    print("\nSteps 6-7: Resize covariance to full resolution + eigenmaps...")

    if use_gpu and n_dims == 3:
        # Use optimized GPU path for 3D
        sens_maps, eigenvalues = caltwo_gpu(
            img_cov,
            spatial_shape,
            num_maps=num_maps,
            orthiter=orthiter,
            num_orthiter=num_orthiter,
        )
    elif use_gpu and n_dims == 2:
        # Fallback to CPU for 2D if GPU path not implemented for 2D in caltwo_gpu
        # (Or you can implement 2D logic in caltwo_gpu)
        print(
            "  Notice: Using CPU for 2D resize/eigenmaps (GPU path optimized for 3D)."
        )
        sens_maps, eigenvalues = caltwo(
            img_cov,
            spatial_shape,
            num_maps=num_maps,
            orthiter=orthiter,
            num_orthiter=num_orthiter,
        )
    else:
        # CPU Path
        sens_maps, eigenvalues = caltwo(
            img_cov,
            spatial_shape,
            num_maps=num_maps,
            orthiter=orthiter,
            num_orthiter=num_orthiter,
        )

    print(f"  [Steps 6-7: {time.perf_counter() - t0:.2f}s]")

    # Step 8: Post-processing
    t0 = time.perf_counter()
    print("\nStep 8: Post-processing...")
    sens_maps, mask = crop_sensitivity_maps(
        sens_maps, eigenvalues, crop_threshold, softcrop=softcrop
    )

    if rotphase:
        rotation_matrix = compute_phase_rotation_matrix(calib_data)
        sens_maps = fix_phase(sens_maps, rotation_matrix)

    if normalize:
        sens_maps = normalize_sensitivity_maps(sens_maps, mode="rss")

    print(f"  [Step 8: {time.perf_counter() - t0:.2f}s]")
    print("\n" + "=" * 70)
    print(f"ESPIRiT calibration complete! Total: {time.perf_counter() - t_total:.2f}s")
    print("=" * 70)

    # info = {
    #     "eigenvalues": eigenvalues,
    #     "mask": mask,
    #     "svals": svals,
    #     "img_cov": img_cov,
    #     "kernels": kernels,
    # }

    return sens_maps


if __name__ == "__main__":
    print("ESPIRiT Python Implementation (GPU/CPU Hybrid)")
    # Add your test code here if needed
