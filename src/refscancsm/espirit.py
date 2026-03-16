#!/usr/bin/env python3
"""
ESPIRiT coil sensitivity calibration.

Python implementation of ESPIRiT (Uecker et al., MRM 2014).
Based on the BART ecalib.c implementation.

Pipeline (see espirit()):
  1. Extract calibration region from k-space centre
  2. Build calibration matrix from overlapping k-space patches
  3. Compute signal-space kernels via eigendecomposition of the Gram matrix
  4. Transform kernels to image domain to obtain PSF-like basis functions
  5. Compute per-voxel covariance matrices from image-domain kernels
  6. Sinc-interpolate covariance to full image resolution and extract CSM
     as the dominant eigenvector at each voxel
  7. Mask sensitivity maps based on eigenvalue threshold
  8. Phase-align maps to remove global phase ambiguity   (optional)
  9. Normalise maps so RSS across coils = 1              (optional)
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import linalg
from tqdm import tqdm

from .utils import (
    cp,
    get_num_threads,
    get_verbose,
    gpu_available,
    ifft2c,
    ifft3c,
    timed,
    vprint,
)

DEFAULT_DTYPE = np.complex64


# =============================================================================
# PUBLIC API
# =============================================================================


def espirit(
    kspace,
    calib_size=None,
    kernel_size=None,
    threshold=0.001,
    mask_threshold=0.8,
    normalize=True,
    rotphase=True,
    orthiter=True,
    num_orthiter=30,
    soft_threshold=False,
):
    """
    Run the full ESPIRiT calibration pipeline and return coil sensitivity maps.

    Parameters
    ----------
    kspace : ndarray, shape (n_coils, *spatial_dims)
        Multi-coil k-space data. Can be 2D or 3D.
    calib_size : int or tuple, optional
        Size of the calibration region extracted from the k-space centre.
        A scalar is broadcast to all spatial dimensions. Default: 24.
    kernel_size : int or tuple, optional
        Size of the sliding-window GRAPPA-like kernel. Default: 6.
    threshold : float
        Relative singular-value threshold for retaining signal-space kernels.
        Only kernels whose singular value exceeds threshold * max are kept.
    mask_threshold : float
        Eigenvalue threshold below which sensitivity maps are zeroed out
        (background / air / low-SNR regions).
    normalize : bool
        If True, normalise maps so RSS across coils = 1 at each voxel.
    rotphase : bool
        If True, remove the global phase ambiguity using a calibration-based
        phase reference.
    orthiter : bool
        Use power iteration instead of full eigendecomposition (much faster
        when only the dominant eigenvector is needed).
    num_orthiter : int
        Number of power-iteration steps (30 is sufficient for typical MRI data).
    soft_threshold : bool
        Use a smooth S-curve transition instead of a hard binary mask.

    Returns
    -------
    csm : ndarray, shape (n_coils, *spatial_dims)
        Coil sensitivity maps in the same geometry as kspace.
    """
    # Detect if input is on GPU and keep it there as long as possible
    is_gpu = hasattr(kspace, "get")
    xp = cp if is_gpu else np

    n_coils = kspace.shape[0]
    spatial_shape = kspace.shape[1:]
    n_dims = len(spatial_shape)

    # Keep data on original device
    if is_gpu:
        kspace = xp.asarray(kspace, dtype=xp.complex64)
    else:
        kspace = kspace.astype(DEFAULT_DTYPE)

    calib_size = _to_size_tuple(calib_size, n_dims, default=24)
    kernel_size = _to_size_tuple(kernel_size, n_dims, default=6)

    backend = "GPU" if gpu_available() else "CPU"
    vprint(
        f"  ESPIRiT ({n_dims}D, {backend})"
        f"  calib={'x'.join(str(c) for c in calib_size)}"
        f"  kernel={'x'.join(str(k) for k in kernel_size)}"
        f"  threshold={threshold}"
    )

    with timed("1. Extract calibration region from k-space centre"):
        calib_data = _extract_calibration_region(kspace, calib_size)

    with timed("2. Build calibration matrix from overlapping patches"):
        cal_matrix = _build_calibration_matrix(calib_data, kernel_size)

    with timed("3. Compute signal-space kernels (eigendecomposition of Gram matrix)"):
        kernels, _ = _compute_kernel_subspace(cal_matrix, threshold=threshold)
    vprint(f"     ({kernels.shape[0]} kernels retained)")

    with timed("4. Transform kernels to image domain"):
        img_kernels = _transform_kernels_to_image_domain(kernels, kernel_size, n_coils)

    with timed("5. Compute per-voxel covariance matrices"):
        img_cov = _compute_image_domain_covariance(img_kernels, kernel_size)

    with timed("6. Sinc-interpolate covariance and extract sensitivity maps"):
        csm, eigenvalues = _interpolate_covariance_and_extract_csm(
            img_cov, spatial_shape, orthiter=orthiter, num_orthiter=num_orthiter
        )

    with timed("7. Mask sensitivity maps by eigenvalue threshold"):
        csm = _mask_sensitivity_maps(csm, eigenvalues, mask_threshold, soft_threshold)

    if rotphase:
        with timed("8. Phase-align sensitivity maps"):
            rotation_matrix = _build_phase_rotation_matrix(calib_data)
            csm = _apply_phase_rotation(csm, rotation_matrix)

    if normalize:
        with timed("9. Normalise sensitivity maps (RSS)"):
            csm = _normalize_sensitivity_maps(csm)

    # Return only the first set of maps (dominant eigenvector)
    return csm[0]


# =============================================================================
# STEPS 1–5: BUILDING THE COVARIANCE MATRIX
# =============================================================================


def _to_size_tuple(value, n_dims, default):
    """Convert a size parameter to a tuple of length n_dims."""
    if value is None:
        return tuple([default] * n_dims)
    if isinstance(value, int):
        return tuple([value] * n_dims)
    return value


def _extract_calibration_region(kspace, calib_size):
    """
    Extract the central calib_size region from k-space.

    ESPIRiT only needs a small fully-sampled ACS region centred in k-space.
    The centre carries the most energy and captures coil geometry without
    needing the rest of k-space (which may be undersampled in accelerated scans).
    """
    xp = cp.get_array_module(kspace) if cp is not None else np
    spatial_dims = kspace.shape[1:]
    slices = [slice(None)]
    for full_size, cal_size_i in zip(spatial_dims, calib_size):
        center = full_size // 2
        start = center - (cal_size_i // 2)
        slices.append(slice(start, start + cal_size_i))
    return xp.asarray(kspace[tuple(slices)])


def _build_calibration_matrix(calib_data, kernel_size):
    """
    Build a Casorati-style matrix where each row is a flattened k-space patch.

    The sliding window sweeps the kernel across the ACS region; each position
    yields one row containing the (n_coils * kernel_elements) k-space values.
    The SVD of this matrix reveals the GRAPPA-like interpolation kernels used
    by ESPIRiT to estimate coil sensitivities.
    """
    # Move to CPU for patch extraction (slicing is complex on GPU)
    xp = cp.get_array_module(calib_data) if cp is not None else np
    if xp != np:
        calib_data = calib_data.get()

    n_coils = calib_data.shape[0]
    spatial_shape = calib_data.shape[1:]
    n_dims = len(spatial_shape)

    n_patches_per_dim = [spatial_shape[i] - kernel_size[i] + 1 for i in range(n_dims)]
    n_patches = int(np.prod(n_patches_per_dim))
    kernel_elements = n_coils * int(np.prod(kernel_size))
    cal_matrix = np.zeros((n_patches, kernel_elements), dtype=complex)

    if n_dims == 2:
        patch_idx = 0
        for py in range(n_patches_per_dim[0]):
            for px in range(n_patches_per_dim[1]):
                cal_matrix[patch_idx] = calib_data[
                    :, py : py + kernel_size[0], px : px + kernel_size[1]
                ].ravel()
                patch_idx += 1
    elif n_dims == 3:
        patch_idx = 0
        for pz in range(n_patches_per_dim[0]):
            for py in range(n_patches_per_dim[1]):
                for px in range(n_patches_per_dim[2]):
                    cal_matrix[patch_idx] = calib_data[
                        :,
                        pz : pz + kernel_size[0],
                        py : py + kernel_size[1],
                        px : px + kernel_size[2],
                    ].ravel()
                    patch_idx += 1
    else:
        raise ValueError(f"Only 2D and 3D data supported, got {n_dims}D")

    return cal_matrix


def _compute_kernel_subspace(cal_matrix, threshold=0.01):
    """
    Find the signal-space kernels via eigendecomposition of the Gram matrix.

    Rather than a full SVD, we form A^H A and compute its eigenvectors.
    Eigenvectors with large eigenvalues represent true signal correlations;
    those below the threshold are noise and are discarded. GPU is used when
    available (complex128 is downcast to complex64 to halve memory bandwidth).
    """
    if gpu_available():
        dtype = (
            cp.complex64
            if cal_matrix.dtype in (np.complex128, np.dtype("complex128"))
            else None
        )
        cal_gpu = (
            cp.asarray(cal_matrix, dtype=dtype) if dtype else cp.asarray(cal_matrix)
        )
        w, v = cp.linalg.eigh(cal_gpu.conj().T @ cal_gpu)
        eigenvalues, eigenvectors = cp.asnumpy(w), cp.asnumpy(v)
    else:
        eigenvalues, eigenvectors = linalg.eigh(cal_matrix.conj().T @ cal_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    svals = np.sqrt(np.abs(eigenvalues))
    max_sval = svals[0]
    n_keep = (
        int(np.sum(svals / max_sval > np.sqrt(threshold))) if max_sval > 1e-12 else 1
    )
    n_keep = max(1, n_keep)

    kernels = eigenvectors[:, :n_keep].T.conj()
    return kernels, svals


def _transform_kernels_to_image_domain(kernels, kernel_size, n_coils):
    """
    Zero-pad each k-space kernel and IFFT to obtain image-domain basis functions.

    The kernel is zero-padded to twice its size before IFFTing. This doubling
    avoids circular-convolution aliasing when computing H^H H in the next step.
    The outer product of these image-domain kernels gives the per-voxel covariance.
    """
    n_kernels = kernels.shape[0]
    n_dims = len(kernel_size)
    img_size = tuple(2 * k for k in kernel_size)
    starts = tuple((img_size[i] - kernel_size[i]) // 2 for i in range(n_dims))

    kernels_spatial = kernels.reshape((n_kernels, n_coils) + kernel_size)
    img_kernels = np.zeros((n_kernels, n_coils) + img_size, dtype=complex)

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


def _compute_image_domain_covariance(img_kernels, kernel_size):
    """
    Compute the per-voxel Hermitian covariance matrix H^H H from image-domain kernels.

    At each spatial position, img_kernels holds (n_kernels, n_coils) complex values.
    The n_coils x n_coils matrix H^H H captures how signal energy correlates across
    coils given the local k-space structure. The dominant eigenvector of this matrix
    IS the coil sensitivity at that voxel — the core identity of ESPIRiT (eq. 9).

    This covariance lives on a coarse grid of size 2*kernel_size (e.g. 12^3 for the
    default kernel_size=6). Sinc-interpolation in step 6 brings it to full resolution.
    """
    n_dims = len(img_kernels.shape) - 2
    normalization = np.prod(kernel_size) / np.prod(img_kernels.shape[2:]) ** 2
    axes_order = list(range(2, 2 + n_dims)) + [0, 1]
    H_t = np.transpose(img_kernels, axes_order)
    return np.einsum("...ki,...kj->...ij", H_t.conj(), H_t) / normalization


# =============================================================================
# STEP 6: SINC-INTERPOLATE COVARIANCE TO FULL RESOLUTION + EXTRACT CSM
# =============================================================================


def _interpolate_covariance_and_extract_csm(
    img_cov, target_shape, orthiter=True, num_orthiter=30
):
    """
    Sinc-interpolate the covariance to full image resolution and extract the CSM.

    Why interpolate and extract eigenmaps slice-by-slice instead of separately?
    Materialising the full-resolution covariance as (nz, ny, nx, nc, nc) before
    computing eigenmaps would require O(nz*ny*nx*nc^2) memory. For a modest
    acquisition (256x256x100, 32 coils) that is ~53 GB — clearly impractical.
    Instead: interpolate z for the whole volume at once (the source grid has only
    2*kernel_size_z rows, so this is cheap), then complete y/x interpolation and
    extract eigenmaps one z-slice at a time, keeping peak memory at one (ny,nx,nc,nc).

    Dispatches to GPU (3D only) or CPU based on gpu_available().
    """
    n_dims = img_cov.ndim - 2
    if gpu_available() and n_dims == 3:
        return _interpolate_covariance_and_extract_csm_gpu(
            img_cov, target_shape, orthiter, num_orthiter
        )
    return _interpolate_covariance_and_extract_csm_cpu(
        img_cov, target_shape, orthiter, num_orthiter
    )


def _build_sinc_interpolation_matrix(n_in, n_out):
    """
    Precompute a sinc interpolation matrix of shape (n_out, n_in).

    Expressing sinc interpolation as a dense matrix multiply is much faster than
    applying a 1D FFT repeatedly when the same resampling is applied to many
    vectors (e.g. all covariance entries for every slice). Built by interpolating
    each standard basis vector individually using FFT zero-padding.
    """
    if n_in == n_out:
        return np.eye(n_in, dtype=complex)

    M = np.zeros((n_out, n_in), dtype=complex)
    for j in range(n_in):
        e_j = np.zeros(n_in, dtype=complex)
        e_j[j] = 1.0
        # Sinc-interpolate the basis vector via FFT zero-padding
        tmp = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(e_j)))
        new_shape = np.zeros(n_out, dtype=complex)
        if n_out > n_in:
            start = (n_out - n_in) // 2
            new_shape[start : start + n_in] = tmp
        else:
            start = (n_in - n_out) // 2
            new_shape = tmp[start : start + n_out].copy()
        result = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(new_shape)))
        result *= n_out / n_in
        M[:, j] = result
    return M


def _interpolate_covariance_and_extract_csm_cpu(
    img_cov, target_shape, orthiter=True, num_orthiter=30
):
    """
    CPU: sinc-interpolate covariance to full resolution and extract eigenmaps.

    The upper triangle of the Hermitian covariance is packed first (nc*(nc+1)/2
    entries instead of nc^2 — about 2x memory saving during interpolation). The
    z-axis is interpolated for the whole volume via a single matrix multiply.
    For each z-slice, y/x interpolation is done and the full Hermitian matrix is
    reconstructed before computing the dominant eigenvector.
    """
    n_coils = img_cov.shape[-1]
    n_dims = img_cov.ndim - 2
    dtype = img_cov.dtype

    if n_dims == 2:
        ny, nx = target_shape
        cov_full = _sinc_interp_axis(img_cov, ny, axis=0)
        cov_full = _sinc_interp_axis(cov_full, nx, axis=1)
        return _compute_eigenmaps_batched(cov_full, orthiter, num_orthiter)

    elif n_dims == 3:
        nz, ny, nx = target_shape
        nz_s, ny_s, nx_s = img_cov.shape[:3]
        nc = n_coils

        # Pack upper triangle to halve memory during interpolation
        cosize = nc * (nc + 1) // 2
        cov_packed = np.zeros((nz_s, ny_s, nx_s, cosize), dtype=dtype)
        tri_i = np.zeros(cosize, dtype=int)
        tri_j = np.zeros(cosize, dtype=int)
        idx = 0
        for i in range(nc):
            for j in range(i + 1):
                cov_packed[..., idx] = img_cov[..., i, j]
                tri_i[idx] = i
                tri_j[idx] = j
                idx += 1

        # Precompute sinc interpolation matrices (matrix multiply is faster than
        # repeated 1D FFTs when the same resampling is applied to many vectors)
        M_z = _build_sinc_interpolation_matrix(nz_s, nz).astype(dtype)
        M_y = _build_sinc_interpolation_matrix(ny_s, ny).astype(dtype)
        M_x = _build_sinc_interpolation_matrix(nx_s, nx).astype(dtype)

        # Interpolate z for the whole volume at once (cheapest axis; only nz_s rows)
        cov_z = (M_z @ cov_packed.reshape(nz_s, -1)).reshape(nz, ny_s, nx_s, cosize)

        csm = np.zeros((1, nc, nz, ny, nx), dtype=dtype)
        eigenvalues = np.zeros((1, nz, ny, nx))

        for z in tqdm(
            range(nz),
            desc="  Computing eigenmaps (CPU, slice-by-slice)",
            disable=not get_verbose(),
        ):
            slc = (M_y @ cov_z[z].reshape(ny_s, -1)).reshape(ny, nx_s, cosize)
            slc = (
                (M_x @ slc.transpose(1, 0, 2).reshape(nx_s, -1))
                .reshape(nx, ny, cosize)
                .transpose(1, 0, 2)
            )

            # Reconstruct full Hermitian matrix from upper triangle
            cov_full = np.zeros((ny, nx, nc, nc), dtype=dtype)
            cov_full[..., tri_i, tri_j] = slc
            cov_full[..., tri_j, tri_i] = slc.conj()

            s, e = _compute_eigenmaps_batched(cov_full, orthiter, num_orthiter)
            csm[:, :, z] = s
            eigenvalues[:, z] = e

        return csm, eigenvalues
    else:
        raise ValueError(f"Only 2D and 3D supported, got {n_dims}D")


def _sinc_interp_axis(data, target_size, axis):
    """Sinc-interpolate data along one axis via FFT zero-padding (CPU utility)."""
    source_size = data.shape[axis]
    if source_size == target_size:
        return data.copy()

    tmp = np.fft.ifftshift(data, axes=axis)
    tmp = np.fft.fft(tmp, axis=axis)
    tmp = np.fft.fftshift(tmp, axes=axis)

    new_shape = list(tmp.shape)
    new_shape[axis] = target_size
    if target_size > source_size:
        padded = np.zeros(new_shape, dtype=tmp.dtype)
        start = (target_size - source_size) // 2
        sl = [slice(None)] * tmp.ndim
        sl[axis] = slice(start, start + source_size)
        padded[tuple(sl)] = tmp
    else:
        start = (source_size - target_size) // 2
        sl = [slice(None)] * tmp.ndim
        sl[axis] = slice(start, start + target_size)
        padded = tmp[tuple(sl)].copy()

    result = np.fft.ifftshift(padded, axes=axis)
    result = np.fft.ifft(result, axis=axis)
    result = np.fft.fftshift(result, axes=axis)
    result *= target_size / source_size
    return result


def _compute_eigenmaps_batched(img_cov, orthiter=True, num_orthiter=30):
    """
    Extract the dominant eigenvector from a batch of per-voxel covariance matrices.

    When orthiter=True, uses multi-threaded power iteration, which is ~10x faster
    than full eigh for the dominant eigenvector. When False, falls back to numpy's
    eigh for correctness / debugging. The result is the coil sensitivity map.
    """
    n_coils = img_cov.shape[-1]
    spatial_shape = img_cov.shape[:-2]
    n_voxels = int(np.prod(spatial_shape))
    dtype = img_cov.dtype

    if orthiter:
        cov_flat = img_cov.reshape(n_voxels, n_coils, n_coils)
        nthreads = get_num_threads()
        if nthreads > 1 and n_voxels > 256:
            chunks = np.array_split(cov_flat, nthreads)
            with ThreadPoolExecutor(max_workers=nthreads) as executor:
                results = list(
                    executor.map(
                        _run_orthogonal_iteration_chunk,
                        [(c, 1, num_orthiter) for c in chunks],
                    )
                )
            all_vecs = np.concatenate([r[0] for r in results], axis=0)
            all_vals = np.concatenate([r[1] for r in results], axis=0)
        else:
            all_vecs, all_vals = _run_orthogonal_iteration_chunk(
                (cov_flat, 1, num_orthiter)
            )

        all_vecs = all_vecs.reshape(spatial_shape + (n_coils, 1))
        all_vals = all_vals.reshape(spatial_shape + (1,))

        csm = np.zeros((1, n_coils) + spatial_shape, dtype=dtype)
        eigenvalues = np.zeros((1,) + spatial_shape)
        csm[0] = np.moveaxis(all_vecs[..., 0], -1, 0)
        eigenvalues[0] = all_vals[..., 0]
    else:
        eigvals_all, eigvecs_all = np.linalg.eigh(img_cov)
        csm = np.zeros((1, n_coils) + spatial_shape, dtype=dtype)
        eigenvalues = np.zeros((1,) + spatial_shape)
        csm[0] = np.moveaxis(eigvecs_all[..., -1], -1, 0)
        eigenvalues[0] = eigvals_all[..., -1]

    return csm, eigenvalues


def _run_orthogonal_iteration_chunk(args):
    """
    CPU worker: batched power iteration on a chunk of voxel covariance matrices.

    Power iteration converges to the dominant eigenvector in ~30 steps for
    typical MRI covariance matrices. Running in parallel threads across spatial
    chunks provides near-linear speedup up to memory bandwidth saturation.
    """
    cov_chunk, num_maps, num_orthiter = args
    B, nc = cov_chunk.shape[0], cov_chunk.shape[1]

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


def _interpolate_covariance_and_extract_csm_gpu(
    img_cov, target_shape, orthiter=True, num_orthiter=30
):
    """
    GPU: sinc-interpolate covariance to full image resolution and extract eigenmaps (3D).

    Same slice-by-slice strategy as the CPU version: z-axis interpolation is done
    on CPU for the full volume (to avoid uploading the full 5D covariance to GPU),
    then each z-slice is transferred to GPU for y/x interpolation and eigenmap
    extraction via batched GPU power iteration.
    """
    n_coils = img_cov.shape[-1]
    n_dims = img_cov.ndim - 2
    dtype = img_cov.dtype

    # GPU runs in single precision
    if dtype in (np.complex128, np.dtype("complex128")):
        dtype = np.complex64
    real_dtype = np.float32 if dtype == np.complex64 else np.float64

    if n_dims != 3:
        raise ValueError("GPU path only supports 3D data.")

    nz, ny, nx = target_shape
    nz_s, ny_s, nx_s = img_cov.shape[:3]
    nc = n_coils

    # Pack upper triangle (same CPU-side packing as the CPU path)
    cosize = nc * (nc + 1) // 2
    cov_packed = np.zeros((nz_s, ny_s, nx_s, cosize), dtype=dtype)
    tri_i = np.zeros(cosize, dtype=int)
    tri_j = np.zeros(cosize, dtype=int)
    idx = 0
    for i in range(nc):
        for j in range(i + 1):
            cov_packed[..., idx] = img_cov[..., i, j]
            tri_i[idx] = i
            tri_j[idx] = j
            idx += 1

    M_z = _build_sinc_interpolation_matrix(nz_s, nz).astype(dtype)
    M_y_cpu = _build_sinc_interpolation_matrix(ny_s, ny).astype(dtype)
    M_x_cpu = _build_sinc_interpolation_matrix(nx_s, nx).astype(dtype)

    # Interpolate z on CPU to avoid uploading the full 5D tensor to GPU
    cov_z = (M_z @ cov_packed.reshape(nz_s, -1)).reshape(nz, ny_s, nx_s, cosize)

    # Upload per-slice interpolation matrices and index arrays to GPU (reused every slice)
    M_y_gpu = cp.asarray(M_y_cpu)
    M_x_gpu = cp.asarray(M_x_cpu)
    tri_i_gpu = cp.asarray(tri_i)
    tri_j_gpu = cp.asarray(tri_j)

    csm = np.zeros((1, nc, nz, ny, nx), dtype=dtype)
    eigenvalues = np.zeros((1, nz, ny, nx), dtype=real_dtype)

    for z in tqdm(
        range(nz),
        desc="  Computing eigenmaps (GPU, slice-by-slice)",
        disable=not get_verbose(),
    ):
        slc_gpu = cp.asarray(cov_z[z])
        slc_gpu = (M_y_gpu @ slc_gpu.reshape(ny_s, -1)).reshape(ny, nx_s, cosize)
        slc_gpu = (
            (M_x_gpu @ slc_gpu.transpose(1, 0, 2).reshape(nx_s, -1))
            .reshape(nx, ny, cosize)
            .transpose(1, 0, 2)
        )

        # Reconstruct full Hermitian matrix from upper triangle
        cov_full_gpu = cp.zeros((ny, nx, nc, nc), dtype=dtype)
        cov_full_gpu[..., tri_i_gpu, tri_j_gpu] = slc_gpu
        cov_full_gpu[..., tri_j_gpu, tri_i_gpu] = cp.conj(slc_gpu)

        if orthiter:
            vecs, vals = _run_power_iteration_gpu(
                cov_full_gpu.reshape(-1, nc, nc), num_iter=num_orthiter
            )
            csm[0, :, z] = cp.asnumpy(vecs.reshape(ny, nx, nc).transpose(2, 0, 1))
            eigenvalues[0, z] = cp.asnumpy(vals.reshape(ny, nx))
        else:
            w, v = cp.linalg.eigh(cov_full_gpu)
            csm[0, :, z] = cp.asnumpy(v[..., -1].transpose(2, 0, 1))
            eigenvalues[0, z] = cp.asnumpy(w[..., -1])

    return csm, eigenvalues


def _run_power_iteration_gpu(cov, num_iter=30):
    """
    GPU batched power iteration to find the dominant eigenvector of each covariance matrix.

    Runs entirely on GPU to avoid per-voxel Python overhead. Random complex
    initialisation avoids pathological starting points that could stall convergence.
    Converges in ~30 steps for typical MRI covariance matrices.
    """
    n_pixels, n_coils, _ = cov.shape
    real_dtype = np.float32 if cov.dtype == np.complex64 else np.float64

    v = cp.random.randn(n_pixels, n_coils, 1, dtype=real_dtype) + 1j * cp.random.randn(
        n_pixels, n_coils, 1, dtype=real_dtype
    )
    v = v / (cp.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    for _ in range(num_iter):
        v = cp.matmul(cov, v)
        v = v / (cp.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    Av = cp.matmul(cov, v)
    eig_val = cp.sum(cp.conj(v) * Av, axis=1).real.reshape(n_pixels)
    eig_vec = v.reshape(n_pixels, n_coils)
    return eig_vec, eig_val


# =============================================================================
# STEPS 7–9: POST-PROCESSING (always on CPU)
# =============================================================================


def _mask_sensitivity_maps(csm, eigenvalues, mask_threshold=0.8, soft_threshold=False):
    """
    Zero out (or smoothly downweight) voxels with eigenvalues below the threshold.

    Low eigenvalues indicate background / air / low-SNR voxels whose sensitivities
    should not contribute to SENSE reconstructions. soft_threshold replaces the
    binary mask with a smooth S-curve transition around mask_threshold.
    """
    dom_eigval = eigenvalues[0]
    if soft_threshold:
        weight = np.sqrt(np.abs(dom_eigval))
        weight = (weight - mask_threshold) / (1.0 - mask_threshold)

        def _scurve(x):
            result = np.zeros_like(x)
            mask_mid = x >= -1
            mask_high = x >= 1
            s = x / (x**2 + 1)
            result[mask_mid] = s[mask_mid]
            result[mask_high] += 1.0
            return result

        weight = _scurve(weight)
    else:
        weight = (np.abs(dom_eigval) >= mask_threshold).astype(float)

    # Broadcast (num_maps, n_coils, *spatial) * (*spatial)
    csm *= weight[np.newaxis, np.newaxis]
    return csm


def _build_phase_rotation_matrix(calib_data):
    """
    Compute a unitary matrix that aligns coil phases to a common reference.

    The dominant eigenvector of the calibration Gram matrix provides a stable,
    data-driven phase reference (analogous to the body coil reference in SENSE).
    Applying its conjugate removes the global phase ambiguity and yields smooth,
    near-real sensitivity maps.
    """
    # Convert to CPU if on GPU (scipy.linalg doesn't support CuPy)
    if hasattr(calib_data, "get"):
        calib_data = calib_data.get()

    n_coils = calib_data.shape[0]
    calib_flat = calib_data.reshape(n_coils, -1)
    gram = calib_flat @ calib_flat.conj().T / calib_flat.shape[1]
    _, eigenvectors = linalg.eigh(gram)
    return eigenvectors[:, ::-1]  # Sort descending by eigenvalue


def _apply_phase_rotation(csm, rotation_matrix):
    """
    Remove the global phase ambiguity using the calibration-based rotation matrix.

    At each voxel, the phase reference is the projection of the sensitivity vector
    onto the dominant coil eigenvector. Multiplying by its conjugate removes the
    voxel-wise phase offset.
    """
    num_maps, n_coils = csm.shape[0], csm.shape[1]
    csm_flat = csm.reshape(num_maps, n_coils, -1)
    ref_vec = rotation_matrix[:, 0].conj()
    for m in range(num_maps):
        phase_ref = np.einsum("c,cp->p", ref_vec, csm_flat[m])
        phase_ref /= np.abs(phase_ref) + 1e-10
        csm_flat[m] *= phase_ref.conj()[np.newaxis, :]
    return csm


def _normalize_sensitivity_maps(csm):
    """
    Normalise sensitivity maps so that the RSS across coils equals 1 at each voxel.

    Voxels with negligible signal (norm < 1e-10) are left unchanged to avoid
    division by near-zero values.
    """
    norm = np.sqrt(np.sum(np.abs(csm) ** 2, axis=1, keepdims=True))
    norm = np.where(norm > 1e-10, norm, 1.0)
    csm /= norm
    return csm
