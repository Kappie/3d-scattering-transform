import numpy as np
import pyculib.fft
import scipy.fftpack
import time
import math

from numba import cuda, vectorize, jit


def extract_scattering_coefficients_cpu(X, phi, downsampling_resolution):
    """
    Assumes X and phi are in Fourier space.
    """
    return downsample(np.abs(scipy.fftpack.ifftn(X * phi)), downsampling_resolution)


def extract_scattering_coefficients(X, phi, downsampling_resolution):
    """
    Implemented on GPU.
    Assumes X and phi are in Fourier space.
    """
    x, y, z = X.shape
    X_gpu = cuda.to_device(X)
    phi_gpu = cuda.to_device(phi)

    blockspergrid, threadsperblock = get_blocks_and_threads(x, y, z)
    # Take elementwise product (corresponds to convolution in spatial domain). Store result in `X_gpu.`
    MultiplyInPlace[blockspergrid, threadsperblock](X_gpu, phi_gpu)
    result = X_gpu
    # Transform to spatial domain.
    pyculib.fft.ifft_inplace(result)
    # Take modulus.
    ModulusInPlace[blockspergrid, threadsperblock](result)

    result = result.copy_to_host()
    n_elements = np.prod(result.shape)
    # Normalise inverse fourier transformation.
    result = (result / n_elements).real.astype(np.float32)
    return downsample(result, downsampling_resolution)


def apply_wavelet_cpu(X, psi, downsampling_resolution):
    """
    Assumes X and psi are in Fourier space.
    """
    transform = np.downsample(np.abs(scipy.fftpack.ifftn(X * psi)), downsampling_resolution)
    return scipy.fftpack.fftn(transform)


def apply_wavelet(X, psi, downsampling_resolution):
    """
    Implemented partially on GPU.
    Assumes X and psi are in Fourier space.
    """
    return scipy.fftpack.fftn( extract_scattering_coefficients(X, psi, downsampling_resolution) )


@jit
def periodize(X, res):
    dimensions = X.shape
    downsampling_factor = 2**res
    width, height, depth = dimensions[0]//downsampling_factor, dimensions[1]//downsampling_factor, dimensions[2]//downsampling_factor
    result = np.zeros((width, height, depth), dtype=np.complex64)

    for k in range(-width//2, width//2):
        for l in range(-height//2, height//2):
            for m in range(-depth//2, depth//2):
                for a in range(downsampling_factor):
                    for b in range(downsampling_factor):
                        for c in range(downsampling_factor):
                            result[k, l, m] += X[k - a*(width//downsampling_factor), l - b*(height//downsampling_factor), m - c*(depth//downsampling_factor)]
    return result/(downsampling_factor**3)


# def extract_scattering_coefficients(X, phi, downsampling_resolution):
#     """
#     Phi is already in fourier space. calculate | x \conv filter | downsampled at 2**downsampling_resolution.
#     """
#     x, y, z = X.shape
#     X_gpu = cuda.to_device(X)
#     phi_gpu = cuda.to_device(phi)
#     result_full_scale = cuda.device_array_like(X)
#     result = cuda.device_array((x//2**downsampling_resolution, y//2**downsampling_resolution, z//2**downsampling_resolution), dtype=np.complex64)
#
#     # Fourier transform X
#     pyculib.fft.fft_inplace(X_gpu)
#     X_fourier = X_gpu
#     # Multiply in Fourier space
#     blockspergrid, threadsperblock = get_blocks_and_threads(x, y, z)
#     MultiplyInPlace[blockspergrid, threadsperblock](X_fourier, phi_gpu)
#     result_multiplication = X_fourier
#
#     # Downsample in Fourier space by cropping the highest frequencies (resolution is inferred by shape of `result`.)
#     blockspergrid, threadsperblock = get_blocks_and_threads(result.shape[0], result.shape[1], result.shape[2])
#     crop_freq_3d_gpu[blockspergrid, threadsperblock](result_multiplication, result)
#     # Transform to real space
#     pyculib.fft.ifft_inplace(result)
#     # Take absolute value
#     ModulusInPlace[blockspergrid, threadsperblock](result)
#     result = result.copy_to_host()
#     n_elements = np.prod(result.shape)
#     # normalise inverse fourier transformation
#     return (result / n_elements).real.astype(np.float32)
#
#
# def extract_scattering_coefficients_cpu(X, phi, downsampling_resolution):
#     """
#     Phi is already in fourier space. calculate | x \conv filter | downsampled at 2**downsampling_resolution.
#     """
#     # Fourier transform X
#     X_fourier = scipy.fftpack.fftn(X)
#     # First low-pass filter: Extract zeroth order coefficients
#     downsampled_product = crop_freq_3d( X_fourier * phi, downsampling_resolution )
#     # Transform back to real space and take modulus.
#     result = np.abs( scipy.fftpack.ifftn(downsampled_product) )
#     return result
#
#
# def abs_after_convolve(A, B, downsampling_resolution):
#     """
#     A and B are both in real space. Calculate | A \conv B |.
#     """
#     x, y, z = A.shape
#     A_gpu = cuda.to_device(A)
#     B_gpu = cuda.to_device(B)
#     result_full_scale = cuda.device_array_like(A)
#     result = cuda.device_array((x//2**downsampling_resolution, y//2**downsampling_resolution, z//2**downsampling_resolution), dtype=np.complex64)
#
#     # Fourier transform X
#     pyculib.fft.fft_inplace(A_gpu)
#     A_fourier = A_gpu
#     pyculib.fft.fft_inplace(B_gpu)
#     B_fourier = B_gpu
#
#     # Multiply in Fourier space
#     blockspergrid, threadsperblock = get_blocks_and_threads(x, y, z)
#     MultiplyInPlace[blockspergrid, threadsperblock](A_fourier, B_fourier)
#     result_multiplication = A_fourier
#
#     # Downsample in Fourier space by cropping the highest frequencies (resolution is inferred by shape of `result`.)
#     blockspergrid, threadsperblock = get_blocks_and_threads(result.shape[0], result.shape[1], result.shape[2])
#     crop_freq_3d_gpu[blockspergrid, threadsperblock](result_multiplication, result)
#     # Transform to real space
#     pyculib.fft.ifft_inplace(result)
#     # Take absolute value
#     ModulusInPlace[blockspergrid, threadsperblock](result)
#     result = result.copy_to_host()
#     n_elements = np.prod(result.shape)
#     # normalise inverse fourier transformation
#     return (result / n_elements).astype(np.complex64)
#
#
# def abs_after_convolve_cpu(A, B, downsampling_resolution):
#     """
#     A and B are both in real space. Calculate | A \conv B |.
#     """
#     A_fourier = scipy.fftpack.fftn(A)
#     B_fourier = scipy.fftpack.fftn(B)
#     downsampled_product = crop_freq_3d(A_fourier * B_fourier, downsampling_resolution)
#     return np.abs( scipy.fftpack.ifftn(downsampled_product) ).astype(np.complex64)


def crop_freq_3d(x, res):
    """
    Crop highest (1 - 2^-res) part of a fourier spectrum.
    (So for res = 1, cut highest half of the spectrum, res = 2 cut highest 3/4, etc.)
    This comes down to only taking the dim/(2**(res+1)) elements at the front and end of each dimension of the original array.
    In 2D, for res = 1 and a 4x4 input, you would get (taking only the single element at the front and back of each dimension)
    [[a00 a03], [a30, a33]]
    Corresponds to a spatial downsampling of the image by a factor (res + 1).
    Expects dimensions of array to be powers of 2.
    """
    if res == 0:
        return x

    M, N, O = x.shape[0], x.shape[1], x.shape[2]
    end_x, end_y, end_z = [int(dim * 2 ** (-res - 1)) for dim in [M, N, O]]
    indices_x, indices_y, indices_z = [ list(range(end_index)) + list(range(-end_index, 0)) for end_index in [end_x, end_y, end_z] ]
    indices = np.ix_(indices_x, indices_y, indices_z)
    return x[indices]


def get_blocks_and_threads(x, y, z):
    if x < 8 or y < 8 or z < 8:
        threadsperblock = (x, y, z)
    else:
        threadsperblock = (8, 8, 8)

    blockspergrid_x = math.ceil(x / threadsperblock[0])
    blockspergrid_y = math.ceil(y / threadsperblock[1])
    blockspergrid_z = math.ceil(z / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    return blockspergrid, threadsperblock


def downsample(X, res):
    """
    Downsampling in real space.
    """
    return np.ascontiguousarray(X[::2**res, ::2**res, ::2**res])


@cuda.jit()
def crop_freq_3d_gpu(signal_fourier, result):
    """
    Result needs to be the correct size, i.e.
    (original_width // 2**res, original_height // 2**res, original_depth // 2**res)
    """
    x, y, z = cuda.grid(3)
    width, height, depth = result.shape
    if x < (width // 2):
        i = x
    elif x < width:
        i = -width + x

    if y < (height // 2):
        j = y
    elif y < height:
        j = -height + y

    if z < (depth // 2):
        k = z
    elif z < depth:
        k = -depth + z

    result[x, y, z] = signal_fourier[i, j, k]


@cuda.jit()
def MultiplyInPlace(A, B):
    """
    Result is saved in A
    """
    x, y, z = cuda.grid(3)
    if x < A.shape[0] and y < A.shape[1] and z < A.shape[2]:
        A[x, y, z] = A[x, y, z] * B[x, y, z]


@cuda.jit()
def ModulusInPlace(A):
    x, y, z = cuda.grid(3)
    if x < A.shape[0] and y < A.shape[1] and z < A.shape[2]:
        A[x, y, z] = abs(A[x, y, z])
