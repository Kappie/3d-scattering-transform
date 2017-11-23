import numpy as np
import numba
import math
import matplotlib.pyplot as plt

from itertools import product

from fibonacci_spiral import rotation_matrices_fibonacci_spiral_unit_x
from plot_utils import plot3d
from my_utils import crop_freq_3d, get_blocks_and_threads, downsample


def filter_bank(dimensions, js, J, n_points_fourier_sphere, sigma, xi):
    """
    Constructs filter bank of 3D Morlet wavelets in Fourier domain (which are Gaussians with a corrective term).
    dimensions: assumed to be powers of two.
    js: length scales for filters. Filters will be dilated in real space, hence stretched in fourier space by 2**j for j in js.
    J: length scale used for averaging over transformed signals. (coefficients will be approximately translationally
    invariant over 2**J pixels.)
    n_points_fourier_sphere: number of rotations from 3D rotation group, chosen to be rotations
    that map (1, 0, 0) onto points on the unit sphere given by the Fibonacci spiral.
    See http://dx.doi.org/10.1007/s11004-009-9257-x.
    sigma: standard deviation (in fourier space) of mother wavelet.
    xi: central frequency of mother wavelet, assumed to be multiple of (1, 0, 0).
    """
    filters = {}
    filters['psi'] = []

    rotation_matrices = rotation_matrices_fibonacci_spiral_unit_x(n_points_fourier_sphere)

    for j, r in product(js, rotation_matrices):
        psi = {'j': j, 'r': r}
        psi_signal_fourier = morlet_fourier_3d_gpu(dimensions, j, r, xi, sigma)
        # When j_1 < j_2 < ... < j_n, we need j_2, ..., j_n downsampled at j_1, j_3, ..., j_n downsampled at j_2, etc.
        # resolution 0 is just the signal itself. See below header "Fast scattering computation" in Bruna (2013).
        for resolution in range(j + 1):
            psi_signal_fourier_res = crop_freq_3d(psi_signal_fourier, resolution)
            psi[resolution] = psi_signal_fourier_res

        filters['psi'].append(psi)

    # Normalize by making sure the raw Littlewood-Paley sum is bounded from above by 1.0.
    # We downsample the LP sum by 2**J to save computation.
    filters['psi'] = normalize_psis(filters['psi'], J)

    filters['phi'] = {}
    filters['phi']['j'] = J
    phi_signal_fourier = gaussian_filter_3d(dimensions, J, sigma)
    phi_signal_fourier = normalize_fourier(phi_signal_fourier)
    # We need the phi signal downsampled at all length scales j.
    for resolution in js:
        phi_signal_fourier_res = crop_freq_3d(phi_signal_fourier, resolution)
        filters['phi'][resolution] = phi_signal_fourier_res

    return filters


def normalize_psis(psis, resolution):
    """
    Normalize all wavelets by requiring that the raw Littlewood-Paley sum is bounded from above by 1.0.
    (Actually only works if maximum of LP sum is larger than 1.0.)
    psis: list of psi wavelets (dicts with meta-information and the actual filters and downsampled versions.)
    """
    # TODO: I only normalize the original (undownsampled) filters. I don't know how to actually implement
    # downsampling in the correct way!!
    psis_original = [psi[0] for psi in psis]
    raw_lp_sum = raw_littlewood_paley_sum(psis_original, resolution)
    largest_element = np.max(raw_lp_sum)
    print("largest element in unnormalised raw lp sum: ", largest_element)
    psis_original_normalized = [psi / largest_element for psi in psis_original]

    result = []
    for i, psi in enumerate(psis):
        psi[0] = psis_original_normalized[i]
        result.append(psi)

    return result


@numba.jit
def raw_littlewood_paley_sum_jit(abs_squared_psis):
    width, height, depth = abs_squared_psis[0].shape
    result = np.empty((width, height, depth))

    for k in range(-width//2, width//2):
        for l in range(-height//2, height//2):
            for m in range(-depth//2, depth//2):
                raw_sum = 0
                for abs_squared_psi in abs_squared_psis:
                    raw_sum = raw_sum + abs_squared_psi[k, l, m] + abs_squared_psi[-k, -l, -m]
                result[k, l, m] = 0.5*raw_sum

    return result


def raw_littlewood_paley_sum(psis, resolution):
    """
    We only calculate the Littlewood-Paley sum for frequencies equally spaced 2**J from each other.
    Assumes filters are already in Fourier space.
    """
    psis_downsampled = [downsample(psi, resolution) for psi in psis]
    # Fourier transform of Morlet is real, so no need for complex conjugate.
    abs_squared_psis = [psi*psi for psi in psis_downsampled]
    return raw_littlewood_paley_sum_jit(abs_squared_psis)


def littlewood_paley_sum(phi, psis, resolution):
    """
    Same comments as for `raw_littlewood_paley_sum` apply.
    """
    phi_downsampled = downsample(phi, resolution)
    abs_squared_phi = phi_downsampled * phi_downsampled
    return abs_squared_phi + raw_littlewood_paley_sum(psis, resolution)


@numba.jit
def morlet_fourier_3d(dimensions, j, r, xi, sigma, a=2.0):
    """
    Assumes dimensions are powers of two.
    r: 3x3 rotation matrix.
    xi: [xi, 0, 0] by convention.
    """
    width, height, depth = dimensions
    result = np.empty((width, height, depth))

    scale_factor = a**j
    normalisation = a**(-3*j)
    kappa_sigma = gauss_3d(-xi[0], -xi[1], -xi[2], sigma) / gauss_3d(0, 0, 0, sigma)
    for k in range(-width//2, width//2):
        for l in range(-height//2, height//2):
            for m in range(-depth//2, depth//2):
                # Rotate and scale.
                k_prime = (r[0, 0]*k + r[0, 1]*l + r[0, 2]*m) * scale_factor
                l_prime = (r[1, 0]*k + r[1, 1]*l + r[1, 2]*m) * scale_factor
                m_prime = (r[2, 0]*k + r[2, 1]*l + r[2, 2]*m) * scale_factor
                result[k, l, m] = normalisation * (
                    gauss_3d(k_prime-xi[0], l_prime-xi[1], m_prime-xi[2], sigma) -
                    kappa_sigma*gauss_3d(k_prime, l_prime, m_prime, sigma) )
    return result


def morlet_fourier_3d_gpu(dimensions, j, r, xi, sigma, a=2.0):
    scale_factor = a**j
    normalisation = a**(-3*j)
    kappa_sigma = gauss_3d(-xi[0], -xi[1], -xi[2], sigma) / gauss_3d(0, 0, 0, sigma)
    result = np.empty(dimensions, dtype=np.float64)
    blockspergrid, threadsperblock = get_blocks_and_threads(dimensions[0], dimensions[1], dimensions[2])
    _morlet_fourier_3d_gpu[blockspergrid, threadsperblock](dimensions, r, xi, a, sigma, scale_factor, normalisation, kappa_sigma, result)
    return result


@numba.cuda.jit()
def _morlet_fourier_3d_gpu(dimensions, r, xi, a, sigma, scale_factor, normalisation, kappa_sigma, result):
    width, height, depth = dimensions
    k, l, m = numba.cuda.grid(3)
    if k < width and l < height and m < depth:
        # Make sure output array orders frequencies in range(-width//2, width//2) as
        # [0 1 2 3 -4 -3 -2 -1] for width = 8, i.e. default dft convention. Same for other dimensions.
        if k >= width//2:
            k_prime = - width + k
        else:
            k_prime = k
        if l >= height//2:
            l_prime = - height + l
        else:
            l_prime = l
        if m >= depth//2:
            m_prime = - depth + m
        else:
            m_prime = m

        # Rotate and scale.
        k_prime = (r[0, 0]*k_prime + r[0, 1]*l_prime + r[0, 2]*m_prime) * scale_factor
        l_prime = (r[1, 0]*k_prime + r[1, 1]*l_prime + r[1, 2]*m_prime) * scale_factor
        m_prime = (r[2, 0]*k_prime + r[2, 1]*l_prime + r[2, 2]*m_prime) * scale_factor

        result[k, l, m] = normalisation * (
            math.exp( -((k_prime-xi[0])**2 + (l_prime-xi[1])**2 + (m_prime-xi[2])**2)/(2*sigma*sigma) ) -
            kappa_sigma * math.exp( -(k_prime**2 + l_prime**2 + m_prime**2)/(2*sigma*sigma) )
        )


@numba.jit
def gaussian_filter_3d(dimensions, j, sigma, a=2.0):
    width, height, depth = dimensions
    result = np.empty((width, height, depth))

    scale_factor = a**j
    for k in range(-width//2, width//2):
        for l in range(-height//2, height//2):
            for m in range(-depth//2, depth//2):
                result[k, l, m] = gauss_3d(k, l, m, sigma/scale_factor)
    return result


@numba.jit
def morlet_fourier_1d(N, j, xi, sigma, a=2.0):
    """
    Assumes signal length N = 2^n.
    """
    result = np.empty(N)
    kappa_sigma = gauss_1d(-xi, sigma) / gauss_1d(0, sigma)
    normalisation = a**(-j)

    for omega in range(-N//2, N//2):
        result[omega] = normalisation * ( gauss_1d(a**j * omega - xi, sigma) - kappa_sigma*gauss_1d(a**j * omega, sigma) )
    return result


@numba.jit
def gauss_3d(x, y, z, sigma):
    return math.exp(-(x*x + y*y + z*z) / (2*sigma*sigma))


@numba.jit
def gauss_1d(x, sigma):
    return math.exp(-x*x / (2*sigma*sigma))


def normalize_fourier(signal_fourier):
    """
    Normalising in Fourier domain means making sure the zeroth frequency component
    is equal to 1.
    """
    return signal_fourier / signal_fourier[0, 0, 0]


def plot(x):
    plt.plot(x)
    plt.show()


if __name__ == '__main__':
    # 1D Case:
    # N = 64
    # xi_radians = 4*np.pi/5
    # xi = N * xi_radians/(2*np.pi)
    # sigma_spatial = 0.6
    # sigma_fourier = 1 / sigma_spatial
    #
    # for j in range(4):
    #     result = morlet_fourier_1d(N, j, xi, sigma_fourier)
    #     print(result[0])
    #     plot(result)

    dimensions = np.array([32, 32, 32])
    xi_radians = 4*np.pi/5
    xi = np.ceil(dimensions[0] * xi_radians/(2*np.pi))
    xi = np.array([xi, 0, 0])
    sigma_spatial = 0.2
    sigma_fourier = 1 / sigma_spatial
    n_points_fourier_sphere = 4
    rotation_matrices = rotation_matrices_fibonacci_spiral_unit_x(n_points_fourier_sphere)

    for j in range(3):
        for r in rotation_matrices:
            result = morlet_fourier_3d(dimensions, j, r, xi, sigma_fourier)
            print("Zero frequency: ", result[0, 0, 0])
            maximum_pos = np.unravel_index(np.argmax(result), result.shape)
            print("Position of maximum: ", maximum_pos)
            plot3d(result)
