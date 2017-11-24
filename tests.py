import numpy as np
import scipy.fftpack
import time

from scipy.misc import imread, imresize
from numba import cuda

from plot_utils import plot3d, plot2d
from my_utils import (downsample, crop_freq_3d, crop_freq_3d_gpu, get_blocks_and_threads,
    extract_scattering_coefficients, extract_scattering_coefficients_cpu, periodize)


def test_filters():
    x = y = z = 32
    j = 1
    alpha = 0
    beta = 0
    gamma = np.pi / 4

    gabor = get_gabor_filter_gpu(x, y, z, j, alpha, beta, gamma)
    gaussian = get_gaussian_filter_gpu(x, y, z, j)

    plot3d(np.real(gaussian))


def test_filters_fourier_support():
    x = z = 128
    y = 256
    j = 0
    alpha = 0
    beta = 0
    gamma = 0
    # xi = np.array([3*np.pi/4, 0, 0])
    xi = np.array([0, 0, 0])
    sigma = 5

    J = 3
    for j in range(J + 1):
        gabor = get_gabor_filter_gpu(x, y, z, j, alpha, beta, gamma, xi=xi, sigma=sigma)
        plot3d(np.real(gabor))

    # gabor_fourier = scipy.fftpack.fftn(gabor)
    # gaussian = get_gaussian_filter_gpu(x, y, z, j)
    # gaussian_fourier = scipy.fftpack.fftn(gaussian)
    # plot3d(np.abs(gabor_fourier))
    # plot3d(np.real(gabor))
    # plot3d(np.abs(gaussian_fourier))
    # plot3d(np.real(gaussian))


def get_test_3d_image():
    # Make a copy of 4 resized mona lisas to serve as test 3d image.
    image = imread("mona_lisa.gif", mode="F")
    image = imresize(image, (32, 32))
    image = np.stack([image, image, image, image, image, image, image, image], axis=2)
    return image


def test_downsampling_real_space():
    image = get_test_3d_image()
    res = 1
    image = downsample(image, res)
    plot3d(image)


def test_downsampling_fourier():
    image = get_test_3d_image()
    res = 1
    image_fourier = scipy.fftpack.fftn(image)
    image_fourier_downsampled = crop_freq_3d(image_fourier, res)
    image_downsampled = np.absolute( scipy.fftpack.ifftn(image_fourier_downsampled) )
    plot3d(image_downsampled)


def test_downsampling_fourier_gpu():
    image = get_test_3d_image()
    x, y, z = image.shape
    res = 1
    image_fourier = scipy.fftpack.fftn(image)

    image_fourier_gpu = cuda.to_device(image_fourier)
    result = cuda.device_array((x//2**res, y//2**res, z//2**res), dtype=np.complex64)
    blockspergrid, threadsperblock = get_blocks_and_threads(result.shape[0], result.shape[1], result.shape[2])
    image_fourier_downsampled = crop_freq_3d_gpu[blockspergrid, threadsperblock](image_fourier_gpu, result)
    result = result.copy_to_host()

    image_downsampled = np.absolute( scipy.fftpack.ifftn(result) )
    plot3d(image_downsampled)


def test_abs_after_convolve():
    x = y = z = 32
    j = 0
    delta = np.zeros((x, y, z)).astype(np.complex64)
    delta[0, 0, 0] = 1
    A = np.random.rand(x, y, z).astype(np.complex64)
    abs_convolution = abs_after_convolve(delta, A, j)
    # Convolution with a delta should be equal to the input signal.
    print("average absolute value difference between abs(A) and abs_after_convolve(A, delta): ", np.average(np.abs(np.abs(A) - abs_convolution)))


def benchmark_cpu_vs_gpu_different_sizes():
    x = z = 128
    y = 256
    max_res = 3
    downsampling_resolution = 1

    # compilation run
    X = np.random.rand(4, 4, 4).astype(np.complex64)
    abs_after_convolve(X, X, downsampling_resolution)

    for res in range(max_res + 1):
        A = np.random.rand(x//2**res, y//2**res, z//2**res).astype(np.complex64)
        B = np.random.rand(x//2**res, y//2**res, z//2**res).astype(np.complex64)

        start = time.time()
        abs_after_convolve(A, B, downsampling_resolution)
        end = time.time()
        print("gpu at res {} took {}.".format(res, end-start))

        start = time.time()
        abs_after_convolve_cpu(A, B, downsampling_resolution)
        end = time.time()
        print("cpu at res {} took {}.".format(res, end-start))


def test_filter_bank():
    width = depth = 32
    height = 64
    js = [0, 1]
    J = 6
    L = 3
    sigma = 5
    # xi = np.array([3*np.pi/4, 3*np.pi/4, 3*np.pi/4])
    xi = np.array([3*np.pi/4, np.pi/4, np.pi/4])

    filters = filter_bank(width, height, depth, js, J, L, sigma, xi=xi)
    # for psi_filter in filters['psi']:
    #     print(psi_filter["j"], psi_filter["alpha"])
    filter = filters['psi'][13]
    print(filter["j"], filter["alpha"], filter["beta"], filter["gamma"])
    plot3d(np.real(filter[0]))
    plot3d(np.imag(filter[0]))


def test_apply_gaussian_filter():
    image = get_test_3d_image()
    width, height, depth = image.shape
    j = 0
    J = 0
    sigma = 0.5
    gaussian_filter = get_gaussian_filter_gpu(width, height, depth, J, sigma=sigma)
    plot3d(gaussian_filter)
    result = abs_after_convolve(image.astype(np.complex64), gaussian_filter.astype(np.complex64), j)
    plot3d(result.astype(np.float32))


def test_apply_gabor_filter():
    image = get_test_3d_image().astype(np.complex64)
    width, height, depth = image.shape
    j = 1
    alpha = np.pi/8
    beta = np.pi/6
    gamma = np.pi/2
    sigma = 1
    xi = np.array([3*np.pi/4, 0.1, 0.1])
    gabor_filter = get_gabor_filter_gpu(width, height, depth, j, alpha, beta, gamma, sigma=sigma, xi=xi)
    plot3d(np.real(gabor_filter))
    plot3d(np.imag(gabor_filter))
    result = abs_after_convolve(image, gabor_filter, j).astype(np.float32)
    plot3d(result)


def test_rotated_gabor_filter():
    x = z = 128
    y = 256
    j = 3
    alpha = 2*np.pi/3
    beta  = np.pi/3
    gamma = 2*np.pi/3
    sigma = 2
    xi = np.array([3*np.pi/4, 0, 0])
    gabor_filter = get_gabor_filter_gpu(x, y, z, j, alpha, beta, gamma, sigma=sigma, xi=xi)
    plot3d(np.real(gabor_filter))
    plot3d(np.imag(gabor_filter))


def test_periodize():
    image = get_test_3d_image()
    plot3d(image)
    image_fourier = scipy.fftpack.fftn(image)
    res = 1
    result = periodize(image_fourier, res)
    result = np.abs(scipy.fftpack.ifftn(result))
    plot3d(result)
    plot3d(downsample(image, res))


if __name__ == '__main__':
    # test_filters()
    # test_downsampling_real_space()
    # test_downsampling_fourier()
    # test_downsampling_fourier_gpu()
    # test_abs_after_convolve()
    # test_filters_fourier_support()
    # benchmark_cpu_vs_gpu_different_sizes()
    # test_filter_bank()
    # test_apply_gaussian_filter()
    # test_apply_gabor_filter()
    test_periodize()
