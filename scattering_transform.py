import numpy as np
import tensorflow as tf
import scipy.fftpack as fft
import time
import os

from plot_slices import plot3d
from scipy.special import binom
from numpy.lib.format import open_memmap

from wavelets import filter_bank, SIGMA_DEFAULT, XI_DEFAULT
from my_utils import abs_after_convolve, extract_scattering_coefficients


def apply_scattering_transform_to_dataset(images, js, J, L, output_location, sigma=SIGMA_DEFAULT, xi=XI_DEFAULT):
    """
    images: images in n_samples x width x height x depth format.
    js: length scales for filters. Filters will be dilated by 2**j for j in js.
    L: number of angles for filters, spaced evenly in (0, pi).
    J: length scale used for averaging over scattered signals. (coefficients will be approximately translationally
    invariant over 2**J pixels.)
    Return scattering coefficients in format: (n_samples, n_transforms, width//2**J, height//2**J, depth//2**J)
    """
    global scan_nr

    n_samples, width, height, depth = images.shape
    coefficients = open_memmap(
        output_location, dtype=np.float32, mode="w+",
        shape=(n_samples, number_of_transforms(js, L), width//2**J, height//2**J, depth//2**J))

    print("Making filter bank...")
    start = time.time()
    filters = filter_bank(width, height, depth, js, J, L, sigma, xi=xi)
    end = time.time()
    print("Done in {}.".format(str(end - start)))

    for n in range(n_samples):
        scan_nr = n
        start = time.time()
        image = images[n, :, :, :]
        coefficients_sample = scattering_transform(images[n, :, :, :], J, filters)
        coefficients[n, :, :, :, :] = coefficients_sample
        end = time.time()
        print("Scattering {} of {} done in {} seconds.".format(n + 1, n_samples, str(end - start)))

    print(coefficients.shape)
    return coefficients


def scattering_transform(X, J, filters):
    """
    X: input image in width x height x depth format.
    Computes only the first two layers of the scattering transform.
    Return scattering coefficients in format: (n_transforms, width//2**J, height//2**J, depth//2**J)
    """

    # save_transform_to_disk(X, {})
    X = X.astype(np.complex64)
    psis = filters['psi']
    phis = filters['phi']
    scattering_coefficients = []
    transforms = []

    # First low-pass filter: Extract zeroth order coefficients
    zeroth_order_coefficients = extract_scattering_coefficients(X, phis[0], J)
    # save_transform_to_disk(zeroth_order_coefficients, {'J': J})
    scattering_coefficients.append(zeroth_order_coefficients)

    for n1 in range(len(psis)):
        j1 = psis[n1]['j']
        alpha1, beta1, gamma1 = psis[n1]['alpha'], psis[n1]['beta'], psis[n1]['gamma']

        # Calculate wavelet transform and apply modulus. Signal can be downsampled at 2**j1 without losing much energy.
        # See Bruna (2013).
        transform1 = abs_after_convolve(X, psis[n1][0], j1)
        # save_transform_to_disk(transform1.astype(np.float32), {'j1': j1, 'alpha': alpha1, 'beta': beta1, 'gamma': gamma1})

        # Second low-pass filter: Extract first order coefficients.
        # The transform is already downsampled by 2**j1, so we take the version of phi that is downsampled by the same
        # factor. The scattering coefficients itself can be sampled at 2**J, so a downsampling of 2**(J - j1) remains.
        first_order_coefficients = extract_scattering_coefficients(transform1, phis[j1], J - j1)
        # save_transform_to_disk(first_order_coefficients, {'j1': j1, 'alpha': alpha1, 'beta': beta1, 'gamma': gamma1, 'J': J})
        scattering_coefficients.append(first_order_coefficients)

        for n2 in range(len(psis)):
            j2 = psis[n2]['j']
            alpha2, beta2, gamma2 = psis[n2]['alpha'], psis[n2]['beta'], psis[n2]['gamma']
            if j1 < j2:
                # transform1 is already downsampled at 2**j1, so we take the wavelet that is downsampled at the same
                # factor.
                # We can downsample transform2 at 2**j2, so here it remains to downsample with the factor 2**(j2-j1).
                transform2 = abs_after_convolve(transform1, psis[n2][j1], j2 - j1)
                # save_transform_to_disk(transform2.astype(np.float32), {'j1': j1, 'alpha': alpha1, 'beta': beta1, 'gamma': gamma1, 'j2': j2, 'alpha2': alpha2, 'beta2': beta2, 'gamma2': gamma2})

                # Third low-pass filter. Extract second-order coefficients.
                # The transform is already downsampled by 2**j2, so we take the version of phi that is downsampled by
                # the same factor. The scattering coefficients itself can be sampled at 2**J, so a downsampling of
                # 2**(J - j2) remains.
                second_order_coefficients = extract_scattering_coefficients(transform2, phis[j2], J - j2)
                # save_transform_to_disk(second_order_coefficients, {'j1': j1, 'alpha': alpha1, 'beta': beta1, 'gamma': gamma1, 'j2': j2, 'alpha2': alpha2, 'beta2': beta2, 'gamma2': gamma2, "J": J})
                scattering_coefficients.append(second_order_coefficients)

    scattering_coefficients = np.array(scattering_coefficients)
    return scattering_coefficients


def number_of_transforms(js, L, m_max=2):
    # original image is the first transform
    total = 1
    n_js = len(js)

    for m in range(1, m_max + 1):
        total += L**(3*m) * int(binom(n_js, m))

    return total


SAVE_TRANSFORMS_PATH = r"F:\GEERT\transforms"
scan_nr = 0


def save_transform_to_disk(transform, dict_of_parameters):
    file_name = "scan_nr{}".format(scan_nr)
    for parameter, value in dict_of_parameters.items():
        if isinstance(value, int):
            value_string = "{}".format(value)
        if isinstance(value, float):
            value_string = "{:03.2f}".format(value)
        file_name += "{}{}".format(parameter, value_string)

    save_path = os.path.join(SAVE_TRANSFORMS_PATH, file_name)
    np.savez_compressed(save_path, transform)
    print("Saved {}.".format(save_path))


if __name__ == '__main__':
    js = [0, 1, 2]
    J = 3
    L = 3
    x = y = 128
    z = 256
    images = np.random.rand(1, x, y, z)
    output_location = "scattering_coefficients.dat"
    S = apply_scattering_transform_to_dataset(images, js, J, L, output_location)
    print(S.shape)
