import numpy as np
import scipy.fftpack
import numba
import time
import math

from filter_bank import filter_bank, littlewood_paley_sum, raw_littlewood_paley_sum
from my_utils import downsample


def littlewood_paley_condition(lp_sum):
    """
    Check if L-P sum satisfies L-P condition and if so, for what epsilon.
    """
    all_elems_lq_one = np.all(lp_sum <= 1)
    epsilons = 1 - lp_sum
    epsilon = np.max(epsilons)
    average_epsilon = np.average(epsilons)

    return all_elems_lq_one, epsilon, average_epsilon

if __name__ == '__main__':
    x = z = 128
    y = 256
    dimensions = np.array([x, y, z])
    js = [0, 1, 2, 3, 4]
    J = js[-1]
    n_points_fourier_sphere = 20
    sigma_spatial = 0.0129
    sigma_fourier = 1/sigma_spatial
    xi_radians = 4*np.pi/5
    xi = np.array([x*xi_radians/(2*np.pi), 0., 0.])

    print("making filter bank...")
    start = time.time()
    filters = filter_bank(dimensions, js, J, n_points_fourier_sphere, sigma_fourier, xi)
    end = time.time()
    print("done in ", end-start)

    # Get original, undownsampled filters (resolution 0) in fourier space.
    phi = filters['phi'][0]
    psis = [psi[0] for psi in filters['psi']]
    lp_sum = littlewood_paley_sum(phi, psis, J)
    bandwidth = 2*sigma_fourier*math.sqrt(math.log(2))
    all_elems_lq_one, epsilon, average_epsilon = littlewood_paley_condition(lp_sum)
    print("LP condition satisfied: ", all_elems_lq_one)
    print("with epsilon:", epsilon)
    print("average epsilon:", average_epsilon)
    print("bandwidth of mother wavelet:", bandwidth)
    print("bandwidth radians:", (bandwidth/x)*2*np.pi)
    print("complete lp sum:")
    print(lp_sum)
