import numpy as np
from numba import cuda, vectorize
import scipy.fftpack
import time
import math


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
