import numpy as np

from my_utils import crop_freq_3d, extract_scattering_coefficients, extract_scattering_coefficients_cpu


if __name__ == '__main__':
    x = y = z = 16
    A = np.random.rand(x, y, z).astype(np.complex64)
    B = np.random.rand(x, y, z).astype(np.complex64)
    res = 2

    result_cpu = extract_scattering_coefficients_cpu(A, B, res)
    result_gpu = extract_scattering_coefficients(A, B, res)

    print(result_cpu - result_gpu)
