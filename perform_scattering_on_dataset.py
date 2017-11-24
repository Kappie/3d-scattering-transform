import numpy as np
import datetime
import os


from collections import namedtuple

from scattering_transform import apply_scattering_transform_to_dataset


def generate_output_location(js, J, n_points_fourier_sphere, sigma):
    BASE_NAME = r"F:\GEERT\results"
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "js{}-{}_J{}_npointsfourier{}_sigma{}_{}.dat".format(js[0], js[-1], J, n_points_fourier_sphere, sigma, datetime_string)
    return os.path.join(BASE_NAME, file_name)


if __name__ == '__main__':
    DATASET_PATH = r"F:\GEERT\DATASET_NUMPIFIED\dataset.npy"
    LABELS_PATH = r"F:\GEERT\DATASET_NUMPIFIED\labels.npy"
    AFFECTED = 1
    UNAFFECTED = -1

    # number of samples of each class.
    n_samples_class = 150
    dataset = np.load(DATASET_PATH, mmap_mode="r")
    labels = np.load(LABELS_PATH)
    if n_samples_class != "all":
        dataset = dataset[np.r_[:n_samples_class, -n_samples_class:0]]

    width = dataset[0].shape[0]
    js = [0, 1, 2, 3, 4, 5]
    J = js[-1]
    n_points_fourier_sphere = 20
    sigma_spatial = 0.0129
    sigma_fourier = 1/sigma_spatial
    xi_radians = 4*np.pi/5
    xi = np.array([width*xi_radians/(2*np.pi), 0., 0.])
    output_location = generate_output_location(js, J, n_points_fourier_sphere, sigma_spatial)

    apply_scattering_transform_to_dataset(dataset, js, J, n_points_fourier_sphere, output_location, sigma_fourier, xi)
