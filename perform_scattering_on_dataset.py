import numpy as np
import datetime
import os


from collections import namedtuple

from scattering_transform import apply_scattering_transform_to_dataset


def generate_output_location(js, J, L, sigma):
    BASE_NAME = r"F:\GEERT\results"
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "js{}-{}_J{}_L{}_sigma{}_{}.dat".format(js[0], js[-1], J, L, sigma, datetime_string)
    return os.path.join(BASE_NAME, file_name)


if __name__ == '__main__':
    DATASET_PATH = r"F:\GEERT\DATASET_NUMPIFIED\dataset.npy"
    LABELS_PATH = r"F:\GEERT\DATASET_NUMPIFIED\labels.npy"
    AFFECTED = 1
    UNAFFECTED = -1

    # number of samples of each class.
    n_samples_class = "all"
    dataset = np.load(DATASET_PATH, mmap_mode="r")
    labels = np.load(LABELS_PATH)
    if n_samples_class != "all":
        dataset = dataset[np.r_[:n_samples_class, -n_samples_class:0]]

    js = [0, 1, 2, 3]
    J = 6
    L = 3
    sigma = 2
    # xi = np.array([np.pi*3/4, np.pi/6, np.pi/6])
    xi = np.array([np.pi*3/4, 0, 0])
    output_location = generate_output_location(js, J, L, sigma)

    apply_scattering_transform_to_dataset(dataset, js, J, L, output_location, sigma=sigma, xi=xi)
