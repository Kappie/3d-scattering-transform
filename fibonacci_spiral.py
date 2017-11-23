import math
import numpy as np
import transforms3d
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def latlon_to_xyz(lat, lon):
    return (math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat))


def rotation_matrix(a, b):
    """
    Gives rotation matrix in R^3 to rotate a into point b if a and b have same length, otherwise
    rotates a into sb with s scalar.
    """
    rotation_angle = math.acos( a.dot(b) / (np.linalg.norm(a)*np.linalg.norm(b)) )
    cross_product = np.cross(a, b)
    rotation_axis = cross_product / np.linalg.norm(cross_product)
    matrix = transforms3d.axangles.axangle2mat(rotation_axis, rotation_angle)
    return matrix


def rotation_matrices_fibonacci_spiral_unit_x(n):
    """
    For each of the n points spread evenly on a hemisphere using fibonacci spiral, return
    rotation matrix that rotates (1, 0, 0) onto that point.
    """
    unit_x = np.array([1, 0, 0])
    points = generate_fibonacci_spiral_hemisphere(n)
    rotation_matrices = [ rotation_matrix(unit_x, np.array(point)) for point in points ]
    return rotation_matrices


def generate_fibonacci_spiral_hemisphere(n):
    """
    Generate N points evenly spaced on hemisphere with northpole (1, 0, 0) using
    a Fibonacci spiral.
    """
    phi = (1 + 5 ** 0.5) / 2
    coordinates = []
    # We generate one more coordinate than we actually need, because the first point of the fibonacci spiral lies in
    # the (y, z)-plane and is therefore useless to center fourier support of a wavelet around.
    N = n + 1

    for i in range(N):
        lat = math.asin(2*i/(2*N + 1))
        lon = 2*math.pi*i*(1/phi)

        # swap x and z since we want a hemisphere with (1, 0, 0) as north pole.
        z, y, x = latlon_to_xyz(lat, lon)
        coordinates.append( (x, y, z) )

    return coordinates[1:]


if __name__ == '__main__':

    N = 30
    points = generate_fibonacci_spiral_hemisphere(N)
    print(points)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]


    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.scatter(xs, ys, zs)
    pyplot.show()
