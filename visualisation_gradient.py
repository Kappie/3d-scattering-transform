import numpy as np
from sklearn import datasets
from scipy.sparse.linalg import eigs
from plot_slices import plot2d
from sklearn_digits_tut import get_digits_classifier

# Implements the visualisation method described in Adel et al.
# 3D scattering transforms for disease classification in neuroimaging (2017).


def approximate_gradient(classifier, x, X, k=6, epsilon=1e-5):
    """
    classifier: outputs probability of vector x belonging to a certain class (G(x) in paper)
    x: input feature vector
    X: training data set (needed in order to calculate priciple directions)
    k: number of principle directions to use in approximating the gradient of the classifier G(x).
    epsilon: magnitude of perturbation (must be small.)
    """

    # Each row represents an observation
    covariance_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = eigs(covariance_matrix, k=k)
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)

    classifier_x = classifier.decision_function(x.reshape((1, -1)))
    gradient = np.zeros(eigenvectors.shape[0])
    for i in range(k):
        eigenvector = eigenvectors[:, i]
        gradient += ((classifier_x - classifier.decision_function((x + epsilon*eigenvector).reshape((1, -1)))) / epsilon) * eigenvector

    return gradient


def visualize_gradient(gradient):
    # reshape back to 2D array representing image.
    gradient = np.reshape(gradient, (int(np.sqrt(gradient.shape[0])), int(np.sqrt(gradient.shape[0]))))
    plot2d(gradient)


def stub_classifier(x):
    return np.random.rand()


if __name__ == '__main__':
    digits = datasets.load_digits()
    feature_vectors = digits.data
    x = feature_vectors[0]
    k = 6

    gradient = approximate_gradient(stub_classifier, x, feature_vectors, k)
    visualize_gradient(gradient)
