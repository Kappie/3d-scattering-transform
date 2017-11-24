import numpy as np
import sklearn as sk
import re

from numpy.lib.format import open_memmap
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer
import sklearn.metrics
from scattering_transform import number_of_transforms


def classify(dataset, labels):
    # classifier = svm.SVC(kernel='linear', C=1)
    # selector = PCA(n_components=9)
    # dataset = selector.fit_transform(dataset)
    # scores = cross_val_score(classifier, dataset, labels, cv=5)
    print(labels)
    print(dataset.shape)
    test_size = 0.2
    n_jobs = 8
    score_function = sklearn.metrics.accuracy_score

    normalize(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=test_size, shuffle=True)
    print(y_train)
    print(y_train.shape)

    # Do a grid search for best SVM.
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ]
    classifier = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring=make_scorer(score_function), n_jobs=n_jobs)
    classifier.fit(X_train, y_train)
    print("Best parameters found on training set:")
    print(classifier.best_params_)
    print("Grid scores on training set:")
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print("Detailed classification report:")
    print()
    print("The model is trained on the full training set.")
    print("The scores are computed on the full test set.")
    print()
    y_true, y_pred = y_test, classifier.predict(X_test)
    print("accuracy on test set:", score_function(y_true, y_pred))
    print(sklearn.metrics.confusion_matrix(y_true, y_pred))
    # print(classification_report(y_true, y_pred))
    print()

    return classifier



def normalize(dataset):
    # Same as in Bruna's thesis.
    minmax_scale(dataset, feature_range=(-1, 1), copy=False)


# def load_coefficients_old():
#     """
#     Only works for results\js0-3_J3_L3_sigma5_2017-11-09_18-58-23.dat !!!
#     """
#     path = "results\js0-3_J3_L3_sigma5_2017-11-09_18-58-23.dat"
#     shape = (100, 4483, 16, 32, 16)
#     dtype = np.float32
#     coefficients = np.memmap(path, shape=shape, dtype=dtype, mode="r")
#     return coefficients


def visualize(classifier):
    return 1


if __name__ == '__main__':
    AFFECTED = 1
    UNAFFECTED = -1

    # This is how I now load coefficients:
    scattering_coefficients_path = r"F:\GEERT\results\js0-3_J6_L3_sigma5_2017-11-10_17-05-55.dat"
    scattering_coefficients = open_memmap(scattering_coefficients_path)
    max_n_samples_class = 200

    n_samples, n_transforms, width, height, depth = scattering_coefficients.shape
    # By convention, we always choose the same number of samples of each class, where all the affected
    # hemispheres come first.
    # n_samples_class = n_samples // 2
    n_samples_class = min([n_samples // 2, max_n_samples_class])

    labels = np.concatenate( [np.repeat(AFFECTED, n_samples_class), np.repeat(UNAFFECTED, n_samples_class)] )
    # flatten all coefficients for each sample. Coefficients are highly correlated. What to do about that?
    # e.g. discrete cosine transform or principal component analysis.
    scattering_coefficients = scattering_coefficients.reshape((n_samples, -1))
    scattering_coefficients = scattering_coefficients[np.r_[:n_samples_class, -n_samples_class:0]]
    print(scattering_coefficients.shape)
    print(labels.shape)
    classify(scattering_coefficients, labels)
