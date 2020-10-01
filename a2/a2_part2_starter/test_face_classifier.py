"""
 Princeton University, COS 429, Fall 2020
"""
import numpy as np
from get_training_data import get_training_data
from get_testing_data import get_testing_data
from logistic_prob import logistic_prob
from logistic_fit import logistic_fit
import matplotlib.pyplot as plt


def test_face_classifier(ntrain, ntest, orientations, wrap180):
    """Train and test a face classifier.

    Args:
        ntrain: number of face and nonface training examples (ntrain of each)
        ntest: number of face and nonface testing examples (ntest of each)
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360
    """
    # Get some training data
    descriptors, classes = get_training_data(ntrain, orientations, wrap180)

    # Train a classifier
    params = logistic_fit(descriptors, classes, 0.001)

    # Evaluate the classifier on the training data
    predicted = logistic_prob(descriptors, params)
    plot_errors(predicted, classes, 'Performance on training set for varying threshold', 1)

    # Get some test data
    tdescriptors, tclasses = get_testing_data(ntest, orientations, wrap180)

    # Evaluate the classifier on the test datals
    tpredicted = logistic_prob(tdescriptors, params)

    plot_errors(tpredicted, tclasses, 'Performance on test set for varying threshold', 2)
    #np.savez('face_classifier.npz', params=params, orientations=orientations, wrap180=wrap180)


def plot_errors(predicted, classes, name, num):
    """Plot a log/log graph of miss rate (false negatives) vs false positives
       for a variety of thresholds on probability.

    Args:
        predicted: probabilities that the class is 1
        classes: ground-truth class labels (0/1)
        name: name of the figure
        num: number of the figure
    """
    nthresh = 99
    npts = predicted.shape[0]

    falsepos = np.zeros([nthresh])
    falseneg = np.zeros([nthresh])

    stepsize = 1. / (nthresh + 1)
    for i in range(nthresh):
        thresh = (i + 1) * stepsize
        falsepos[i] = np.sum(np.logical_and(predicted >= thresh, classes == 0)) / npts
        falseneg[i] = np.sum(np.logical_and(predicted < thresh, classes == 1)) / npts

    limit = 1e-4
    plt.figure(num)
    plt.title(name)
    plt.loglog(np.maximum(falsepos, limit), np.maximum(falseneg, limit))
    plt.axis([limit, 1, limit, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('False negative rate')

    plt.show()
