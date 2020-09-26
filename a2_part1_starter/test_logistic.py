"""
 Princeton University, COS 429, Fall 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from logistic_fit import logistic_fit
from logistic_predict import logistic_predict


def test_logistic():
    """ Test logistic regression on some simple data in 'training_data.txt'
        and 'test_data.txt'
    """
    # Load the training data
    training = np.loadtxt('training_data.txt')

    # Do the training
    num_pts = training.shape[0]
    X = np.concatenate([np.ones([num_pts, 1]), training[:, :2]], axis=1)
    z = training[:, 2]
    l = 0.001
    params = logistic_fit(X, z, l)

    # Apply the learned model to the training data and print out statistics
    predicted = logistic_predict(X, params)
    training_accuracy = np.sum(predicted == z) / num_pts
    print('training_accuracy =', training_accuracy)

    # Plot "ground truth" and predictions
    plot_classes(training, z, 'Training ground truth', 1)
    plot_classes(training, predicted, 'Training predicted', 2)

    # Apply the learned model to the test data
    testing = np.loadtxt('test_data.txt')
    num_pts = testing.shape[0]
    X = np.concatenate([np.ones([num_pts, 1]), testing[:, :2]], axis=1)
    z = testing[:, 2]
    predicted = logistic_predict(X, params)
    testing_accuracy = np.sum(predicted == z) / num_pts
    print('testing_accuracy =', testing_accuracy)

    # Plot "ground truth" and predictions
    plot_classes(testing, z, 'Testing ground truth', 3)
    plot_classes(testing, predicted, 'Testing predicted', 4)


def plot_classes(data, z, name, num):
    """Create a scatterplot of the given data.  It is assumed that the input
       data is 2-dimensional.

    Args:
        data: datapoints (one per row, only first two columns used)
        z: labels (0/1)
        name: name of the figure
        num: number of the figure
    """
    plt.figure(num)
    plt.title(name)
    positive = data[z > 0]
    negative = data[z == 0]
    plt.scatter(positive[:, 0], positive[:, 1], c='red', alpha=0.5)
    plt.scatter(negative[:, 0], negative[:, 1], c='blue', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    test_logistic()
