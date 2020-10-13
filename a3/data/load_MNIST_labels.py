import numpy as np
import os

def load_MNIST_labels(filename):
    """
    Args:
        filename: ubyte filename for the MNIST labels

    Returns:
        labels: [number of MNIST labels] matrix containing the labels for the MNIST images.
                The label values are in the range of [0, ..., 9].
    """

    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    labels = np.fromfile( filename, dtype = 'ubyte' )[2 * intType.itemsize:]

    return labels
