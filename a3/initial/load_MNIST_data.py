"""
Example script for loading MNIST data
"""

import sys
sys.path += ['../data']

from load_MNIST_images import load_MNIST_images
from load_MNIST_labels import load_MNIST_labels

def main():
    # Load training data
    train_data = load_MNIST_images('../data/train-images.idx3-ubyte')
    train_label = load_MNIST_labels('../data/train-labels.idx1-ubyte')
    # Load testing data
    test_data = load_MNIST_images('../data/t10k-images.idx3-ubyte')
    test_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte')

if __name__ == '__main__':
    main()
