import sys
sys.path += ['layers']
import numpy as np
from init_layers import init_layers
from init_model import init_model
from inference import inference
from loss_euclidean import loss_euclidean

sys.path += ['../data']

from load_MNIST_images import load_MNIST_images
from load_MNIST_labels import load_MNIST_labels
from train import train
from matplotlib import pyplot as plt


def main():
    # Load training data
    train_data = load_MNIST_images('../data/train-images.idx3-ubyte')
    train_label = load_MNIST_labels('../data/train-labels.idx1-ubyte')
    # Load testing data
    test_data = load_MNIST_images('../data/t10k-images.idx3-ubyte')
    test_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte')

    total_input_data = np.concatenate((train_data, test_data), axis=3)
    total_label_data = np.concatenate((train_label, test_label))

    l = [init_layers('conv', {'filter_size': 3,
                              'filter_depth': 1,
                              'num_filters': 35}),
         init_layers('relu', {}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('flatten', {}),
         init_layers('linear', {'num_in': 5915,
                                'num_out': 100}),
         init_layers('relu', {}),
         init_layers('linear', {'num_in': 100,
                                'num_out': 10}),
         init_layers('softmax', {})]

    model = init_model(l, [28, 28, 1], 10, True)


    #model = np.load('depth_model_35_2.npz', allow_pickle=True)
    #model = dict(model)
    params = {}


    model, loss = train(model, total_input_data, total_label_data, params, 450)

    plt.scatter(list(range(loss.shape[0])), loss)
    plt.title('Cross-Entropy Training Loss over Iteration Counts')
    plt.xlabel('Num Iterations')
    plt.ylabel('Training Loss')
    plt.show()
    plt.savefig('training_loss.png')

    test_output, _ = inference(model, test_data)
    max_test_values = np.argmax(test_output, axis=0)
    test_accuracy = np.sum(np.equal(max_test_values, test_label)) / len(max_test_values)
    print(f"Final Accuracy: {test_accuracy}")


if __name__ == '__main__':
    main()
