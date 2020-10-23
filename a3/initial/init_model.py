import numpy as np
import sys
sys.path += ['pyc_code']
from inference_ import inference as inference_

def init_model(layers, input_size, output_size, display):
    """
    Initialize a network model given an array of layers.
    Expected input and output size must be provided to verify that the network
    is properly defined.
    """

    model = {'layers': layers,
             'input_size': input_size,
             'output_size': output_size}

    # Sanity check that layer input/output sizes are correct
    # Batch sizes of 1 and greater than 1 are used to ensure that both cases are handled properly by the code

    num_layers = len(model['layers'])
    input1 = np.random.rand(*(input_size + [1]))
    input5 = np.random.rand(*(input_size + [5]))

    output1, act1 = inference_(model, input1)
    output5, act5 = inference_(model, input5)

    network_output_size = output5.shape
    network_output_size = network_output_size[:-1]

    # While designing your model architecture it can be helpful to know the
    # intermediate sizes of activation matrices passed between layers. 'display'
    # is an option you set when you call 'init_model'.
    if display:
        print('Input size:')
        print(input_size)

        for i in range(num_layers - 1):
            print('Layer %d output size: ' % i)
            
            act_size_1 = act1[i].shape
            act_size_5 = act5[i].shape

            if len(act_size_1) == len(act_size_5) and act_size_1[:-1]==act_size_5[:-1]:
                print(act_size_1[:-1])
            else:
                print('Error in layer %d, size mismatch between different batch sizes' % i)
                print('With batch size 5:')
                print(act_size_5[:-1])
                print('With batch size 1:')
                print(act_size_1[:-1])

        print('Final output size:')
        print(network_output_size)
        print('Provided output size (should match above):')
        print(output_size)
        print('(Batch dimension not included)')

    # If you defined all of your layers correctly you should know the final
    # size of the output matrix, this is just a sanity check.

    if isinstance(output_size, (int, float)):
        assert len(network_output_size) == 1, 'Network output does not match up with provided output size'
        assert network_output_size[0] == output_size, 'Network output does not match up with provided output size'
    else:
        assert list(network_output_size) == list(output_size), 'Network output does not match up with provided output size'

    return model