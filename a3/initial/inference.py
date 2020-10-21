import numpy as np

def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    for i in range(num_layers):
        curr_layer = model['layers'][i]
        activations[i], _, _ = curr_layer['fwd_fn'](input, curr_layer['params'], curr_layer['hyper_params'], backprop=False)
        input = activations[i]

    output = activations[-1]
    return output, activations
