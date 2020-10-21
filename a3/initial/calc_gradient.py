import numpy as np

def calc_gradient(model, input, layer_acts, dv_output):
    '''
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns: 
        grads:  A list of gradients of each layer in model["layers"]
    '''
    num_layers = len(model["layers"])
    grads = [None,] * num_layers

    for i in range(num_layers-1, -1, -1):
        curr_layer = model["layers"][i]
        curr_input = input
        if i > 0:
            curr_input = layer_acts[i-1]

        _, dv_output, grads[i] = curr_layer['fwd_fn'](curr_input, curr_layer['params'], curr_layer['hyper_params'],
                                                           backprop=True, dv_output=dv_output)

    return grads


