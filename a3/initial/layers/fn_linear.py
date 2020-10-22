import numpy as np

def fn_linear(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [num_in] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [num_out] x [num_in] array
            params['b']: layer bias, [num_out] x 1 array
        hyper_params: Information describing the layer.
            hyper_params['num_in']: number of inputs for layer
            hyper_params['num_out']: number of outputs for layer
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [num_out] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """

    W = params['W']
    b = params['b']

    num_in, batch_size = input.shape
    if num_in != hyper_params['num_in']:
        print('Incorrect number of inputs provided at linear layer.\n Got %d inputs,  expected %d.' % num_in, hyper_params['num_in'])
        raise

    # Initialize
    output = np.zeros([hyper_params['num_out'], batch_size])
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}

    # TODO: FORWARD CODE
    #       Update output with values
    output = W @ input + b


    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad = {'W': np.zeros(W.shape),
                'b': np.zeros(b.shape)}

        # TODO: BACKPROP CODE
        #       Update dv_input and grad with values
        dv_input = W.transpose() @ dv_output # W transpose is dy/dx. Left multiplication to match input dimensions
        grad['W'] = dv_output @ input.transpose() / batch_size # Right multiplication to match W dimensions
        grad['b'][:, 0] = np.sum(dv_output, axis=1) / batch_size

    return output, dv_input, grad
