import numpy as np

def fn_softmax(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [num_nodes] x [batch_size] array
        params: Dummy input. This is included to maintain consistency across all layers, but the input argument is not used.
        hyper_params: Dummy input. This is included to maintain consistency across all layers, but the input argument is not used.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [num_nodes] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: Dummy output. This is included to maintain consistency in the return values of layers, but there is no gradient to calculate in the softmax layer since there are no weights to update.
    """

    num_nodes, batch_size = input.shape
    #exp_input = np.exp(input)

    # Initialize
    output = np.zeros([num_nodes, batch_size])
    dv_input = np.zeros(0)
    # grad is included to maintain consistency in the return values of layers,
    # but there is no gradient to calculate in the softmax layer since there
    # are no weights to update.
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}

    # TODO: FORWARD CODE
    #       Update output with values
    for i in range(batch_size):
        exp_input = np.exp(input[:,i])
        output[:,i] = exp_input / np.sum(exp_input)

    if backprop:
        assert dv_output is not None
        dv_input = np.zeros([num_nodes, batch_size])

        # TODO: BACKPROP CODE
        #       Update dv_input with values
        # https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
        for k in range(batch_size):
            dy_dx = np.zeros([num_nodes, num_nodes])
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        dy_dx[i, j] = output[i, k] * (1 - output[i, k])
                    else:
                        dy_dx[i, j] = - output[i, k] * output[j, k]

            dv_input[:, k] = dy_dx @ dv_output[:, k]


    return output, dv_input, grad
