import numpy as np

def fn_flatten(input, params, hyper_params, backprop, dv_output=None):
    """
    Flatten all but the last dimension of the input. 
    Args:
        input: The input data to the layer function. [any dimensions] x [batch_size] array
        params: Dummy input. This is included to maintain consistency across all layers, but the input argument is not used.
        hyper_params: Dummy input. This is included to maintain consistency across all layers, but the input argument is not used.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [product of first input dims] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: Dummy output. This is included to maintain consistency in the return values of layers, but there is no gradient to calculate in the softmax layer since there are no weights to update.
    """

    in_dim = input.shape
    batch_size = in_dim[-1]

    output = np.reshape(input, (-1, batch_size), order='F')

    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}

    if backprop:
        assert dv_input is not None
        dv_input = np.reshape(dv_output, in_dim, order='F')

    return output, dv_input, grad
