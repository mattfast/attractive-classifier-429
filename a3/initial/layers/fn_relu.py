import numpy as np

def fn_relu(input, params, hyper_params, backprop, dv_output=None):
    """
    Rectified linear unit activation function
    """

    output = np.maximum(input, 0)

    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}

    if backprop:
        dv_input = dv_output
        dv_input[output == 0] = 0

    return output, dv_input, grad
