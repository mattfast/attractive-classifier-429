import numpy as np
import scipy.signal

def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """

    in_height, in_width, num_channels, batch_size = input.shape
    _, _, filter_depth, num_filters = params['W'].shape
    out_height = in_height - params['W'].shape[0] + 1
    out_width = in_width - params['W'].shape[1] + 1

    assert params['W'].shape[2] == input.shape[2], 'Filter depth does not match number of input channels'

    # Initialize
    output = np.zeros((out_height, out_width, num_filters, batch_size))
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}

    print(dv_output.shape)

    for i in range(batch_size):
        for j in range(num_filters):
            for k in range(filter_depth):
                img = input[:,:,k,i]
                filter = params['W'][:,:,k,j]
                output[:,:,j,i] += scipy.signal.convolve(img, filter, mode='valid')

            output[:,:,j,i] += params['b'][j, 0]

    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)

        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(filter_depth):
                    flipped_img = np.flip(input[:,:,k,i],axis=(0,1))
                    filter = dv_output[:,:,j,i]
                    grad['W'][:,:,k,j] += scipy.signal.convolve(flipped_img, filter, mode='valid')

        grad['W'] = grad['W'] / batch_size
        grad['b'][:,0] = np.sum(dv_output, axis=(0,1,3)) / batch_size

        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(filter_depth):
                    img = dv_output[:,:,j,i]
                    filter = np.flip(params['W'][:,:,k,j],axis=(0,1))
                    dv_input[:,:,k,i] += scipy.signal.convolve(img, filter, mode='full')
        

            
        # TODO: BACKPROP CODE
        #       Update dv_input and grad with values

    return output, dv_input, grad
