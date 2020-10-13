import numpy as np
import scipy.signal
import scipy.ndimage

def fn_pool(input, params, hyper_params, backprop, dv_output=None):
    """
    Do pooling, currently only max pooling is implemented
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Dummy input. This is included to maintain consistency across all layers, but the input argument is not used.
        hyper_params: Information describing the layer.
            hyper_params['filter_size']: int
            hyper_params['method']: 'max', or 'mean' (not fully implemented)
            hyper_params['stride']: int, stride size
            hyper_params['pad']: int, number of paddings on each side of the input for the first two dimensions
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_channels] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: Dummy output. This is included to maintain consistency in the return values of layers, but there is no gradient to calculate in the softmax layer since there are no weights to update.
    """

    in_height, in_width, num_channels, batch_size = input.shape
    filter_size = hyper_params['filter_size']

    # Check for hyperparameters or use default values
    method = hyper_params.get('method', 'max')
    stride = hyper_params.get('stride', 1)
    
    if 'pad' in hyper_params.keys():
        pad = hyper_params['pad']
        input = np.pad(input, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='edge')

    # out_size = (in_size + 2 * pad - filter_size + 1) / stride
    # this is written for square filters and padding
    h = input.shape[0] - filter_size + 1
    w = input.shape[1] - filter_size + 1
    out_height = np.ceil(h / stride)
    out_width = np.ceil(w / stride)

    # Do pooling
    output = np.zeros((int(out_height), int(out_width), num_channels, batch_size))

    if method == 'max' and backprop:
        max_rows = np.zeros(output.shape).astype('i')
        max_cols = np.zeros(output.shape).astype('i')

    for i in range(batch_size):
        for j in range(num_channels):
            filter = np.ones((filter_size, filter_size))

            if method == 'mean':
                pooled = scipy.signal.convolve(input[:, :, j, i], filter, 'valid') / (filter_size ** 2)
            elif method == 'max':
                pooled = scipy.ndimage.rank_filter(input[:, :, j, i], -1, filter_size)
                # rank_filter generic_filter do their own padding which we don't want, so this line undoes that
                offset = filter_size // 2
                pooled = pooled[offset:offset+h, offset:offset+w]

                # If we are doing backpropagation we need to save where the max values came from
                if backprop:
                    # use generic_filter to avoid getting ties
                    max_idxs = scipy.ndimage.generic_filter(input[:, :, j, i], np.argmax, filter_size)
                    max_idxs = max_idxs[offset:offset+h, offset:offset+w]

                    # The indices returned in max_idxs are all local, so we need to adjust them to global subscripts
                    max_r, max_c = np.unravel_index(max_idxs.astype('i'), filter.shape)
                    r_offset = np.arange(h)
                    c_offset = np.arange(w)
                    max_r = max_r + np.expand_dims(r_offset, 1)
                    max_c = max_c + np.expand_dims(c_offset, 0)

                    max_rows[:, :, j, i] = max_r[:h:stride, :w:stride]
                    max_cols[:, :, j, i] = max_c[:h:stride, :w:stride]
            else:
                assert False, 'metdhod %s not supported' % method

            output[:, :, j, i] = pooled[:h:stride, :w:stride]

    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}

    if backprop:
        dv_input = np.zeros(input.shape)
        for i in range(batch_size):
            for j in range(num_channels):
                if method == 'mean':
                    # TODO
                    pass
                elif method == 'max':
                    rows = max_rows[:, :, j, i]
                    cols = max_cols[:, :, j, i]
                    dv_out = dv_output[:, :, j, i]

                    for mm in range(rows.shape[0]):
                        for nn in range(rows.shape[1]):
                            dv_input[rows[mm, nn], cols[mm, nn], j, i] = dv_input[rows[mm, nn], cols[mm, nn], j, i] + dv_out[mm, nn]


    return output, dv_input, grad
