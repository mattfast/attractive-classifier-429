import numpy as np
import sys
sys.path += ['layers']
from fn_flatten import fn_flatten
from fn_relu import fn_relu
from fn_pool import fn_pool

######################################################
# Set use_pcode to True to use the provided pyc code for layer functions
use_pcode = False

# You can modify the imports of this section to indicate 
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ['pyc_code']
    from fn_linear_ import fn_linear
    from fn_conv_ import fn_conv
    from fn_softmax_ import fn_softmax
else:
    # import your own implementation
    from fn_linear import fn_linear
    from fn_conv import fn_conv
    from fn_softmax import fn_softmax
######################################################

def init_layers(type, info):
    """
    Given a layer name, initializes the layer structure properly with the
    weights randomly initialized.

    Input:
    	type - Layer name (options: 'linear', 'conv', 'pool', 'softmax', 'flatten', 'relu')
    	info - Dictionary holding hyper parameters that define the layer

      Examples: init_layers('linear', {'num_in': 18, 'num_out': 10})
     			init_layers('softmax',{})
    """

    # Parameters for weight initialization
    weight_init = np.random.randn

    ws = info.get('weight_scale', 0.1)
    bs = info.get('bias_scale', 0.1)

    params = {'W': np.zeros(0),
              'b': np.zeros(0)}

    if type == 'linear':
        # Requires num_in, num_out
        fn = fn_linear
        W = weight_init(info['num_out'], info['num_in']) * ws
        b = weight_init(info['num_out'], 1) * bs
        params['W'] = W
        params['b'] = b
    elif type == 'conv':
        # Requires filter_size, filter_depth, num_filters
        fn = fn_conv
        W = weight_init(info['filter_size'], info['filter_size'], info['filter_depth'], info['num_filters']) * ws
        b = weight_init(info['num_filters'], 1) * bs
        params['W'] = W
        params['b'] = b
    elif type == 'pool':
        # Requires filter_size and optionally stride (default stride = 1)
        fn = fn_pool
    elif type == 'softmax':
        fn = fn_softmax
    elif type == 'flatten':
        fn = fn_flatten
    elif type == 'relu':
        fn = fn_relu
    else:
        assert False, 'type %s not supported' % type

    layers = {'fwd_fn': fn,
              'type': type,
              'params': params,
              'hyper_params': info}

    return layers
