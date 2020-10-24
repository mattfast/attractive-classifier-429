import numpy as np

def update_weights(model, grads, hyper_params):
    '''
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params: 
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns: 
        updated_model:  Dictionary holding the updated model
    '''
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]
    updated_model = model

    for i in range(num_layers):
        curr_bias = updated_model['layers'][i]['params']['b']
        curr_bias += (-a*grads[i]['b'])
        updated_model['layers'][i]['params']['b'] = curr_bias

        curr_weight = updated_model['layers'][i]['params']['W']
        curr_weight += (-a*grads[i]['W']) + (-lmd*curr_weight)
        updated_model['layers'][i]['params']['W'] = curr_weight

    return updated_model