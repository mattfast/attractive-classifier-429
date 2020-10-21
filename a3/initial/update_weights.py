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
        updated_model['layers'][i]['params']['b'] += (-a*grads[i]['b'])
        curr_weight = updated_model['layers'][i]['params']['W']
        updated_model['layers'][i]['params']['W'] += (-a*grads[i]['W']) + (-lmd*curr_weight)

    return updated_model