import sys
sys.path += ['layers']
import numpy as np

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = True

# You can modify the imports of this section to indicate 
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ['pyc_code']
    from inference_ import inference
    from calc_gradient_ import calc_gradient
    from loss_crossentropy_ import loss_crossentropy
    from update_weights_ import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from loss_crossentropy import loss_crossentropy
    from update_weights import update_weights
######################################################

def train(model, input, label, params, numIters):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        label: [batch_size]
        params: Paramters for configuring training
            params["learning_rate"] 
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
    '''
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", .01)
    # Weight decay
    wd = params.get("weight_decay", .0005)
    # Friction term
    rho = params.get("rho", .9)
    # Batch size
    batch_size = params.get("batch_size", 128)

    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'model.npz')
    
    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr, 
                     "weight_decay": wd,
                     "rho": rho}
    
    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,))
    num_layers = len(model["layers"])
    velocity = [None,] * num_layers
    for i in range(num_layers):
        velocity[i]['W'] = np.zeros(model['layers'][i]['params']['W'].shape)
        velocity[i]['b'] = np.zeros(model['layers'][i]['params']['b'].shape)

    for i in range(numIters):
        batch = np.random.choice(num_inputs, batch_size, replace=False)
        train_inputs = input[..., batch]
        train_labels = label[batch]

        output, activations = inference(model, train_inputs)
        curr_loss, dv_output = loss_crossentropy(output, train_labels, hyper_params=None, backprop=True)
        loss[i] = curr_loss
        curr_grads = calc_gradient(model, train_inputs, activations, dv_output)

        # TODO: calculate accuracy

        for j in range(num_layers):
            velocity[j]['W'] = (rho*velocity[j]['W'] + curr_grads[j]['W'])
            velocity[j]['b'] = (rho*velocity[j]['b'] + curr_grads[j]['b'])

        # Passing in velocity for gradient accomplishes the same update step for momentum
        model = update_weights(model, velocity, update_params)
        print(f"Current loss on {i}th iteration: {curr_loss}.")

        print("Saving model...")
        np.savez(save_file, **model)
        print("Model Saved.")

        # Calculate and store Velocity
        # Steps:
        #   (1) Select a subset of the input to use as a batch
        #   (2) Run inference on the batch
        #   (3) Calculate loss and determine accuracy
        #   (4) Calculate gradients
        #   (5) Update the weights of the model
        # Optionally,
        #   (1) Monitor the progress of training
        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``
    np.savez(save_file, **model)
    return model, loss

