import sys
sys.path += ['layers']
import numpy as np
from matplotlib import pyplot as plt
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
    lr = params.get("learning_rate", .05)
    # Weight decay
    wd = params.get("weight_decay", .00005)
    # Friction term
    rho = params.get("rho", .99)
    # Batch size
    batch_size = params.get("batch_size", 128)
    plateau_iteration_cutoff = 50
    plateau_ratio_cutoff = 0.995

    # Input contains both train and test data. Split this up.
    train_set = input[..., :60000]
    train_label = label[:60000]

    test_set = input[..., 60000:]
    test_label = label[60000:]

    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'full_model.npz')
    
    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr, 
                     "weight_decay": wd,
                     "rho": rho}
    
    num_inputs = train_set.shape[-1]
    train_loss = np.zeros((0,))
    test_loss = np.zeros((0, ))
    num_layers = len(model["layers"])
    velocity = [{}] * num_layers
    for i in range(num_layers):
        velocity_val = {}
        velocity_val['W'] = np.zeros(model['layers'][i]['params']['W'].shape)
        velocity_val['b'] = np.zeros(model['layers'][i]['params']['b'].shape)
        velocity[i] = velocity_val
        print(f"Shape of {i}: {velocity[i]['W'].shape}")


    curr_best_loss = -1
    num_iterations_since_best = 0

    for i in range(numIters):

        batch = np.random.choice(num_inputs, batch_size, replace=False)
        train_inputs = train_set[..., batch]
        train_labels = train_label[batch].astype('int')

        output, activations = inference(model, train_inputs)
        print("Finished inference step")
        curr_loss, dv_output = loss_crossentropy(output, train_labels, hyper_params=None, backprop=True)
        curr_grads = calc_gradient(model, train_inputs, activations, dv_output)
        print("Finished Calculating Gradients")

        train_loss = np.append(train_loss, curr_loss)

        # Calculate accuracy
        max_values = np.argmax(output, axis=0)
        assert len(max_values) == len(train_labels) # Should be equal in size
        train_accuracy = np.sum(np.equal(max_values, train_labels)) / len(max_values)

        for j in range(num_layers):
            new_velocity = {}
            new_velocity['W'] = rho*velocity[j]['W'] + curr_grads[j]['W']
            new_velocity['b'] = rho*velocity[j]['b'] + curr_grads[j]['b']
            velocity[j] = new_velocity

        # Passing in velocity for gradient accomplishes the same update step for momentum
        model = update_weights(model, velocity, update_params)
        print(f"Current training loss on {i}th iteration: {curr_loss}.")
        print(f"Current training accuracy on {i}th iteration: {train_accuracy}.")

        lr = lr * 0.99
        update_params['lr'] = lr

        # Calculate Test Loss on mini-Batch
        test_batch = np.random.choice(test_set.shape[-1], batch_size, replace=False)
        test_inputs = test_set[..., test_batch]
        test_labels = test_label[test_batch].astype('int')

        test_output, _ = inference(model, test_inputs)
        curr_test_loss, _ = loss_crossentropy(test_output, test_labels, hyper_params=None, backprop=True)

        max_test_values = np.argmax(test_output, axis=0)
        test_loss = np.append(test_loss, curr_test_loss)
        test_accuracy = np.sum(np.equal(max_test_values, test_labels)) / len(max_test_values)
        print(f"Current test loss on {i}th iteration: {curr_test_loss}.")
        print(f"Current test accuracy on {i}th iteration: {test_accuracy}.")

        if test_accuracy > 0.98:
            print("Saving model...")
            np.savez(save_file, **model)
            print("Model Saved.")
            break

        plateau_ratio = 1

        if curr_best_loss == -1:
            plateau_ratio = 0
        elif curr_loss < curr_best_loss:
            plateau_ratio = curr_loss / curr_best_loss

        if plateau_ratio < plateau_ratio_cutoff:
            curr_best_loss = curr_loss
            num_iterations_since_best = 0

            print("Saving model...")
            np.savez(save_file, **model)
            print("Model Saved.")
        else:
            num_iterations_since_best += 1

        if num_iterations_since_best > plateau_iteration_cutoff:
            break

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

    plt.scatter(list(range(test_loss.shape[0])), test_loss)
    plt.title('Cross-Entropy Test Loss over Iteration Counts')
    plt.xlabel('Num Iterations')
    plt.ylabel('Test Loss')
    plt.show()
    plt.savefig('test_loss.png')

    return model, train_loss

