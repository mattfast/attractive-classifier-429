import numpy as np

def loss_crossentropy(input, labels, hyper_params, backprop):
    """
    Args:
        input: [num_nodes] x [batch_size] array
        labels: [batch_size] array
        hyper_params: Dummy input. This is included to maintain consistency across all layer and loss functions, but the input argument is not used.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.

    Returns:
        loss: scalar value, the loss averaged over the input batch
        dv_input: The derivative of the loss with respect to the input. Same size as input.
    """

    assert labels.max() <= input.shape[0]
    loss = 0
    num_inputs = input.shape[-1]
    labels = labels.astype('int')
    for i in range(num_inputs):
        curr_label = labels[i]
        loss += -np.log(input[curr_label, i])/num_inputs

    eps = 0.00001
    dv_input = np.zeros(0)
    if backprop:
        dv_input = np.zeros(input.shape)
        for i in range(num_inputs):
            curr_label = labels[i]
            dv_input[curr_label, i] = -1/(input[curr_label, i] + eps)

    return loss, dv_input
