import numpy as np

def loss_euclidean(input, labels, hyper_params, backprop):
    """
    Args:
        input: [any dimensions] x [batch_size]
        labels: same size as input
        hyper_params: Dummy input. This is included to maintain consistency across all layer and loss functions, but the input argument is not used.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
    
    Returns:
        loss: scalar value, the loss averaged over the input batch
        dv_input: The derivative of the loss with respect to the input. Same size as input.
    """

    assert labels.shape == input.shape

    batch_size = input.shape[-1]

    diff = input - labels
    loss = np.sum(diff ** 2) / (2 * batch_size)

    dv_input = np.zeros(0)
    if backprop:
        dv_input = diff

    return loss, dv_input
