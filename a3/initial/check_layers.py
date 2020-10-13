import numpy as np
import os
import sys
sys.path += ['pyc_code', 'layers']

from fn_linear import fn_linear
from fn_linear_ import fn_linear as fn_linear_
from fn_softmax import fn_softmax
from fn_softmax_ import fn_softmax as fn_softmax_
from loss_crossentropy import loss_crossentropy
from loss_crossentropy_ import loss_crossentropy as loss_crossentropy_
from fn_conv import fn_conv
from fn_conv_ import fn_conv as fn_conv_

def pass_fail(cond):
    if cond:
        str = 'Passed!'
    else:
        str = 'Failed.'
    return str

def mse(val1, val2):
    assert val1.shape == val2.shape
    return np.sum((val1 - val2) ** 2) / val1.size

def check_layers():
    # test layers
    # test forward, gradient, backprop
    np.random.seed(0)
    err_thresh = 1e-6

    #-------------------------
    # Linear
    #-------------------------
    inp = np.random.rand(50, 20)
    w = np.random.rand(25, 50)
    b = np.random.rand(25, 1)
    dv = np.random.rand(25, 20)

    params = {'W': w, 'b': b}
    hyper = {'num_in': 50, 'num_out': 25}

    out, dv_in, grad = fn_linear(inp, params, hyper, True, dv)
    out_, dv_in_, grad_ = fn_linear_(inp, params, hyper, True, dv)

    print('Linear layer')

    if not (out.shape==out_.shape):
        print('\tout size does not match!')
        print('\tForward: %s' % pass_fail(False))
    else:
        err_out = mse(out, out_)
        print('\tForward: %s' % pass_fail(err_out < err_thresh))
    
    if not (grad["W"].shape==grad_["W"].shape):
        print('\tgrad W size does not match!')
        print('\tGradient: %s' % pass_fail(False))
    elif not (grad["b"].shape==grad_["b"].shape):
        print('\tgrad b size does not match!')
        print('\tGradient: %s' % pass_fail(False))
    else:
        err_grad = mse(np.concatenate((grad['W'].reshape(-1), grad['b'].reshape(-1))),
                       np.concatenate((grad_['W'].reshape(-1), grad_['b'].reshape(-1))))
        print('\tGradient: %s' % pass_fail(err_grad < err_thresh))
    
    if not (dv_in.shape==dv_in_.shape):
        print('\tdv_in size does not match!')
        print('\tBackpropagated error: %s' % pass_fail(False))
    else:
        err_dv = mse(dv_in, dv_in_)
        print('\tBackpropagated error: %s' % pass_fail(err_dv < err_thresh))

    #-------------------------
    # Softmax
    #-------------------------
    inp = np.random.rand(10, 20)
    dv = np.random.rand(10, 20)

    out, dv_in, _ = fn_softmax(inp, {}, {}, True, dv)
    out_, dv_in_, _ = fn_softmax_(inp, {}, {}, True, dv)


    print('Softmax layer')

    if not (out.shape==out_.shape):
        print('\tout size does not match!')
        print('\tForward: %s' % pass_fail(False))
    else:
        err_out = mse(out, out_)
        print('\tForward: %s' % pass_fail(err_out < err_thresh))

    if not (dv_in.shape==dv_in_.shape):
        print('\tdv_in size does not match!')
        print('\tBackpropagated error: %s' % pass_fail(False))
    else:
        err_dv = mse(dv_in, dv_in_)
        print('\tBackpropagated error: %s' % pass_fail(err_dv < err_thresh))


    #-------------------------
    # Crossentropy loss
    #-------------------------
    inp = np.random.rand(10, 20)
    labels = np.floor(np.random.rand(20,) * 10)

    loss, dv_in = loss_crossentropy(inp, labels, {}, True)
    loss_, dv_in_ = loss_crossentropy_(inp, labels, {}, True)

    print('Crossentropy loss')

    if not isinstance(loss, float):
        print('\tloss should be a scalar value!')
        print('\tLoss calculation: %s' % pass_fail(False))
    else:
        err_loss = (loss - loss_) ** 2
        print('\tLoss calculation: %s' % pass_fail(err_loss < err_thresh * 10))
    
    if not (dv_in.shape==dv_in_.shape):
        print('\tdv_in size does not match!')
        print('\tBackpropagated error: %s' % pass_fail(False))
    else:
        err_dv = mse(dv_in, dv_in_)
        print('\tBackpropagated error: %s' % pass_fail(err_dv < err_thresh * 10))


    #-------------------------
    # Convolution
    #-------------------------
    inp = np.random.rand(40, 30, 6, 10)
    w = np.random.rand(5, 5, 6, 7)
    b = np.random.rand(7, 1)

    w_flipped = np.zeros([5, 5, 6, 7])
    for i in range(7):
        for j in range(6):
            w_flipped[:, :, j, i] = np.rot90(w[:, :, j, i], 2)

    params = {'W': w, 'b': b}
    params_fl = {'W': w_flipped, 'b': b}

    hyper = {'filter_size': 5,
             'filter_depth': 6,
             'num_filters': 7}

    out_height = inp.shape[0] - params['W'].shape[0] + 1
    out_width = inp.shape[1] - params['W'].shape[1] + 1

    dv = np.random.rand(out_height, out_width, 7, 10)

    out, dv_in, grad = fn_conv(inp, params, hyper, True, dv)
    out_fl, dv_in_fl, grad_fl = fn_conv(inp, params_fl, hyper, True, dv)

    for i in range(7):
        for j in range(6):
            grad_fl['W'][:, :, j, i] = np.rot90(grad_fl['W'][:, :, j, i], 2)

    out_, dv_in_, grad_ = fn_conv_(inp, params, hyper, True, dv)

    # compare normal
    err_out = mse(out, out_)
    # compare flipped
    err_out_fl = mse(out_fl, out_)
    
    noflip = err_out < err_out_fl
    print('Convolutional layer')
    if not (out.shape==out_.shape):
        print('\tout size does not match!')
        print('\tForward: %s' % pass_fail(False))
    else:
        print('\tForward: %s' % pass_fail((err_out if noflip else err_out_fl) < err_thresh))

    if not (grad["W"].shape==grad_["W"].shape):
        print('\tgrad W size does not match!')
        print('\tGradient: %s' % pass_fail(False))
    elif not (grad["b"].shape==grad_["b"].shape):
        print('\tgrad b size does not match!')
        print('\tGradient: %s' % pass_fail(False))
    else:
        # compare normal
        err_grad = mse(np.concatenate((grad['W'].reshape(-1), grad['b'].reshape(-1))),
                   np.concatenate((grad_['W'].reshape(-1), grad_['b'].reshape(-1))))
        # compare flipped
        err_grad_fl = mse(np.concatenate((grad_fl['W'].reshape(-1), grad_fl['b'].reshape(-1))),
                   np.concatenate((grad_['W'].reshape(-1), grad_['b'].reshape(-1))))

        print('\tGradient: %s' % pass_fail((err_grad if noflip else err_grad_fl) < err_thresh))
    
    if not (dv_in.shape==dv_in_.shape):
        print('\dv_in size does not match!')
        print('\tBackpropagated error: %s' % pass_fail(False))
    else:
        # compare normal
        err_dv = mse(dv_in, dv_in_)
        # compare flipped
        err_dv_fl = mse(dv_in_fl, dv_in_)
        print('\tBackpropagated error: %s' % pass_fail((err_dv if noflip else err_dv_fl) < err_thresh))


if __name__ == '__main__':
    check_layers()