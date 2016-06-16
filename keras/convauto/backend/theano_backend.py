import theano.tensor as T

def unpool2d(maxpool_out, maxpool_in, output):
    f = T.grad(T.sum(maxpool_out), wrt= maxpool_in) * output
    return f

def conv2d_transpose(value, filter, output_shape, subsample, border_mode, dim_ordering='th'):
    raise NotImplementedError, "Conv2D Transpose not implemented in Theano yet"

def l2_loss(x, x_):
    raise NotImplementedError, "L2 loss not implemented in Theano yet"

def get_shape_with_batch(x, dim_ordering='th'):
    raise NotImplementedError, "shape_with_batch not implemented in Theano, and may not even be necissary..."
