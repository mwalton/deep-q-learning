import tensorflow as tf


L2M_COLLECTION = '_l2m'

''
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("MaxPoolGrad")
def _MaxPoolGradGrad(op, grad):
    ksize = op.get_attr("ksize")
    strides = op.get_attr("strides")
    padding = op.get_attr("padding")
    data_format = op.get_attr("data_format")
    grads = []
    for i in op.inputs:
        igrad = gen_nn_ops._max_pool_grad(i, op.outputs[0], grad, ksize, strides,
                                          padding=padding,data_format=data_format)
        grads.append(igrad)

    return grads

def unpool2d(maxpool_out, maxpool_in, output):
    f = tf.gradients(tf.reduce_sum(maxpool_out), maxpool_in)[0] * output
    return f

def get_shape_with_batch(x, dim_ordering='th'):
    if dim_ordering == 'th':
        shape = x.get_shape()
        batch_shape = tf.shape(x)[0]
        output_shape = tf.pack([batch_shape, shape[2], shape[3], shape[1]])
    elif dim_ordering == 'tf':
        shape = x.get_shape()
        batch_shape = tf.shape(x)[0]
        output_shape = tf.pack([batch_shape, shape[1], shape[2], shape[3]])

    return output_shape


def conv2d_transpose(x, kernel, output_shape, subsample, border_mode, dim_ordering='th'):
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    strides = (1,) + subsample + (1,)

    if dim_ordering == 'th':
        x = tf.transpose(x, (0, 2, 3, 1))
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, strides, padding=padding)
        x = tf.transpose(x, (0, 3, 1, 2))
    elif dim_ordering == 'tf':
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, strides, padding=padding)
    else:
        raise Exception('Unknown dim_ordering: ' + str(dim_ordering))

    return x


def l2_loss(x, x_):
    return l2(x - x_)

def l2(tensor, weight=1.0, scope=None):
    """Define a L2Loss, useful for regularize, i.e. weight decay.
    Args:
      tensor: tensor to regularize.
      weight: an optional weight to modulate the loss.
      scope: Optional scope for op_scope.
    Returns:
      the L2 loss op.
    """
    with tf.op_scope([tensor], scope, 'L2Loss'):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
        return loss
