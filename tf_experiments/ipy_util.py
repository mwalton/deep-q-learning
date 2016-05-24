import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import matplotlib.pyplot as plt

def get_mnist():
    return input_data.read_data_sets('MNIST_data', one_hot=True)

def img_shape(x):
    return tf.reshape(x, [1, 28, 28, 1])

def show_img(x):
    plt.imshow( 1 - x.reshape((x.shape[1],x.shape[2])), cmap='Greys_r', interpolation='nearest')

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def pool(value, name='pool'):
    """Downsampling operation.
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, d0/2, d1/2, ..., dn/2, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        out = value
        for sh_i in sh[1:-1]:
            assert sh_i % 2 == 0
        for i in range(len(sh[1:-1])):
            out = tf.reshape(out, (-1, 2, np.prod(sh[i + 2:])))
            out = out[:, 0, :]
        out_size = [-1] + [int(math.ceil(s / 2)) for s in sh[1:-1]] + [sh[-1]]
        print out_size
        out = tf.reshape(out, out_size, name=scope)
    return out