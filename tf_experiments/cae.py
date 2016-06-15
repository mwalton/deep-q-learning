import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax):
  return gen_nn_ops._max_pool_grad(op.inputs[0],
                                   op.outputs[0],
                                   grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format='NHWC')

L1_COLLECTION = '_l1'
L2M_COLLECTION = '_l2m'

# int and get tf flags dict; set from command line may be a good idea
def get_flags(args):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model_type', 'cae', 'Model topology spec, either cnn, upsample, zeiler, swwae')
    flags.DEFINE_integer('batch_size', 100, 'Number of examples per batch')
    flags.DEFINE_integer('image_size', 28, 'Size of n x n images, should be 28 for MNIST')
    flags.DEFINE_integer('num_classes', 10, 'Number of labels or classes, should be 10 for MNIST')
    flags.DEFINE_integer('num_channels', 1, 'Input channels, should be 1 for MNIST')
    flags.DEFINE_integer('num_epochs', 1, 'Number of epochs')
    flags.DEFINE_integer('seed', 666, 'The number of the seed')
    flags.DEFINE_string('logdir', 'stats', 'Location of tensorboard log directory')
    flags.DEFINE_string('chkdir', 'checkpoints', 'Location to save model checkpoints')
    flags.DEFINE_integer('eval_frequency', 100, 'Number of steps between validations on hold out')
    flags.DEFINE_integer('eval_batch_size', 100, 'Batch size for validation')
    flags.DEFINE_integer('checkpoint_freq', 200, 'Save model parameters on this step interval')
    flags.DEFINE_boolean('load_checkpoint', False, 'Load weights from a previous checkpoint')
    flags.DEFINE_boolean('train', True, 'Should the model be trained')
    flags.DEFINE_boolean('test', False, 'Should the model be tested')
    return FLAGS

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

# helper function for attaching an activation summary
def activation_summary(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def l1_loss(tensor, weight=1.0, scope=None):
  """Define a L1Loss, useful for regularize, i.e. lasso.
  Args:
    tensor: tensor to regularize.
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.
  Returns:
    the L1 loss op.
  """
  with tf.op_scope([tensor], scope, 'L1Loss'):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
    tf.add_to_collection(L1_COLLECTION, loss)
    return loss


def l2_loss(tensor, weight=1.0, scope=None):
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
    tf.add_to_collection(L2M_COLLECTION, loss)
    return loss

def weight_variable(shape, name=None):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=1e-4)
    #initial = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    return tf.Variable(initial, name=name, dtype=tf.float32)

def bias_variable(shape, name=None):
    """Create a bias variable with appropriate initialization."""
    initial = tf.zeros(shape=shape)
    return tf.Variable(initial, name=name)

# helper function for initializing conv layers
def conv2d(name, input, kernel_shape, padding='SAME'):
    with tf.variable_scope(name) as scope:
        kernel = weight_variable(kernel_shape, name='weights')
        bias = bias_variable([kernel_shape[-1]], name='bias')
        conv = tf.nn.conv2d(input, kernel, strides=[1,1,1,1], padding=padding)
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)
        l1 = l1_loss(relu)

        activation_summary(relu)


    return (relu, kernel)


def deconv2d(name, input, kernel, output_shape, padding='SAME'):
    with tf.variable_scope(name) as scope:
        bias = bias_variable([kernel.get_shape()[-2]], name='bias')
        deconv = tf.nn.conv2d_transpose(input, kernel, output_shape,
                                        strides=[1,1,1,1], padding=padding)
        relu = tf.nn.relu(tf.nn.bias_add(deconv, bias), name=scope.name)
        activation_summary(relu)

    return relu


# helper function for initializing max pooling layers
def max_pool(name, input, padding='SAME'):
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],
                              padding=padding, name=scope.name)

    return pool

def max_pool_argmax(name, input, padding='SAME'):
    with tf.variable_scope(name) as scope:
        maxpool, argmax = tf.nn.max_pool_with_argmax(input,
                                                     [1,2,2,1],
                                                     [1,2,2,1],
                                                     padding=padding,
                                                     name=scope.name)
    return (maxpool, argmax)

def argmax_unpool(name, maxpool, argmax, batch_size, padding='SAME'):
    with tf.variable_scope(name) as scope:
        max_shape = [s.value for s in maxpool.get_shape()]
        flat_len = batch_size * max_shape[1] * max_shape[2] * max_shape[3]
        unpool_shape = [-1, 2*max_shape[1], 2*max_shape[2], max_shape[3]]

        '''Need to correct 1D index to include batch offset. This
           is apparently a bug in tensorflow max_pool_with_argmax which
           should compute flattend index as ((b * height + y) * width + x) * channels + c
           according to https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#max_pool_with_argmax
           bug tracked here: https://github.com/tensorflow/tensorflow/issues/2821'''
        # base offset for first dimension
        base = tf.cast(unpool_shape[1] * unpool_shape[2] * unpool_shape[3], tf.int64)
        # range index over each batch [1 ... B]
        b = tf.cast(tf.range(0, batch_size), tf.int64)
        batch_offset = tf.mul(base, tf.reshape(b, [-1,1,1,1]))
        argmax_ = tf.add(batch_offset, argmax)

        # flatten and initialize sparse tensor
        maxflat = tf.reshape(maxpool, [-1])
        argflat = tf.reshape(argmax_, [-1])
        sparse_unpool = tf.sparse_to_dense(argflat, [flat_len * 4], maxflat, validate_indices=False)
        unpool = tf.reshape(sparse_unpool, unpool_shape)
        activation_summary(unpool)

    return unpool

def upsample(name, input):
    with tf.variable_scope(name) as scope:
        shape = [s.value for s in input.get_shape()]
        upsample_shape = [2 * shape[1], 2 * shape[2]]
        ups = tf.image.resize_nearest_neighbor(input, size=upsample_shape)
        activation_summary(ups)

    return ups

# fully input
def flatten(name, input):
    with tf.variable_scope(name) as scope:
        in_shape = [s.value for s in input.get_shape()]
        flat = tf.reshape(input, [-1, in_shape[1] * in_shape[2] * in_shape[3]])
    return flat

def fully_connected(name, input, out_dim):
    with tf.variable_scope(name) as scope:
        in_dim = input.get_shape()[1].value
        weights = weight_variable([in_dim, out_dim], name='weights')
        bias = bias_variable([out_dim], name='bias')
        relu = tf.nn.relu(tf.matmul(input, weights) + bias, name=scope.name)
        activation_summary(relu)

    return relu

def softmax(name, input, out_dim):
    with tf.variable_scope(name) as scope:
        in_dim = input.get_shape()[1].value
        weights = weight_variable([in_dim, out_dim], name='weights')
        bias = bias_variable([out_dim], name='bias')
        softmax_op = tf.nn.softmax(tf.matmul(input, weights) + bias)
    return softmax_op

def cross_entropy(name, y_target, y_pred):
    with tf.variable_scope(name) as scope:
        xentropy = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(y_pred), reduction_indices=[1]))
        tf.scalar_summary('cross entropy', xentropy)

    return xentropy

def mean_squared_err(name, x, x_):
    with tf.variable_scope(name) as scope:
        meansq = tf.reduce_mean(tf.square(x_ - x))
        tf.scalar_summary('mean square error', meansq)

    return meansq

def zeiler_loss(name, x, x_, reconstruction_weight, sparsity_weight):
    with tf.variable_scope(name) as scope:
        feature_l1 = tf.pack(tf.get_collection(L1_COLLECTION))
        sparsity_loss = tf.mul(sparsity_weight, tf.reduce_sum(feature_l1))
        reconstruction_loss = l2_loss(x - x_, reconstruction_weight)

        loss = tf.add(sparsity_loss, reconstruction_loss, name=scope.name)
        tf.scalar_summary(scope.name, loss)
    return loss

def swwae_loss(name, x, x_, y=None, lambda_L2rec=1.0, lambda_L2M=1.0):
    with tf.variable_scope(name) as scope:
        reconstruction_loss = tf.nn.l2_loss(x - x_)

        loss = reconstruction_loss
        tf.scalar_summary(scope.name, loss)

    return loss

def accuracy(name, y_target, y_pred):
    with tf.variable_scope(name) as scope:
        correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_target,1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary(scope.name, acc)
    return acc


def train(loss):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return train_op

def feed_dict(x, y, data, batch_size, train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        images, labels = data.train.next_batch(batch_size)
    else:
        images, labels = data.test.next_batch(batch_size)
    return {x: images, y: labels, B: batch_size}


if __name__=='__main__':
    FLAGS = get_flags(None)
    tf.set_random_seed(FLAGS.seed)

    # make directories if not exist
    make_dirs([FLAGS.logdir, FLAGS.chkdir])

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # shape variables for images and classes, first dim is batch sz
    x_shape = (None, FLAGS.image_size * FLAGS.image_size)
    y_shape = (None, FLAGS.num_classes)

    # initializes the input and output nodes of the graph
    x = tf.placeholder(dtype=tf.float32, shape=x_shape)
    y = tf.placeholder(dtype=tf.float32, shape=y_shape)
    B = tf.placeholder(tf.int32)

    # reshape the input into images
    x_image = tf.reshape(x, [-1,FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels])
    tf.image_summary('input', x_image, max_images=5)


    if FLAGS.model_type == 'cnn':
        # fully connected layer
        conv, _ = conv2d('conv1', x_image, [5,5,FLAGS.num_channels,32])
        pool = max_pool('max_pool1', conv)
        conv, _ = conv2d('conv2', pool, [5,5,32,64])
        pool = max_pool('max_pool2', conv)
        flat = flatten('flatten', pool)
        hidden = fully_connected('hidden', flat, 512)
        y_ = softmax('softmax', hidden, FLAGS.num_classes)

        # train using cross entropy
        loss = cross_entropy('loss', y, y_)
        accuracy_op = accuracy('accuracy', y, y_)

    elif FLAGS.model_type == 'upsample':
        shape0 = [s.value for s in x_image.get_shape()]
        shape0 = [FLAGS.batch_size] + shape0[1:]
        conv, kernel0 = conv2d('conv1', x_image, [3,3,FLAGS.num_channels,32])
        pool = max_pool('max_pool1', conv)
        shape1 = [s.value for s in pool.get_shape()]
        shape1 = [FLAGS.batch_size] + shape1[1:]
        conv, kernel1 = conv2d('conv2', pool, [5,5,32,64])
        pool = max_pool('max_pool2', conv)
        unpool = upsample('upsample1', pool)
        convT = deconv2d('deconv1', unpool, kernel1, shape1)
        unpool = upsample('upsample', convT)
        convT = deconv2d('deconv2', unpool, kernel0, shape0)

        tf.image_summary('reconstruction', convT, max_images=5)

        loss = mean_squared_err('loss', x_image, convT)

    elif FLAGS.model_type == 'swwae':
        conv0 = tf.contrib.layers.convolution2d(x_image, 16, [5,5], normalizer_fn='batch_norm')
        activation_summary(conv0)
        conv1 = tf.contrib.layers.convolution2d(conv0, 32, [3,3], normalizer_fn='batch_norm')
        activation_summary(conv1)
        pool, argmax0 = max_pool_argmax('pool0', conv1)
        unpool = argmax_unpool('unpool0', pool, argmax0, FLAGS.batch_size)
        deconv = tf.contrib.layers.convolution2d(x_image, 1, [5,5])

        tf.image_summary('reconstruction', deconv, max_images=5)

        loss = swwae_loss('loss', x_image, deconv)

    elif FLAGS.model_type == 'single_unpool':
        shape0 = [s.value for s in x_image.get_shape()]
        shape0 = [FLAGS.batch_size] + shape0[1:]
        conv, kernel0 = conv2d('conv1', x_image, [5,5,FLAGS.num_channels,16])
        shape1 = [s.value for s in conv.get_shape()]
        shape1 = [FLAGS.batch_size] + shape1[1:]
        conv, kernel1 = conv2d('conv2', conv, [3,3,16,32])
        pool, argmax = max_pool_argmax('pool1', conv)
        unpool = argmax_unpool('unpool1', pool, argmax, FLAGS.batch_size)
        convT = deconv2d('deconv1', unpool, kernel1, shape1)
        convT = deconv2d('deconv2', convT, kernel0, shape0)

        tf.image_summary('reconstruction', convT, max_images=5)

        loss = swwae_loss('loss', x_image, convT)


    elif FLAGS.model_type == 'zeiler':
        # convolutional and max pooling layers
        shape0 = [s.value for s in x_image.get_shape()]
        shape0 = [FLAGS.batch_size] + shape0[1:]
        conv, kernel0 = conv2d('conv1', x_image, [3,3,FLAGS.num_channels,32])
        pool, argmax0 = max_pool_argmax('max_pool1', conv)
        shape1 = [s.value for s in pool.get_shape()]
        shape1 = [FLAGS.batch_size] + shape1[1:]
        conv, kernel1 = conv2d('conv2', pool, [5,5,32,64])
        pool, argmax1 = max_pool_argmax('max_pool2', conv)
        unpool = argmax_unpool('unpool1', pool, argmax1, FLAGS.batch_size)
        convT = deconv2d('deconv1', unpool, kernel1, shape1)
        unpool = argmax_unpool('unpool2', convT, argmax0, FLAGS.batch_size)
        convT = deconv2d('deconv2', unpool, kernel0, shape0)

        tf.image_summary('reconstruction', convT, max_images=5)

        #loss = mean_squared_err('loss', x_image, convT)
        loss = zeiler_loss('zeiler_loss', x_image, convT, reconstruction_weight=1.0, sparsity_weight=0.)
    else:
        raise ValueError, "Invalid model topology, options are \'cae\',\'cnn\'"

    train_op = train(loss)

    # create a saver to export model params
    saver = tf.train.Saver(tf.all_variables())

    # build the summary operation
    summary_op = tf.merge_all_summaries()

    # initialize all variables
    init_op = tf.initialize_all_variables()

    # set global step count
    train_size = mnist.train.images.shape[0]
    total_steps = int(train_size * FLAGS.num_epochs) // FLAGS.batch_size

    with tf.Session() as sess:
        # load model from checkpoint
        if FLAGS.load_checkpoint:
            ckpt = tf.train.get_checkpoint_state(FLAGS.chkdir)
            if ckpt and ckpt.model_checkpoint_path:
                logging.info('Loading model from %s' % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

        if FLAGS.train:
            logging.info('Training for %d epoch(s) [%d minibatches, batch size = %d]'
                         % (FLAGS.num_epochs, total_steps, FLAGS.batch_size))
            train_writer = tf.train.SummaryWriter(FLAGS.logdir + '/train', sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.logdir + '/test')
            sess.run(init_op)
            for step in tqdm(xrange(total_steps)):
                # validate on test set once per epoch
                if step % FLAGS.eval_frequency == 0:
                    feed = feed_dict(x,y,mnist,FLAGS.eval_batch_size,train=False)
                    summary = sess.run(summary_op, feed_dict=feed)
                    test_writer.add_summary(summary, global_step=step)
                else:
                    feed = feed_dict(x,y,mnist,FLAGS.batch_size,train=True)
                    summary, _ = sess.run([summary_op, train_op], feed_dict=feed)
                    train_writer.add_summary(summary, global_step=step)

                if step % FLAGS.checkpoint_freq == 0 or (step + 1) == total_steps:
                    checkpoint_path = os.path.join(FLAGS.chkdir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        if FLAGS.test:
            test_writer = tf.train.SummaryWriter(FLAGS.logdir + '/test')
            sess.run(init_op)
            feed = feed_dict(x,y,mnist,FLAGS.eval_batch_size, train=False)
            summary = sess.run(summary_op, feed_dict=feed)
            test_writer.add_summary(summary, global_step=0)
            '''
            if FLAGS.model_type == 'cnn':
                acc = sess.run(accuracy_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
                logging.info('Evaluating model accuracy on %d test examples' % mnist.test.images.shape[0])
                print('Accuracy: %1.4f' % acc)
            elif FLAGS.model_type == 'cae':
                mse = sess.run(loss, feed_dict={x: mnist.test.images, y: mnist.test.labels})
                logging.info('Evaluating model accuracy on %d test examples' % mnist.test.images.shape[0])
                print("MSE: %1.4f" % mse)
            else:
                raise ValueError, 'Invalid model type'
            '''





