import tensorflow as tf


class MnistConvNet:
    def __init__(self):
        pass

    def _activation_summary(self,x):
      """Helper to create summaries for activations.
      Creates a summary that provides a histogram of activations.
      Creates a summary that measure the sparsity of activations.
      Args:
        x: Tensor
      Returns:
        nothing
      """
      # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
      # session. This helps the clarity of presentation on tensorboard.
      tensor_name = x.op.name
      tf.histogram_summary(tensor_name + '/activations', x)
      tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _random_init(self, name, shape, stddev):
        initializer = tf.truncated_normal_initializer(stddev=stddev)
        return self._variable_on_cpu(name, shape, initializer)

    def _const_init(self, name, shape, value):
        initializer = tf.constant_initializer(value)
        return self._variable_on_cpu(name, shape, initializer)


    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of truncated normal initializer
        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def preprocess(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
            tf.image_summary('input', image_shaped_input, max_images=100)
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
            return (x, y_)

    def inference(self, images):
        with tf.name_scope('conv1') as scope:
            kernel = self._random_init('weights', shape=[5,5,1,32],
                                           stddev=1e-4)
            conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
            biases = self._const_init('biases', [32], 0.0)
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv1)

        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME', name='pool1')

        with tf.name_scope('conv2') as scope:
            kernel = self._random_init('weights', shape=[5,5,32,64],
                                           stddev=1e-4)
            conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
            biases = self._const_init('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv2)

        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],
                               padding='SAME', name='pool1')

        with tf.name_scope('flatten') as scope:
            # shape of -1 flattens to 1-D, may need to change None to explicit batch_size
            reshape = tf.reshape(pool2, [None, -1])
            dim = reshape.get_shape()[1].value
            weights = self._random_init('weights', shape=[dim, 1024], stddev=0.1)
            biases = self._const_init('biases', [1024], 0.1)
            flatten = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            self._activation_summary(flatten)

        with tf.name_scope('hidden') as scope:
            weights = self._random_init('weights', shape=[1024, 256], stddev=0.1)
            biases = self._const_init('biases', [256], 0.1)
            hidden = tf.nn.relu(tf.matmul(flatten, weights) + biases, name=scope.name)
            self._activation_summary(hidden)

        with tf.variable_scope('softmax') as scope:
            weights = self._random_init('weights', [256, 10], stddev=0.1) # 10 classes
            biases = self._const_init('biases', [10], 0.0)
            softmax = tf.add(tf.matmul(hidden, weights), biases, name=scope.name)
            self._activation_summary(softmax)

        return softmax


    def loss(self, softmax, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(softmax,
                                                                       labels,
                                                                       name='xentropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('loss', cross_entropy_mean)

        return tf.add_n(tf.get_collection('loss'), name='total_loss')

    def _loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.
          Generates moving average for all losses and associated summaries for
          visualizing the performance of the network.
          Args:
            total_loss: Total loss from loss().
          Returns:
            loss_averages_op: op for generating moving averages of losses.
          """
        # Compute the moving average of all individual losses and the total loss.
        loss_avg = tf.train.ExponentialMovingAverage(.9, name='loss_avg')
        losses = tf.get_collection('losses')
        loss_avg_op = loss_avg.apply(losses + [total_loss]) # append the total loss

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # name each loss '(raw)' and give original name to averaged version
            tf.scalar_summary(l.op.name + ' (raw)', l)
            tf.scalar_summary(l.op.name, loss_avg.average(l))

        return loss_avg_op


    def train(self, total_loss):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
        return train_step
