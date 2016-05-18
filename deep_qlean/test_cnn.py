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

    def input(self):
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
            conv = tf.nn.conv2d(self.x, kernel, [1,1,1,1], padding='SAME')
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

    def loss(self):
        pass
    def train(self):
        pass
