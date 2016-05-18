import tensorflow as tf


class MnistConvNet:
    def __init__(self):
        pass

    def _activation_summary(x):
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

    def _variable_on_cpu(self, name, shape, stddev):
        """Helper to create a Variable stored on CPU memory.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of truncated normal initializer
        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            initializer = tf.truncated_normal_initializer(stddev=stddev)
            var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
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
            kernel = self._variable_on_cpu('weights', shape=[5,5,1,32],
                                           stddev=1e-4)
            conv = tf.nn.conv2d(self.x, kernel, [1,1,1,1], padding='SAME')
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv1)


    def loss(self):
        pass
    def train(self):
        pass
