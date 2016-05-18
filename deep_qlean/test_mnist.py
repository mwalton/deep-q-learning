from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from test_cnn import MnistConvNet

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mnist_dir', 'MNIST_data', 'Directory for storing data')
flags.DEFINE_string('train_dir', 'mnist_train', 'Directory for event logs and checkpoints')

def train():
    mnist = input_data.read_data_sets(FLAGS.mnist_dir, one_hot=True)

    # initialize the model
    model = MnistConvNet()

    # get images and labels
    x, y = model.input()

    # init a saver
    #saver = tf.train.Saver(tf.all_variables())

    # build the summary operation
    summary_op = tf.merge_all_summaries()

    # initialize all variables
    init = tf.initialize_all_variables()

    # start running operations on the graph
    with tf.Session() as sess:
        sess.run(init)

        batch = mnist.train.next_batch(100)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        summary = sess.run(summary_op, feed_dict={x:batch[0]})
        summary_writer.add_summary(summary, global_step=0)




if __name__=='__main__':
    train()
