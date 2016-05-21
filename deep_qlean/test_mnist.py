from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from test_cnn import MnistConvNet

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('steps', 1000, 'Number of training steps')
flags.DEFINE_string('mnist_dir', 'MNIST_data', 'Directory for storing data')
flags.DEFINE_string('train_dir', 'mnist_train', 'Directory for event logs and checkpoints')

def train():
    mnist = input_data.read_data_sets(FLAGS.mnist_dir, one_hot=True)

    # initialize the model
    model = MnistConvNet()

    # build a subgraph for preprocessing the input
    x, y = model.preprocess()

    # computes the predicitons of the inference model
    softmax = model.inference(x)

    # calculate loss
    loss = model.loss(softmax,y)

    # subgraph for parameter updates given a batch of parameters
    train_op = model.train(loss)

    # init a saver
    #saver = tf.train.Saver(tf.all_variables())

    # build the summary operation
    summary_op = tf.merge_all_summaries()

    # initialize all variables
    init_op = tf.initialize_all_variables()

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # start running operations on the graph
    with tf.Session() as sess:
        sess.run(init_op)

        for step in tqdm(xrange(FLAGS.steps)):
            batch = mnist.train.next_batch(100)
            _, loss_value= sess.run([train_op, loss], feed_dict={x: batch[0], y:batch[1]})

            if step % 100 == 0:
                summary = sess.run(summary_op, feed_dict={x:batch[0]})
                summary_writer.add_summary(summary, global_step=step)




if __name__=='__main__':
    train()
