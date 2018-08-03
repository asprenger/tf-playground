"""
A MNIST classifier using the TensorFlow low-level API.

Nothing really interesting here. This implentation serves as a baseline to 
compare the other implementations in terms of complexity and functionality.
The low-level API gives you a maximum of flexibility. The disadvantage is of
course that you need to implement all the boiler-plate code youtself.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tempfile
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def build_model(x):

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("model"):

        conv1 = layers.convolution2d(x_image,
                    num_outputs=32,
                    kernel_size=5,
                    stride=1,
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    scope='conv1')

        pool1 = layers.max_pool2d(
            inputs=conv1,
            kernel_size=2,
            stride=2,
            padding='SAME',
            scope='pool1')

        conv2 = layers.convolution2d(pool1,
                    num_outputs=64,
                    kernel_size=5,
                    stride=1,
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    scope='conv2')

        pool2 = layers.max_pool2d(
            inputs=conv2,
            kernel_size=2,
            stride=2,
            padding='SAME',
            scope='pool2')

        flattened = layers.flatten(pool2)

        fc1 = layers.fully_connected(flattened, 
            1024, 
            activation_fn=tf.nn.relu, 
            scope='fc1')

        keep_prob = tf.placeholder(tf.float32)
        drop1 = layers.dropout(
            fc1,
            keep_prob=keep_prob,
            scope='drop1')

        logits = layers.fully_connected(drop1, 
            10, 
            activation_fn=None, 
            scope='fc2')

        return logits, keep_prob


def main(_):

    print('Loading dataset')
    mnist = input_data.read_data_sets(FLAGS.data_dir)
    print('%d train images' % mnist.train.num_examples)
    print('%d test images' % mnist.test.num_examples)

    batch_size = 128
    max_steps = 10000
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int64, [None])
    logits, keep_prob = build_model(x)
    
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(max_steps):
            x_train, y_train = mnist.train.next_batch(batch_size)
            feed_dict = { x: x_train, y: y_train, keep_prob: 0.2}
            sess.run(train_op, feed_dict=feed_dict)

            if i % 100 == 0 and i > 0:
                feed_dict = { x: x_train, y: y_train, keep_prob: 1.0}
                train_loss, train_acc = sess.run([loss, accuracy], feed_dict=feed_dict)
                print('step %d: train_loss=%f train_acc=%g' % (i, train_loss, train_acc))

        feed_dict = { x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0 }
        eval_loss, eval_acc = sess.run([loss, accuracy], feed_dict=feed_dict)
        print('eval_loss=%f eval_acc=%f ' % (i, eval_loss, eval_acc))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
