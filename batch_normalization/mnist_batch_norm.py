"""
A MNIST classifier using batch normalization.
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

def dense(x, size, scope):
    with tf.variable_scope(scope):
        return layers.fully_connected(x, size, activation_fn=None,
                                      scope='fc')

def dense_batch_relu(x, size, is_training, scope):
    with tf.variable_scope(scope):
        fc = layers.fully_connected(x, size, activation_fn=None,
                                    scope='fc')
        bn = layers.batch_norm(fc, center=True, scale=True, 
                               is_training=is_training, scope='bn')
        return tf.nn.relu(bn, name='relu')

def build_model(x, is_training):
    h1 = dense_batch_relu(x, 256, is_training,'layer1')
    h2 = dense_batch_relu(h1, 256, is_training, 'layer2')
    logits = dense(h2, 10, 'logits')
    return logits

def main(_):

    print('Loading dataset')
    mnist = input_data.read_data_sets(FLAGS.data_dir)
    print('%d train images' % mnist.train.num_examples)
    print('%d test images' % mnist.test.num_examples)

    batch_size = 128
    max_steps = 10000
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool, name='is_training')

    logits = build_model(x, is_training)
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

    # The batch_norm layer keeps track of input statistics in variables `moving_mean` 
    # and `moving_variance`. This variables must be updates before every train step.
    # Unfortunately, the update_ops are not parents of train_op in the computational 
    # graph. Therefore we must define the dependency explicitly. By default the 
    # update_ops are placed in tf.GraphKeys.UPDATE_OPS.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        #train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(max_steps):
            x_train, y_train = mnist.train.next_batch(batch_size)
            feed_dict = { x: x_train, y: y_train, is_training: True }
            sess.run(train_op, feed_dict=feed_dict)

            if i % 100 == 0 and i > 0:
                feed_dict = { x: mnist.test.images, y: mnist.test.labels, is_training: False }
                eval_loss, eval_acc = sess.run([loss, accuracy], feed_dict=feed_dict)
                print('step %d: eval_loss=%f eval_acc=%f ' % (i, eval_loss, eval_acc))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

