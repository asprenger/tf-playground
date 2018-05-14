
'''A MNIST classifier that calls tf.estimator.train_and_evaluate()'''

# This implementation creates a `tf.estimator.Estimator`, train_spec and eval_spec and 
# than calls `tf.estimator.train_and_evaluate()` that trains and evaluates the model.
# This utility function supports local and distributed configurations. For further infos
# see: https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate

# A `tf.estimator.inputs.numpy_input_fn` is used to create an input function that feeds 
# numpy arrays into the model. The alternative would be to use the Dataset API.

# The function `tf.estimator.train_and_evaluate()` is implemented in tensorflow/python/estimator/training.py
# It delegates all interesting work to `_TrainingExecutor.run()`. In the non-distributed case this delegates
# to _TrainingExecutor.run_local(). 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

class TrainHook(tf.train.SessionRunHook):
    '''Print training steps, for debugging'''
    def __init__(self):
        self.count = 0
    def before_run(self, run_context):
        self.count += 1
        print('Train step %d' % self.count)

class EvalHook(tf.train.SessionRunHook):
    '''Print eval steps, for debugging'''
    def __init__(self):
        self.count = 0
    def before_run(self, run_context):
        self.count += 1
        print('Eval step %d' % self.count)
    

def network(x):

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("model"):

        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

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

        drop1 = layers.dropout(
            fc1,
            #keep_prob=keep_prob,
            keep_prob=1.0, # TODO
            scope='drop1')

        logits = layers.fully_connected(drop1, 
            10, 
            activation_fn=None, 
            scope='fc2')

        return logits, keep_prob


def model_fn(features, labels, mode, params):
    '''Model function for Estimator.'''

    logits, keep_prob = network(features["x"]) 

    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"pred": predictions})

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]) 
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step()) 

    acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions, name='acc')
    eval_metric_ops = { "acc": (acc, acc_op) }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def main(_):

    print('Loading dataset')
    mnist = input_data.read_data_sets(FLAGS.data_dir)
    print('%d train images' % mnist.train.num_examples)
    print('%d test images' % mnist.test.num_examples)
    
    epochs = 5
    batch_size = 10
    
    train_max_steps = (100 // batch_size) * epochs
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": mnist.train.images[0:100]},
        y=mnist.train.labels[0:100].astype(np.int32), 
        num_epochs=None, # cycle forever over the examples
        batch_size=batch_size,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": mnist.test.images[0:20]},
        y=mnist.test.labels[0:20].astype(np.int32), 
        num_epochs=1,
        batch_size=batch_size,
        shuffle=False)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=train_max_steps,
        hooks=[TrainHook()]
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None, # evaluates until input_fn raises an EOF exception
        hooks=[EvalHook()]
    )

    # TODO: need better understanding how the train_and_evaluate() workflow works.
    # When numpy_input_fn.num_epochs is set to a number, training terminates after 
    # the given number of epochs but the loop does not terminate and the model
    # is evaluated multiple times.

    # Make sure to set `eval_input_fn.num_batch=1` and `eval_spec.steps=None`. Evaluation stops
    # on EOF and this setting makes sure there is exactly one iteration over the eval data.
    
    model_params = {"learning_rate": 1e-4}
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
