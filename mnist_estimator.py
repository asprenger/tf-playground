'''A MNIST classifier using the TensorFlow Estimator API.'''

# This implementation creates a `tf.estimator.Estimator` and calls the `train()`, 
# `evaluate()` and `predict()` methods directly.

# A `tf.estimator.inputs.numpy_input_fn` is used to create an input function that feeds 
# numpy arrays into the model. The alternative would be to use the Dataset API.

# Note: One unexpected problem has been setting the dropout rate that must be different
# for training and evaluation/prediction. In the end the problem could be solved by adding
# a SessionRunHook to the training phase that adds the dropout rate to the feed_dict. This
# is pretty cumbersome and there should be a simpler solution. Maybe a tf.case() with the
# train mode in the predicate!?

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

class OverwriteKeepProbabilityHook(tf.train.SessionRunHook):
    '''Overwrite dropout layer 'keep_prob' property'''

    # For details about SessionRunHooks look at the source:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/session_run_hook.py

    def __init__(self, keep_prob, tensor_name='model/keep_prob'):
        self.keep_prob = keep_prob
        self.tensor_name = tensor_name

    def before_run(self, run_context):
        # overwrite the 'keep_prob' property in the feed_dict
        placeholder = tf.get_default_graph().get_tensor_by_name(self.tensor_name + ':0')
        feed_dict = run_context.original_args.feed_dict
        if feed_dict:
            feed_dict[placeholder] = self.keep_prob
        else:
            feed_dict = {placeholder: self.keep_prob}
        fetches = run_context.original_args.fetches
        return tf.train.SessionRunArgs(fetches, feed_dict=feed_dict)


def get_all_variables_with_name(var_name):
    name = var_name + ':0'
    return [var for var in tf.all_variables() if var.name.endswith(name)]

def network(x):

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("model"):

        # The default dropout layer 'keep_prob' is 1.0, this must be used in
        # EVAL/PREDICT mode. Use the OverwriteKeepProbabilityHook to overwrite
        # the value for training.
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
            keep_prob=keep_prob,
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

    # must use TFs metrics instead of implement this yourself
    acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
    eval_metric_ops = { "acc": (acc, acc_op) }

    # EstimatorSpec for ModeKeys.EVAL and ModeKeys.TRAIN
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
    
    batch_size = 64
    train_steps = 100 # steps equals number of batches 

    # Note: The current implementation terminates training after 'train_steps' batches.
    # An alternative would be to set 'model.train(steps=None)' and specify the number of
    # epochs in 'train_input_fn'. There should also be a hook that terminates training
    # when the loss does not change any more.

    # Use numpy_input_fn() to create functions that feed numpy arrays into the model. 
    # The obvious disadvantage is that the whole dataset must fit into memory.

    # labels must be cast because sparse_softmax_cross_entropy() requires int32
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": mnist.train.images},
        y=mnist.train.labels.astype(np.int32), 
        num_epochs=None, # cycle forever over the examples
        batch_size=batch_size,
        shuffle=True)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": mnist.test.images},
        y=mnist.test.labels.astype(np.int32), 
        num_epochs=1,
        batch_size=batch_size,
        shuffle=False)

    predict_labels = mnist.test.labels.astype(np.int32)[0:10]
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": mnist.test.images[0:10]},
        y=predict_labels, 
        num_epochs=1,
        shuffle=False)


    model_params = {"learning_rate": 1e-4}
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

    # train model
    train_hooks = [OverwriteKeepProbabilityHook(keep_prob=0.2)]
    estimator.train(input_fn=train_input_fn, hooks=train_hooks, steps=train_steps) 

    # evaluate model
    ev = estimator.evaluate(input_fn=test_input_fn)
    print("Loss: %s" % ev["loss"])
    print("Test accuracy: %s" % ev["acc"])

    # use model to predict
    print('Some prediction results:')
    predictions = estimator.predict(input_fn=predict_input_fn)
    for i, pred in enumerate(predictions):
        print(i, pred['pred'], predict_labels[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    