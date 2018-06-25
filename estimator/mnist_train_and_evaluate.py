"""
A MNIST classifier using the tf.estimator.train_and_evaluate() function

The function `tf.estimator.train_and_evaluate` trains and evaluates a model by 
using a given Estimator. It provides consistent behavior for both local and 
distributed training.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import shutil
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
from utils import delete_dir
import dataset

FLAGS = None

class TrainHook(tf.train.SessionRunHook):
    '''Print training steps, for debugging'''
    def __init__(self):
        self.count = 0
    def after_create_session(self, session, coord):
      print('Train Session created.')
    def before_run(self, run_context):
        self.count += 1
        print('Train step %d' % self.count)
    def end(self, session):
      print('End train Session')    

class EvalHook(tf.train.SessionRunHook):
    '''Print eval steps, for debugging'''
    def __init__(self):
        self.count = 0
    def after_create_session(self, session, coord):
      print('Eval Session created.')    
    def before_run(self, run_context):
        self.count += 1
        print('Eval step %d' % self.count)
    def end(self, session):
      print('End eval Session')    

    
class MyExporter(tf.estimator.Exporter):
    def __init__(self):
        self._name = 'foo'
    @property
    def name(self):
        return self._name
    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        print('EVAL RESULT', eval_result)

def build_model(x, hidden_size, keep_prob):
    print('BUILD MODEL(x=%s, hidden_size=%d, keep_prob=%f)' % (x.shape, hidden_size, keep_prob))
    with tf.variable_scope("model"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
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
            hidden_size, 
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
        return logits


def model_fn(features, labels, mode, params):
    '''Model function for Estimator.'''

    image = features
    if isinstance(features, dict):
        image = features['X'] # used if input is read from Numpy arrays
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = build_model(image, params['hidden_size'], 1.0)
        predictions = { 
            "class": tf.argmax(logits, axis=1, output_type=tf.int32),
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = build_model(image, params['hidden_size'], params['keep_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]) 
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step()) 

        accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(params['learning_rate'], 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = build_model(image, params['hidden_size'], 1.0)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1, output_type=tf.int32))
        eval_metric_ops = { "accuracy": (acc, acc_op) }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

def main(_):

    data_dir = '/tmp/mnist'
    model_dir = '/tmp/model'
    batch_size = 128
    train_epochs_before_evals = 1
    train_max_steps = 500

    delete_dir(FLAGS.model_dir)

    def train_input_fn():
        ds = dataset.train(data_dir)
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.batch(batch_size)
        ds = ds.repeat(train_epochs_before_evals)
        return ds      

    def eval_input_fn():
        ds = dataset.test(data_dir)
        ds = ds.batch(batch_size)
        return ds

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=train_max_steps,
        hooks=[TrainHook()]
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None, # evaluates until input_fn raises an EOF exception
        exporters=[MyExporter()],
        hooks=[EvalHook()]
    )
    
    model_params = {'learning_rate': 1e-4, 'hidden_size': 512, 'keep_rate': 0.5}
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('--model_dir', type=str,
                  default='/tmp/tensorflow/mnist/model',
                  help='Directory for storing model checkpoints')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
