"""A MNIST classifier using the TensorFlow Estimator API. Parameters, evaluation 
results and model files are logged using MLFLow."""

import os
import argparse
import datetime
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.saved_model import tag_constants
from tensorflow.examples.tutorials.mnist import input_data
import dataset

tf.logging.set_verbosity(tf.logging.INFO)

def current_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def summary_name(name):
    return name.replace(':', '_')

def report_var_grad_distributions(grads_and_vars):
    for grad, var in grads_and_vars:
        if var.name.endswith('weights:0'):
            tf.summary.histogram(summary_name(var.name), var)
            tf.summary.histogram(summary_name(var.name+'_grad'), grad)

def report_var_grad_ratios(grads_and_vars):
    for grad, var in grads_and_vars:
        if var.name.endswith('weights:0'):
            # calculate the mean absolute value for all variables
            mean_abs_var = tf.math.reduce_mean(tf.math.abs(var), axis=None)
            # calculate the mean absolute value for all gradients
            mean_abs_grad = tf.math.reduce_mean(tf.math.abs(grad), axis=None)
            # calculate the ratio between the mean absolute variables and gradients
            var_grad_ratio = mean_abs_var / mean_abs_grad
            tf.summary.scalar('var_grad_ratio/'+summary_name(var.name), mean_abs_var)
            tf.summary.scalar('var_grad_ratio/'+summary_name(var.name)+'_grad', mean_abs_grad)
            tf.summary.scalar('var_grad_ratio/'+summary_name(var.name)+'_grad_ratio', var_grad_ratio)



def report_logits_grad_ratios(grads_and_vars):

    logits_grad = [grad for grad, var in grads_and_vars if var.name == 'model/Logits/weights:0'][0]
    mean_abs_logit_grad = tf.math.reduce_mean(tf.math.abs(logits_grad), axis=None)

    for grad, var in grads_and_vars:
        if var.name.endswith('weights:0'):
            mean_abs_grad = tf.math.reduce_mean(tf.math.abs(grad), axis=None)
            ratio = mean_abs_logit_grad / mean_abs_grad
            tf.summary.scalar('logits_grad_ratio/'+summary_name(var.name), ratio)




def build_model(images, hidden_size, keep_prob):
    """Build the model."""

    with tf.variable_scope("model"):
        net = tf.reshape(images, [-1, 28, 28, 1], name='images')

        net = slim.conv2d(net, 32, 5, stride=1, padding='SAME', normalizer_fn=None, activation_fn=tf.nn.relu, scope='Conv2d_1')
        net = slim.max_pool2d(net, kernel_size=2, stride=2, padding='SAME', scope='Pool_1')

        net = slim.conv2d(net, 64, 5, stride=1, padding='SAME', normalizer_fn=None, activation_fn=tf.nn.relu, scope='Conv2d_2')
        net = slim.max_pool2d(net, kernel_size=2, stride=2, padding='SAME', scope='Pool_2')

        net = slim.flatten(net)
        net = slim.fully_connected(net, hidden_size, activation_fn=tf.nn.relu, scope='Dense_1')
        net = slim.dropout(net, keep_prob=keep_prob, scope='Drop_1')
        logits = slim.fully_connected(net, 10, activation_fn=None,  scope='Logits')

        return logits


def model_fn(features, labels, mode, params, config):
    """Model function that returns EstimatorSpecs."""

    if mode == tf.estimator.ModeKeys.PREDICT:
        images = features['images']
        logits = build_model(images, params['hidden_size'], 1.0)
        predictions = { 
            "class": tf.argmax(logits, axis=1, output_type=tf.int32),
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = build_model(features, params['hidden_size'], params['keep_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]) 
    
        grads_and_vars = optimizer.compute_gradients(loss)

        report_var_grad_ratios(grads_and_vars)
        report_logits_grad_ratios(grads_and_vars)
        #report_var_grad_distributions(grads_and_vars)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = build_model(features, params['hidden_size'], 1.0)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
        eval_metric_ops = { "accuracy": (acc, acc_op) }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metric_ops)


def main(learning_rate, batch_size, checkpoint_base_path, data_path, tracking_url):

    checkpoint_path = os.path.join(checkpoint_base_path, current_timestamp())
    os.makedirs(checkpoint_path, exist_ok=True)

    params = {
        'hidden_size': 512, 
        'keep_rate': 0.8, 
        'learning_rate': learning_rate, 
        'nb_epochs': 5, 
        'batch_size': batch_size,
        'checkpoint_path': checkpoint_path,
        'report_summaries': False
    }

    def train_input_fn():
        ds = dataset.train(data_path)
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.batch(params['batch_size'])
        ds = ds.repeat(params['nb_epochs'])
        return ds      

    def eval_input_fn():
        ds = dataset.test(data_path)
        ds = ds.batch(params['batch_size'])
        return ds
        
    run_config = tf.estimator.RunConfig(log_step_count_steps=10, save_summary_steps=10)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=checkpoint_path, 
                                       params=params, config=run_config)

    estimator.train(input_fn=train_input_fn)
    eval_results = estimator.evaluate(input_fn=eval_input_fn)
    tf.logging.info('Eval loss: %s' % eval_results['loss'])
    tf.logging.info('Eval accuracy: %s' % eval_results['accuracy'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=128)
    parser.add_argument('--data-path', help='Data directory', default='/tmp/mnist')
    parser.add_argument('--checkpoint-path', help='Checkpoint directory', default='/tmp/models')
    parser.add_argument('--tracking-url', help='MLFlow tracking URL', default='file:/tmp/mlruns')

    args = parser.parse_args()
    main(learning_rate=args.learning_rate, batch_size=args.batch_size, 
         data_path=args.data_path, checkpoint_base_path=args.checkpoint_path,
         tracking_url=args.tracking_url)

