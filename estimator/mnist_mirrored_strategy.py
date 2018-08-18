"""
A MNIST classifier using the tf.contrib.distribute.MirroredStrategy with Estimators.

The MirroredStrategy does in-graph replication with synchronous training on many GPUs 
on a single machine.

The strategy create copies of all model variables on each device. It then uses all-reduce 
to combine gradients across the devices before applying them to the variables to keep them 
in sync. 
"""

import types
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.client import device_lib
import dataset
from utils import delete_dir

# Enable logging so that the output from tf.train.LoggingTensorHook
# is printed on the console
tf.logging.set_verbosity(tf.logging.INFO)

def build_model(x, hidden_size, keep_prob):
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

def model_fn(features, labels, mode, params, config):
    '''Model function for Estimator.'''

    image = features
    
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
        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = build_model(image, params['hidden_size'], 1.0)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
        eval_metric_ops = { "accuracy": (acc, acc_op) }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def distribution_strategy():
    num_gpus = len(get_available_gpus())
    tf.logging.info('Number of GPUs: %d' % num_gpus)
    if num_gpus == 0:
        return tf.contrib.distribute.OneDeviceStrategy(device='/cpu:0')
    elif num_gpus == 1:
        tf.logging.info('Use OneDeviceStrategy')
        return tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    else:
        tf.logging.info('Use MirroredStrategy')
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)

def main():

    data_dir = '/tmp/mnist'
    model_dir = '/tmp/model'
    batch_size = 128
    use_dataset = True

    delete_dir(model_dir)

    # Use `tf.data.Dataset` to read train and eval data. 
    def train_input_fn():
        ds = dataset.train(data_dir)
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.batch(batch_size)
        ds = ds.repeat(10)
        return ds      

    def eval_input_fn():
        ds = dataset.test(data_dir)
        ds = ds.batch(batch_size)
        return ds

    run_config = tf.estimator.RunConfig(train_distribute=distribution_strategy())

    model_params = {'learning_rate': 1e-4, 'hidden_size': 512, 'keep_rate': 0.5}
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params, config=run_config)

    print('Train model')
    estimator.train(input_fn=train_input_fn) 

    print('Evaluate model')
    eval_results = estimator.evaluate(input_fn=eval_input_fn)
    print('Eval loss: %s' % eval_results['loss'])
    print('Eval accuracy: %s' % eval_results['accuracy'])

    print('Generate some predictions:')
    preds = estimator.predict(input_fn=eval_input_fn)
    for _ in range(5):
        print(preds.__next__()['class'])


if __name__ == '__main__':
    main()
