"""A MNIST classifier using the TensorFlow Estimator API."""

# Shows how to read data using tf.data.Dataset or alternative numpy_input_fn()

import types
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import dataset
from utils import delete_dir

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

class EvaluationHook(tf.train.SessionRunHook):
    '''Print evaluation steps, for debugging'''
    def __init__(self):
        self.count = 0
    def after_create_session(self, session, coord):
      print('Evaluation Session created.')
    def before_run(self, run_context):
        self.count += 1
        print('Evaluation step %d' % self.count)
    def end(self, session):
      print('End evaluation Session')    


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
        image = features['x'] # used if input is read from Numpy arrays

    
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = build_model(image, params['hidden_size'], 1.0)
        predictions = { 
            "classes": tf.argmax(logits, axis=1, output_type=tf.int32),
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions)


    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = build_model(image, params['hidden_size'], 0.5)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]) 
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step()) 

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




def main():

    data_dir = '/tmp/mnist'
    model_dir = '/tmp/model'
    batch_size = 128
    train_epochs_before_evals = 1
    use_dataset = True

    delete_dir(model_dir)

    if use_dataset:
        # Use `tf.data.Dataset` to read train and eval data. 
        def train_input_fn():
            ds = dataset.train(data_dir)
            ds = ds.cache()
            ds = ds.shuffle(buffer_size=50000)
            ds = ds.batch(batch_size)
            ds = ds.repeat(train_epochs_before_evals)
            return ds      

        def eval_input_fn():
            ds = dataset.test(data_dir)
            ds = ds.cache().batch(batch_size)
            return ds

    else:
        # Use `numpy_input_fn()` to read train and evaluation data
        # from Numpy arrays.
        mnist = input_data.read_data_sets(data_dir)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": mnist.train.images},
            y=mnist.train.labels.astype(np.int32), 
            num_epochs=1,
            batch_size=batch_size,
            shuffle=True)    
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": mnist.test.images},
            y=mnist.test.labels.astype(np.int32), 
            num_epochs=1,
            batch_size=batch_size,
            shuffle=False)
        
    model_params = {'learning_rate': 1e-4, 'hidden_size': 512}
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params)

    # train model
    estimator.train(input_fn=train_input_fn, hooks=[TrainHook()]) 

    # evaluate model
    eval_results = estimator.evaluate(input_fn=eval_input_fn, hooks=[EvaluationHook()])
    print('Eval loss: %s' % eval_results['loss'])
    print('Eval accuracy: %s' % eval_results['accuracy'])

    # TODO do some prediction

if __name__ == '__main__':
    main()
