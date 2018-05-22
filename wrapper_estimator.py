"""
A prototype for hyperparameter search with TensorFlow.
 * Should support RandomizedSearchCV and GridSearchCV
 * Should support numpy_input_fn() and Datasets
 * Non-distributed training only
"""

import os
import uuid
import shutil
import copy
import numpy as np
from sklearn.grid_search import RandomizedSearchCV
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from utils import has_arg, ts_rand, dict_to_str
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data



def build_network(x, hidden_size):
    print('Model(x=%s, hidden_size=%d)' % (x.shape, hidden_size))
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
            keep_prob=1.0,
            scope='drop1')
        logits = layers.fully_connected(drop1, 
            10, 
            activation_fn=None, 
            scope='fc2')
        return logits


def build_model(features, labels, mode, params):
    '''Model function for the Estimator'''

    logits = build_network(features["x"], params['hidden_size'])
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"pred": predictions})

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]) 
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step()) 

    acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
    eval_metric_ops = { "accuracy": (acc, acc_op) }

    # EstimatorSpec for ModeKeys.EVAL and ModeKeys.TRAIN
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


class MyWrapper(object):
    '''Implementation of the scikit-learn classifier API for a TensorFlow Estimator.'''

    def __init__(self, build_fn, eval_metric_name='accuracy', train_epochs=1, model_base_dir=None, train_hooks=None, eval_hooks=None, predict_hooks=None, **params):
        self.build_fn = build_fn
        self.eval_metric_name = eval_metric_name
        self.train_epochs = train_epochs
        self.model_base_dir = model_base_dir
        self.train_hooks = train_hooks
        self.eval_hooks = eval_hooks
        self.predict_hooks = predict_hooks
        self.params = params
        self.estimator = None

    def get_params(self, **params):
        result = copy.deepcopy(self.params)
        result.update({'build_fn': self.build_fn})
        result.update({'eval_metric_name': self.eval_metric_name})
        result.update({'train_epochs': self.train_epochs})
        result.update({'model_base_dir': self.model_base_dir})
        result.update({'train_hooks': self.train_hooks})
        result.update({'eval_hooks': self.eval_hooks})
        result.update({'predict_hooks': self.predict_hooks})
        return result

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y):
        print('\nfit(X=%s, y=%s, params=%s)' % (str(X.shape), str(y.shape), str(self.params)))

        assert 'batch_size' in self.params, "Parameters must contain a 'batch_size'"
        batch_size = self.params['batch_size']

        model_dir = None
        if self.model_base_dir:
            model_dir = os.path.join(self.model_base_dir, str(uuid.uuid1()))

        print('Estimator(model_dir=%s, params=%s)' % (model_dir, str(self.params)))
        self.estimator = tf.estimator.Estimator(model_fn=self.build_fn, model_dir=model_dir, params=self.params)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            y=y,
            num_epochs=self.train_epochs,
            batch_size=batch_size,
            shuffle=True)
        
        # TODO remove 'steps=100' after development done
        self.estimator.train(input_fn=train_input_fn, steps=100, hooks=self.train_hooks)
        


    def score(self, X, y):
        print('score(X=%s, y=%s, params=%s)' % (str(X.shape), str(y.shape), str(self.params)))

        if not self.estimator:
            raise ValueError('First call fit() to train a model.')

        assert 'batch_size' in self.params, "Parameters must contain a 'batch_size'"
        batch_size = self.params['batch_size'] 

        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            y=y, 
            num_epochs=1,
            batch_size=batch_size,
            shuffle=False)

        metrics = self.estimator.evaluate(input_fn=test_input_fn, hooks=self.eval_hooks)
        score = metrics[self.eval_metric_name]
        print('Eval accuracy: %f' % score)
        return score


    def predict(self, X):
        print('predict(X=%s, params=%s)' % (str(X.shape), str(self.params)))

        if not self.estimator:
            raise ValueError('First call fit() to train a model.')

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            num_epochs=1,
            shuffle=False)

        predictions = self.estimator.predict(input_fn=predict_input_fn, hooks=self.predict_hooks)
        return np.array([p['pred'] for p in predictions]) # TODO configure


def main():

    model_base_dir = '/tmp/mnist_model'
    shutil.rmtree(model_base_dir, ignore_errors=True)

    mnist = input_data.read_data_sets('/tmp/mnist_model/mnist')
    x_train = mnist.train.images
    y_train = mnist.train.labels.astype('int32')
    x_test = mnist.test.images
    y_test = mnist.test.labels.astype('int32')

    # Single parameters that are not part of the parameter search can be passed added
    # to 'param_distributions' or passed as named parameters to the wrapper.

    param_distributions = {
        'hidden_size': [128],
        'batch_size': [64],
        'learning_rate': [1e-2, 1e-3, 1e-4],
    }
    sampling_iterations = 3 # TODO

    wrapper = MyWrapper(build_fn=build_model, model_base_dir=model_base_dir)

    # Note: RandomizedSearchCV splits up the train data according to a cross-validation 
    # strategy specified by the 'cv' parameter. The final evaluation is performed on the
    # test data. Sklearn implements several cross validation generators:
    # http://scikit-learn.org/stable/modules/cross_validation.html

    # Note: If the RandomizedSearchCV.scoring property is set the cross-validation strategy
    # calls Estimator.predict with the validation data set and calculates the eval score with
    # the specified scoring strategy. Otherwise the Estimator must implement the evaluation 
    # scoring and the cross-validation strategy calls Estimator.score().

    validator = RandomizedSearchCV(wrapper,
                             param_distributions=param_distributions,
                             n_iter = sampling_iterations,
                             verbose=0,
                             cv=[[np.arange(20000), np.arange(10000)]],
                             refit=True,
                             #scoring='accuracy', 
                             n_jobs=1)

    validator.fit(x_train, y_train)

    print('\nBest params: %s' % str(validator.best_params_))
    print('Best eval score: %f' % validator.best_score_)

    if hasattr(validator, 'best_estimator_'):
        # attribute exists if the cross-validation strategy has refitted the model
        best_estimator = validator.best_estimator_.estimator
        os.symlink(best_estimator.model_dir, os.path.join(model_base_dir, 'best_model'))
        print('Final model path: %s' % best_estimator.model_dir)

        # Evaluate the best model on the test dataset
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": x_test},
                y=y_test, 
                num_epochs=1,
                shuffle=False)
        ev = best_estimator.evaluate(input_fn=test_input_fn)
        print("Test loss: %s" % ev["loss"])
        print("Test accuracy: %s" % ev["accuracy"])

        # Use the best model to do some predictions
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_test[0:10]},
            num_epochs=1,
            shuffle=False)
        preds = best_estimator.predict(input_fn=predict_input_fn)
        preds = np.array([p['pred'] for p in preds])
        print(preds)
        print(y_test[0:10])


if __name__ == '__main__':
    main()
