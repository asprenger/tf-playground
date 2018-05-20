import random
import copy
import numpy as np
import keras
from keras.datasets import mnist
from sklearn.grid_search import RandomizedSearchCV
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from utils import has_arg, ts_rand, dict_to_str
import tensorflow as tf
import tensorflow.contrib.layers as layers



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



def build_estimator(hidden_size=128, learning_rate=1e-4):
    print('Estimator(hidden_size=%d)' % hidden_size)
    model_params = {
        'hidden_size':hidden_size,
        'learning_rate': learning_rate
    }
    return tf.estimator.Estimator(model_fn=build_model, params=model_params)
    







class MyWrapper(object):
    '''Implementation of the scikit-learn classifier API for a TensorFlow Estimator.'''

    def __init__(self, build_fn, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.estimator = None
        # TODO self.check_params(sk_params)

    def get_params(self, **params):
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        # TODO self.check_params(params)
        self.sk_params.update(params)
        return self

    def fit(self, X_train, y_train):
        print('\nfit(X_train=%s, y_train=%s, sk_params=%s)' % (str(X_train.shape), str(y_train.shape), str(self.sk_params)))
        self.estimator = self.build_fn(**self.filter_sk_params(self.build_fn))

        batch_size = self.sk_params['batch_size']
        train_steps = 100

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train,
            num_epochs=None, # cycle forever over the examples
            batch_size=batch_size,
            shuffle=True)
        
        self.estimator.train(input_fn=train_input_fn, steps=train_steps)  
        # other parameter: hooks, max_steps, saving_listeners



    def score(self, X_test, y_test):
        # called if 'RandomizedSearchCV(scoring=None)'
        print('score(x_test=%s, y_test=%s, sk_params=%s)' % (str(X_test.shape), str(y_test.shape), str(self.sk_params)))

        if not self.estimator:
            raise ValueError('First call fit() to train an estimator.')

        batch_size = self.sk_params['batch_size']
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_test},
            y=y_test, 
            num_epochs=1,
            batch_size=batch_size,
            shuffle=False)

        ev = self.estimator.evaluate(input_fn=test_input_fn)
        # other parameters: steps. hooks, checkpoint_path, name

        score = ev['accuracy']
        print('Eval accuracy: %f' % score)
        return score

    def predict(self, X):
        print('predict(x=%s, sk_params=%s)' % (str(X.shape), str(self.sk_params)))

        if not self.estimator:
            raise ValueError('First call fit() to train an estimator.')

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            num_epochs=1,
            shuffle=False)

        predictions = self.estimator.predict(input_fn=predict_input_fn)
        return np.array([p['pred'] for p in predictions])

    def filter_sk_params(self, fn):
        """Filters `sk_params` and returns those in `fn`'s arguments.
        # Arguments
            fn : arbitrary function
        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        res = {}
        for name, value in self.sk_params.items():
            if has_arg(fn, name):
                res.update({name: value})
        return res










def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    y_train = y_train.astype('int32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('int32')
    x_train /= 255
    x_test /= 255

    param_distributions = {
        'hidden_size': [128],
        'batch_size': [64],
        'learning_rate': [1e-2, 1e-3, 1e-4],
    }
    sampling_iterations = 3

    wrapper = MyWrapper(build_fn=build_estimator, hidden_size=None)

    # Note: RandomizedSearchCV splits up the train data according to a cross-validation 
    # strategy specified by the 'cv' parameter. The final evaluation is performed on the
    # test data. Sklearn implements several cross validation generators:
    # http://scikit-learn.org/stable/modules/cross_validation.html

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
