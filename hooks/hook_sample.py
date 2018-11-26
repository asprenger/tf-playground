
import types
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import dataset
from utils import delete_dir
from tensorflow.python.training.summary_io import SummaryWriterCache

tf.logging.set_verbosity(tf.logging.INFO)

# SessionRunHook: 
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/training/session_run_hook.py

# LoggingTensorHook:
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/training/basic_session_run_hooks.py

class SampleHook(tf.train.SessionRunHook):

    def __init__(self, loss_tensor, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.loss_tensor = loss_tensor

    def begin(self):
        """Called once before finalizing the graph."""
        self.global_step = tf.train.get_or_create_global_step()
        self.loss_tensor = self._as_tensor(self.loss_tensor)
        self.mean_loss, self.mean_update_op = tf.metrics.mean(self.loss_tensor)

        # Create histogram summary ops for all trainable variable
        weight_histograms = []
        for var in tf.trainable_variables():
            weight_histograms.append(tf.summary.histogram(self._summary_name('Weights/'+var.name), var, collections=[]))
        self.summaries = tf.summary.merge(weight_histograms, collections=[])

    def _summary_name(self, name):
        return name.replace(':', '_')
        
    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created"""
        pass

    def before_run(self, run_context):
        """Called before each call to run()"""
        # Make sure to execute metric update ops here!
        return tf.train.SessionRunArgs([self.global_step, self.loss_tensor, self.mean_update_op])
        
    def after_run(self, run_context, run_values):
        """Called after each call to run()"""
        global_step, loss, mean_loss = run_values.results
        # do something cool with the values!

    def end(self, session):
        """Called at the end of session"""
        global_step, mean_loss, summaries = session.run([self.global_step, self.mean_loss, self.summaries])
        self._write_summary(summaries, global_step, self.checkpoint_path)

        # This is an example how to manually create a Summary
        summary = tf.Summary()
        summary.value.add(tag='custom_loss', simple_value=mean_loss)
        self._write_summary(summary, global_step, self.checkpoint_path)
        

    def _write_summary(self, summary, global_step, checkpoint_path):
        summary_writer = SummaryWriterCache.get(checkpoint_path)
        summary_writer.add_summary(summary, global_step)
        summary_writer.flush()

    def _as_tensor(self, name_or_tensor):
        """Retrieve a tensor from the default graph."""
        graph = tf.get_default_graph()
        if not isinstance(name_or_tensor, str):
            if not hasattr(name_or_tensor, "graph") or name_or_tensor.graph != graph:
                raise ValueError("Passed %s should have graph attribute that is equal "
                                 "to current graph %s." % (name_or_tensor, graph))
            return name_or_tensor
        if ":" in name_or_tensor:
            element = graph.as_graph_element(name_or_tensor)
        else:
            element = graph.as_graph_element(name_or_tensor + ":0")
            # Check that there is no :1 (e.g. it's single output).
            try:
                graph.as_graph_element(name_or_tensor + ":1")
            except (KeyError, ValueError):
                pass
            else:
                raise ValueError("Name %s is ambiguous, "
                                 "as this `Operation` has multiple outputs "
                                 "(at least 2)." % name_or_tensor)
        return element






def build_model(x, hidden_size, keep_prob):
    with tf.variable_scope("model"):
        fc1 = layers.fully_connected(x, 
            512, 
            activation_fn=tf.nn.relu, 
            scope='fc1')
        logits = layers.fully_connected(fc1, 
            10, 
            activation_fn=None, 
            scope='fc2')
        return logits

def model_fn(features, labels, mode, params, config):
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
        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

        # Name tensor that will be referenced from the sample hook
        tf.identity(loss, 'cross_entropy')

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


def main():

    data_dir = '/tmp/mnist'
    model_dir = '/tmp/model'
    batch_size = 128
    use_dataset = True

    delete_dir(model_dir)

    def train_input_fn():
        ds = dataset.train(data_dir)
        ds = ds.repeat(None)
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.batch(batch_size)
        return ds      

    def eval_input_fn():
        ds = dataset.test(data_dir)
        ds = ds.batch(batch_size)
        return ds
        
    model_params = {'learning_rate': 1e-4, 'hidden_size': 512, 'keep_rate': 0.5}
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params)

    for iter in range(100):

        print('Train model')
        train_hooks = [
            tf.train.LoggingTensorHook(tensors=['global_step', 'cross_entropy'], every_n_iter=1),
            SampleHook(loss_tensor='cross_entropy', checkpoint_path=model_dir)
            ]
        estimator.train(input_fn=train_input_fn, steps=200, hooks=train_hooks) 

        print('Evaluate model')
        eval_hooks = [tf.train.LoggingTensorHook(tensors=['global_step'], every_n_iter=1)]
        eval_results = estimator.evaluate(input_fn=eval_input_fn, hooks=eval_hooks)
        print('Eval loss: %s' % eval_results['loss'])
        print('Eval accuracy: %s' % eval_results['accuracy'])



if __name__ == '__main__':
    main()