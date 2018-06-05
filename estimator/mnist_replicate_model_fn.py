"""A MNIST classifier that supports multiple GPUs."""

# uses replicate_model_fn() from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/estimator/python/estimator/replicate_model_fn.py

# There are two steps required if using multi-GPU: (1) wrap the model_fn,
# and (2) wrap the optimizer. 

# replicate_model_fn() is depricated since 2018-05-31, use tf.contrib.distribute.MirroredStrategy
# see: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute

import numpy as np
import tensorflow as tf
import dataset
from utils import delete_dir

tf.logging.set_verbosity(tf.logging.INFO)


def validate_batch_size_for_multi_gpu(batch_size):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.
  Note that this should eventually be handled by replicate_model_fn
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.
  Args:
    batch_size: the number of examples processed in each training batch.
  Raises:
    ValueError: if no GPUs are found, or selected batch_size is invalid.
  """
  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top

  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. '
           'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)


def build_model(data_format):
  """Build the model.
  Args:
    data_format: 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. 
  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  layers = tf.keras.layers

  inputs = layers.Input((784,))
  x = layers.Reshape(input_shape)(inputs)
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(512, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  logits = layers.Dense(10, activation=None, name='logits')(x)

  return tf.keras.models.Model(inputs=inputs, outputs=logits)

  


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  model = build_model(params['data_format'])
  image = features
  if isinstance(image, dict):
    image = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])


    # MULTI_GPU
    # Wrap the optimizer for multi-GPU support
    optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)


    logits = model(image, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(params['learning_rate'], 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy[1])

    # TODO extract 'train_op'

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy':
                tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)),
        })

def main():

  data_dir = '/tmp/mnist'
  model_dir = '/tmp/model'
  batch_size = 128
  train_epochs_before_evals = 1

  delete_dir(model_dir)

  data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  params = {
      'data_format': data_format,
      'learning_rate': 1e-4
  }

  # MULTI_GPU
  validate_batch_size_for_multi_gpu(batch_size)
  model_function = tf.contrib.estimator.replicate_model_fn(model_fn, loss_reduction=tf.losses.Reduction.MEAN)

  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=model_dir,
      params=params)

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

  print('Train model for %d epoch(s)' % train_epochs_before_evals)
  train_hooks = [tf.train.LoggingTensorHook(tensors=['learning_rate', 'cross_entropy', 'train_accuracy'], every_n_iter=100)]
  mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)

  print('Evaluate model')
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print('Eval loss: %s' % eval_results['loss'])
  print('Eval accuracy: %s' % eval_results['accuracy'])


if __name__ == '__main__':
  main()
