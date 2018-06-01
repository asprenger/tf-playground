"""A MNIST classifier using the Estimator API. The model is build using Keras."""

import numpy as np
import tensorflow as tf
import dataset
from utils import delete_dir

tf.logging.set_verbosity(tf.logging.INFO)


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

  


def build_model_old(data_format):
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

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])

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