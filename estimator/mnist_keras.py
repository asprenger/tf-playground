"""A MNIST classifier using the Estimator API. The model is build using Keras."""

import six
import numpy as np
import tensorflow as tf
import dataset
from utils import delete_dir

tf.logging.set_verbosity(tf.logging.INFO)

def build_model(data_format):
  """Build the model.
  Args:
    data_format: 'channels_first' or 'channels_last'.
  Returns:
    A tf.keras.Model
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28] # faster on GPUs
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1] # faster on CPUs

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
  
  image = features
  if isinstance(image, dict):
    image = features['image']


  model = build_model(params['data_format'])




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
    logits = model(image, training=True)    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]) 
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(params['learning_rate'], 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op)


  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
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


  train_hooks = [MyLoggingTensorHook(tensors=['learning_rate', 'cross_entropy', 'train_accuracy'], every_n_iter=100)]


  #train_hooks = [tf.train.LoggingTensorHook(tensors=['learning_rate', 'cross_entropy', 'train_accuracy'], every_n_iter=100)]
  mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)

  print('Evaluate model')
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print('Eval loss: %s' % eval_results['loss'])
  print('Eval accuracy: %s' % eval_results['accuracy'])


if __name__ == '__main__':
  main()  