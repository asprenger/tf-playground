"""
A MNIST classifier using the Keras API.
"""

from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

def build_model(hp):
    inputs = layers.Input((28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    logits = layers.Dense(10, activation='softmax', name='logits')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=logits)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', 
                                                            values=[1e-2, 1e-3, 1e-4])),
                  metrics=['accuracy'])

    return model


batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('x_train:', x_train.shape, x_train.dtype)
print('y_train:', y_train.shape, y_train.dtype)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Instantiate a tuner, specify the model-building function.
# Logs and checkpoints will be stored in folder /tmp/keras_runs/mnist_keras_tuned
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='/tmp/keras_runs',
    project_name='mnist_keras_tuned')

# Print a summary of the search space
tuner.search_space_summary()

# Start training the model with different hyper parameters
# tuner.search() has the same signature as model.fit()
tuner.search(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Print a summary of the results
tuner.results_summary()

# Fetch the best model and evaluate again
best_model = tuner.get_best_models(num_models=1)[0]
score = best_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
