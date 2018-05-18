'''
Builds simple CNN models on MNIST and uses RandomizedSearchCV and KerasClassifier
to tune the model parameters.
'''

from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


def shuffled_train_and_validation_indices(nb_examples, val_split = 0.2):
    indices = np.arange(nb_examples) # indices = (0,...,len(data) - 1)
    np.random.shuffle(indices)
    cutoff_index = int(nb_examples * val_split)
    val_indices = indices[:cutoff_index]
    train_indices = indices[cutoff_index:]
    return train_indices, val_indices

def build_model(dense_layer_sizes, filters, kernel_size, pool_size):
    '''Creates model comprised of 2 convolutional layers followed by dense layers

    dense_layer_sizes: List of layer sizes.
        This list has one number for each layer
    filters: Number of convolutional filters in each convolutional layer
    kernel_size: Convolutional kernel size
    pool_size: Size of pooling area for max pooling
    '''

    print("**** make_model(dense_layers={}, filters={}, kernel={}, pool={})".format(dense_layer_sizes, filters, kernel_size, pool_size))

    model = Sequential()
    model.add(Conv2D(filters, kernel_size,
                     padding='valid',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# load training data and do basic data normalization
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

train_indices, val_indices = shuffled_train_and_validation_indices(x_train.shape[0], val_split = 0.2)

# Labels must be one-hot encoded because we use categorical cross-entropy as loss 
# For training RandomizedSearchCV takes care of this, for testing we need to do it
# explicitly.
y_test = keras.utils.to_categorical(y_test, num_classes)

keras_classifier = KerasClassifier(build_model)

dense_size_candidates = [[32], [64]]
param_distributions={'dense_layer_sizes': dense_size_candidates,
    'filters': [8],
    'kernel_size': [3],
    'pool_size': [2],
    # epochs and batch_size are available for tuning even when not
    # an argument to model building function
    'epochs': [2],
    'batch_size': [16],
    }


# The 'scoring' parameter accepts a string or a custom Scorer. When set to 
# 'accuracy' a sklearn.metrics.scorer_PredictScorer instance parametrized
# with a sklearn.metrics.classification.accuracy_score function is used.
# _PredictScorer predicts labels for the validation dataset by calling
# Estimator.predict() and calls accuracy_score() to calculate the accuracy
# of the prediction.

validator = RandomizedSearchCV(keras_classifier,
                         param_distributions=param_distributions,
                         n_iter = 2,
                         scoring='accuracy',
                         verbose=5,
                         cv=[[train_indices, val_indices]],
                         n_jobs=1)

validator.fit(x_train, y_train)

print('The parameters of the best model are: ')
print(validator.best_params_)
print("Best validation accuracy: {}".format(validator.best_score_))

best_estimator = validator.best_estimator_ # estimator trained during refitting
best_model = best_estimator.model

# Calculate the test accuracy using the refitted model

metric_names = best_model.metrics_names
metric_values = best_model.evaluate(x_test, y_test)
test_metrics = dict(zip(metric_names, metric_values))
print("Test accuracy: {}".format(test_metrics['acc']))
print("Test loss: {}".format(test_metrics['loss']))
