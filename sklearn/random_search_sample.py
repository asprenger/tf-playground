import random
import numpy as np
import keras
from keras.datasets import mnist
from sklearn.grid_search import RandomizedSearchCV
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator

def generate_validation_split(nb_examples, val_split = 0.2, shuffle=True):
    indices = np.arange(nb_examples) # indices = (0,...,len(data) - 1)
    if shuffle:
        np.random.shuffle(indices)
    cutoff_index = int(nb_examples * val_split)
    val_indices = indices[:cutoff_index]
    train_indices = indices[cutoff_index:]
    return train_indices, val_indices

class MyEstimator(BaseEstimator):
    def __init__(self, hidden_size=100):
        self.hidden_size = hidden_size
    def fit(self, X_train, y_train):
        print('\nfit(X_train=%s, y_train=%s, hidden_size=%s)' % (str(X_train.shape), str(y_train.shape), str(self.hidden_size)))
    def predict(self, X):
        # called if 'RandomizedSearchCV(scoring='scoring_name')'
        print('(X=%s, hidden_size=%s)' % (str(X.shape), str(self.hidden_size)))
        return np.zeros((X.shape[0],))
    def score(self, x_test, y_test):
        # called if 'RandomizedSearchCV(scoring=None)'
        print('score(x_test=%s, y_test=%s, hidden_size=%s)' % (str(x_test.shape), str(y_test.shape), str(self.hidden_size)))
        score = random.random()
        print('score=%f' % score)
        return score

# load training data and do basic data normalization
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# split 60000 images into 2 sets of 48000 and 12000
train_indices, val_indices = generate_validation_split(x_train.shape[0], val_split = 0.2)


estimator = MyEstimator()
param_names = ','.join(estimator.get_params().keys())
print('Parameters: %s' % str(param_names))

cross_val_split = [[train_indices, val_indices]]
num_hidden_layer_candidates = [[32], [64], [128]]
param_distributions={
    'hidden_size': num_hidden_layer_candidates
}
sampling_iterations = 3

validator = RandomizedSearchCV(estimator,
                         param_distributions=param_distributions,
                         n_iter = sampling_iterations,
                         # see http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
                         #scoring='accuracy', 
                         verbose=0,
                         #cv=cross_val_split,
                         cv=2,
                         n_jobs=1)

validator.fit(x_train, y_train)

print('Best params: %s' % str(validator.best_params_))
print('Best score: %f' % validator.best_score_)
print('Refitted estimator: %s' % str(validator.best_estimator_))
