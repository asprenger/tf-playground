
This examples are about non-distributed training on single GPU:

 * [mnist_basic.py](mnist_basic.py) - Build and train a classifier using the low-level API
 * [mnist_estimator.py](mnist_estimator.py) - Build and train a classifier using the Estimator API
 * [mnist_keras.py](mnist_keras.py) - Build a Keras model and train it with the Estimator API
 * [mnist_train_and_evaluate.py](mnist_train_and_evaluate.py) - Build and train a classifier using the train_and_evaluate() function

export TF_CONFIG='{
      "cluster": {
          "chief": ["127.0.0.1:2000"],
          "ps": ["127.0.0.1:3000"]
      },
      "task": {"type": "chief", "index": 0}
  }'


export TF_CONFIG='{
      "cluster": {
          "chief": ["127.0.0.1:2000"],
          "ps": ["127.0.0.1:3000"]
      },
      "task": {"type": "ps", "index": 0}
  }'  
