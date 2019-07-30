# Keras Tuner

[Keras Tuner](https://github.com/keras-team/keras-tuner) is an hyperparameter tuner for tf.keras with TensorFlow 2.0.

Install TensorFlow 2.0 beta:

	pip install tensorflow==2.0.0b1

Install Keras Tuner:

	git clone https://github.com/keras-team/keras-tuner.git
	cd keras-tuner
	pip install .

`mnist_keras.py` contains a base version of a Keras MNIST classifier. 
`mnist_keras_tuned.py` is the base version modified with hyperparameter tuning.

The parameter sampler are defined in [https://github.com/keras-team/keras-tuner/blob/master/kerastuner/engine/hyperparameters.py](https://github.com/keras-team/keras-tuner/blob/master/kerastuner/engine/hyperparameters.py).
The search strategies are defined in [https://github.com/keras-team/keras-tuner/tree/master/kerastuner/tuners](https://github.com/keras-team/keras-tuner/tree/master/kerastuner/tuners).