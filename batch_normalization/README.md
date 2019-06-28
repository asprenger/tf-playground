# Batch normalization

Batch normalization as described in [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167) normalizes 
a mini-batch of train data by mean and variance. Batch Normalization allows us to use much higher learning rates 
(faster conversion) and be less careful about parameter initialization (more robust training).

TensorFlow provides the [tf.nn.moments](https://www.tensorflow.org/api_docs/python/tf/nn/moments) and 
[tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization) layers that are
supposed to be used together. There is also [tf.contrib.layers.batch_norm](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm)
that is probably the easiest to use.

The typical combination of fully-connected + batch_norm + relu can be implemented like this:

    def dense_batch_relu(x, hidden, is_training, scope):
        with tf.variable_scope(scope):
            fc = tf.contrib.layers.fully_connected(x, hidden, 
                                                   activation_fn=None,
                                                   scope='fc')
            bn = tf.contrib.layers.batch_norm(fc, center=True, scale=True, 
                                              is_training=is_training, scope='bn')
            return tf.nn.relu(bn, name='relu')

`is_training` is a boolean placeholder that is a binary indicator for whether we are in training mode. In training 
mode the layer accumulates the statistics of the moments (mean and variance) of the layer activation using an 
exponential moving average. When not in training mode the layer does not update the statistics.