"""Demonstration of the metric function tf.metrics.accuracy"""

import numpy as np
import tensorflow as tf

logits = tf.placeholder(tf.float32, [None, 3])
labels = tf.placeholder(tf.int64, [None])

# Metric functions are used to calculate scores during model evaluation 
# when using the Estimator API.

# A metric function returns a tensor that represents the metric's value and
# an operation that is called to update the metric.
acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

with tf.Session() as sess:
    
    # The accuracy function creates two local variables `total` and `count`,
    # that must be initialized.
    sess.run(tf.local_variables_initializer())

    print('Initial accuracy: %f' % sess.run(acc))

    # First batch...
    logits_val = np.array([
        [0.1, 0.5, 0.4],
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.2]])
    labels_val = np.array([1,0,2]) # right, right, wrong

    sess.run(acc_op, {logits: logits_val, labels: labels_val})
    print('Accuracy after first batch: %f' % sess.run(acc))

    # Second batch...
    logits_val = np.array([
        [0.1, 0.5, 0.4],
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.2]])
    labels_val = np.array([0,0,2]) # wrong, right, wrong

    sess.run(acc_op, {logits: logits_val, labels: labels_val})
    print('Accuracy after second batch: %f' % sess.run(acc))
