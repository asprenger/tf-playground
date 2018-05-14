'''Demonstration of the value and update operations for tf.metrics.accuracy'''

import numpy as np
import tensorflow as tf

logits = tf.placeholder(tf.float32, [None, 3])
labels = tf.placeholder(tf.int64, [None])

acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) # must init local vars for accuracy to work

    # Batch 1
    logits_val = np.array([
        [0.1, 0.5, 0.4],
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.2]])
    labels_val = np.array([1,0,2]) # right, right, wrong

    print( sess.run([acc, acc_op], {logits: logits_val, labels: labels_val}) )
    # => [0.0, 0.6666667]

    # Batch 2
    logits_val = np.array([
        [0.1, 0.5, 0.4],
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.2]])
    labels_val = np.array([0,0,2]) # wrong, right, wrong

    print( sess.run([acc, acc_op], {logits: logits_val, labels: labels_val}) )
    # => [0.6666667, 0.5]

    print( sess.run(acc) )
    # => 0.5
    # This returns the final correct value