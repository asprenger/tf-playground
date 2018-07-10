'''A distributed MNIST classifier using the TensorFlow low-level API.'''

import time
import argparse
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training import session_run_hook

class MyHook(session_run_hook.SessionRunHook):
  def end(self, session):
    print('GOOD BUY')

FLAGS = None

def build_model(x):

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("model"):

        conv1 = layers.convolution2d(x_image,
                    num_outputs=32,
                    kernel_size=5,
                    stride=1,
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    scope='conv1')

        pool1 = layers.max_pool2d(
            inputs=conv1,
            kernel_size=2,
            stride=2,
            padding='SAME',
            scope='pool1')

        conv2 = layers.convolution2d(pool1,
                    num_outputs=64,
                    kernel_size=5,
                    stride=1,
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    scope='conv2')

        pool2 = layers.max_pool2d(
            inputs=conv2,
            kernel_size=2,
            stride=2,
            padding='SAME',
            scope='pool2')

        flattened = layers.flatten(pool2)

        fc1 = layers.fully_connected(flattened, 
            1024, 
            activation_fn=tf.nn.relu, 
            scope='fc1')

        keep_prob = tf.placeholder(tf.float32)
        drop1 = layers.dropout(
            fc1,
            keep_prob=keep_prob,
            scope='drop1')

        logits = layers.fully_connected(drop1, 
            10, 
            activation_fn=None, 
            scope='fc2')

        return logits, keep_prob


def main(_):

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name, # 'ps' or 'worker'
                           task_index=FLAGS.task_index)

  queues = []
  for i in range(len(worker_hosts)):
    print('Create queue%d' % i)
    queues.append(tf.FIFOQueue(1, tf.int32, shared_name="queue%d" % i))

  if FLAGS.job_name == "ps":
    server.join()

  elif FLAGS.job_name == "worker":

    # The device function assigns each node in the graph to a device.
    device_fn = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)

    with tf.device(device_fn):

      print('Loading dataset')
      mnist = input_data.read_data_sets(FLAGS.data_dir)
      print('%d train images' % mnist.train.num_examples)
      print('%d test images' % mnist.test.num_examples)

      x = tf.placeholder(tf.float32, [None, 784])
      y = tf.placeholder(tf.int64, [None])
      logits, keep_prob = build_model(x)

      global_step = tf.train.get_or_create_global_step()
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
      loss = tf.reduce_mean(cross_entropy)
      train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

      correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y), tf.float32)
      accuracy = tf.reduce_mean(correct_prediction)

      signal_termination_op = queues[FLAGS.task_index].enqueue(1)

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=500)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when 
    # done or an error occurs. 
    # `server.target` returns a URL for a tf.Session to connect to this server.
    # The master worker has the task_index 0.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir=FLAGS.log_dir,
                                           hooks=hooks) as mon_sess:

      step = 0
      while not mon_sess.should_stop():
        
        batch = mnist.train.next_batch(50)

        # Run a training step asynchronously.
        feed_dict = { x: batch[0], y: batch[1], keep_prob: 0.5 }
        mon_sess.run(train_op, feed_dict=feed_dict)

        # Calculate train accuracy
        if step % 50 == 0 and step > 0 and not mon_sess.should_stop():
          feed_dict = { x: batch[0], y: batch[1], keep_prob: 1.0 }
          train_accuracy = mon_sess.run(accuracy, feed_dict=feed_dict)
          gstep = tf.train.global_step(mon_sess, global_step)
          print('step %d: task_index=%d global_step=%s train_acc=%f' % (step, FLAGS.task_index, gstep, train_accuracy))

        step = step + 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")

  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )

  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )

  parser.add_argument(
      "--data_dir",
      type=str,
      default="/tmp/tensorflow/mnist",
      help="Directory for storing input data")

  parser.add_argument(
      "--log_dir",
      type=str,
      default="/tmp/train_logs",
      help="Directory for train logs")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
